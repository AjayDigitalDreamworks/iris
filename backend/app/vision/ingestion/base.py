"""Frame source abstraction.

Every AI module consumes :class:`~app.vision.types.Frame` objects and never
knows whether they came from an RTSP camera, a webcam, a video file, an image
or the DEMO fixture generator.  Swapping a source never requires touching the
models.

Sources are *pull* based generators.  They are deliberately fault-isolated:
:class:`SourceUnavailableError` and driver errors are converted into a status
update plus a bounded exponential-backoff retry, never an unhandled crash.
"""

from __future__ import annotations

import abc
import queue
import threading
import time
from collections.abc import Iterator

import numpy as np

from app.core.errors import SourceUnavailableError
from app.core.logging import get_logger
from app.vision.types import Frame, SourceStatus

camera_log = get_logger("camera")


class FrameSource(abc.ABC):
    """A video source that yields frames."""

    source_type: str = "unknown"

    def __init__(self, *, target_fps: float = 5.0, frame_skip: int = 1) -> None:
        self.target_fps = max(0.1, target_fps)
        self.frame_skip = max(1, int(frame_skip))
        self.status = SourceStatus(source_type=self.source_type)
        self._stop = threading.Event()

    # ── lifecycle ───────────────────────────────────────────────────────────
    @abc.abstractmethod
    def _open(self) -> None:
        """Open the underlying resource. Raise on failure."""

    @abc.abstractmethod
    def _read(self) -> np.ndarray | None:
        """Return the next raw BGR frame, or None when the stream ended."""

    def _close(self) -> None:
        """Release the underlying resource. Must be safe to call twice."""

    def open(self) -> None:
        self._open()
        self.status.state = "running"
        self.status.started_at = time.time()
        self.status.message = "Connected"
        camera_log.info(
            "Source opened",
            extra={"source_type": self.source_type, "target_fps": self.target_fps},
        )

    def close(self) -> None:
        self._stop.set()
        try:
            self._close()
        except Exception as exc:  # pragma: no cover - defensive
            camera_log.warning("Error while closing source", exc_info=exc)
        self.status.state = "stopped"
        camera_log.info("Source closed", extra={"source_type": self.source_type})

    def stop(self) -> None:
        self._stop.is_set  # keep attribute access explicit for readers
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return not self._stop.is_set()

    # ── iteration ───────────────────────────────────────────────────────────
    def frames(self) -> Iterator[Frame]:
        """Yield sampled frames forever (or until the source ends/is stopped).

        Reconnection with bounded exponential backoff is handled here so every
        source type inherits the same reliability behaviour.
        """
        attempt = 0
        opened = False
        while not self._stop.is_set():
            if not opened:
                self.status.state = "connecting"
                try:
                    self.open()
                    opened = True
                    attempt = 0
                except Exception as exc:
                    attempt += 1
                    self.status.state = "error"
                    self.status.last_error = f"{type(exc).__name__}: {exc}"
                    self.status.message = "Reconnecting…"
                    camera_log.warning(
                        "Source open failed; will retry",
                        extra={
                            "source_type": self.source_type,
                            "attempt": attempt,
                            "error": self.status.last_error,
                        },
                    )
                    if not self._wait_backoff(attempt):
                        break
                    continue

            try:
                frame = self._next_frame()
            except Exception as exc:
                # A dead stream must never take the backend down.
                self.status.state = "error"
                self.status.last_error = f"{type(exc).__name__}: {exc}"
                self.status.message = "Stream error; reconnecting…"
                self.status.reconnect_count += 1
                camera_log.error(
                    "Stream read failed; reconnecting",
                    extra={
                        "source_type": self.source_type,
                        "reconnects": self.status.reconnect_count,
                        "error": self.status.last_error,
                    },
                )
                self._safe_close()
                opened = False
                attempt += 1
                if not self._wait_backoff(attempt):
                    break
                continue

            if frame is None:
                if self.is_finite_source():
                    self.status.state = "stopped"
                    self.status.message = "Source finished"
                    break
                self._safe_close()
                opened = False
                attempt += 1
                if not self._wait_backoff(attempt):
                    break
                continue

            attempt = 0
            self.status.frames_read += 1
            self.status.state = "running"
            self.status.last_frame_at = time.time()
            yield frame

        self._safe_close()

    def is_finite_source(self) -> bool:
        """True when the source ends naturally (files/images) vs. endless."""
        return False

    def _advance(self) -> bool:
        """Consume one frame without yielding it.

        Overridden by sources that can skip cheaply (video files grab without
        paying for colour conversion); the default decodes and discards.
        """
        return self._read() is not None

    def _next_frame(self) -> Frame | None:
        """Read a raw frame, apply frame_skip and capture-FPS throttling."""
        raw = self._read()
        # Advance through the skipped frames. This must be a bounded `for`: an
        # unbounded loop here spins forever once the source runs dry. A failed
        # skip keeps the frame we already have, so the tail of a clip is
        # delivered rather than dropped.
        for _ in range(self.frame_skip - 1):
            if not self._advance():
                break
        if raw is None:
            return None
        self.status.frames_yielded += 1
        self.status.width = int(raw.shape[1])
        self.status.height = int(raw.shape[0])
        frame = Frame(
            image=raw,
            index=self.status.frames_yielded,
            timestamp=time.time(),
            source_fps=self.status.source_fps,
            camera_id=None,
        )
        self._throttle()
        return frame

    def apply_realtime_pacing(self) -> None:
        """Advance a recorded source so video time tracks wall-clock time.

        A 30 fps clip analysed at 3 fps delivers 3 frames per second, but if
        each delivered frame is also only ``frame_skip`` source frames ahead,
        the clip advances 3 source-frames per second and plays back roughly 10x
        slower than real life — which looks like a broken "live" feed.

        So for prerecorded sources the skip is widened to
        ``source_fps / target_fps``: every delivered frame represents the amount
        of video that really elapsed in the time it took to produce it. Live
        sources (webcam, RTSP) do not call this — there is no file position to
        run ahead of, and the network is the clock.
        """
        source_fps = self.status.source_fps
        if source_fps <= 0 or self.target_fps <= 0:
            return
        needed = int(round(source_fps / self.target_fps))
        if needed > self.frame_skip:
            self.frame_skip = needed
        self.status.realtime = True
        self.status.realtime_ratio = round(
            (self.frame_skip * self.target_fps) / source_fps, 3
        )

    def _throttle(self) -> None:
        """Sleep so the delivered frame rate approaches ``target_fps``."""
        if self.target_fps <= 0:
            return
        interval = 1.0 / self.target_fps
        elapsed = time.time() - (self.status.last_frame_at or time.time())
        sleep_for = interval - elapsed
        if 0 < sleep_for < interval:
            time.sleep(sleep_for)
        now = time.time()
        window = 5.0
        if self.status.started_at and now - self.status.started_at >= window:
            self.status.delivered_fps = round(
                self.status.frames_yielded / (now - self.status.started_at), 2
            )

    def _wait_backoff(self, attempt: int) -> bool:
        """Bounded exponential backoff. Returns False when stopped."""
        base, maximum = self.backoff_bounds()
        delay = min(base * (2 ** max(0, attempt - 1)), maximum)
        camera_log.info(
            "Backing off before reconnect",
            extra={"source_type": self.source_type, "delay_s": round(delay, 2)},
        )
        return not self._stop.wait(delay)

    def backoff_bounds(self) -> tuple[float, float]:
        return 2.0, 60.0

    def _safe_close(self) -> None:
        try:
            self._close()
        except Exception:  # pragma: no cover - defensive
            pass

    # ── single-frame probing (used by the connection-test endpoint) ─────────
    def probe(self, max_frames: int = 5, timeout: float = 10.0) -> dict[str, object]:
        """Open, read a few frames, report and close. Never raises."""
        started = time.perf_counter()
        frames = 0
        try:
            self.open()
            deadline = started + timeout
            while frames < max_frames and time.perf_counter() < deadline:
                raw = self._read()
                if raw is None:
                    if self.is_finite_source():
                        break
                    time.sleep(0.05)
                    continue
                frames += 1
                self.status.width = int(raw.shape[1])
                self.status.height = int(raw.shape[0])
            latency_ms = (time.perf_counter() - started) * 1000
            if frames == 0:
                return {
                    "ok": False,
                    "message": "No readable frames were received.",
                    "error": self.status.last_error,
                    "latency_ms": round(latency_ms, 2),
                    "frames_read": 0,
                }
            return {
                "ok": True,
                "message": f"Read {frames} frame(s).",
                "latency_ms": round(latency_ms, 2),
                "frames_read": frames,
                "width": self.status.width,
                "height": self.status.height,
                "fps": self.status.source_fps or None,
            }
        except Exception as exc:
            camera_log.warning(
                "Source probe failed",
                extra={"source_type": self.source_type, "error": f"{type(exc).__name__}: {exc}"},
            )
            return {
                "ok": False,
                "message": "Could not open the source.",
                "error": f"{type(exc).__name__}: {exc}",
                "frames_read": frames,
            }
        finally:
            self._safe_close()
            self.status.state = "stopped"


class LatestFrameBuffer:
    """A bounded, always-freshest buffer with drop-stale semantics.

    When the consumer (inference) is slower than the producer we keep the
    NEWEST frame and drop the stale ones — freshness beats completeness for live
    retail monitoring, and it makes queue growth impossible.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self._latest: Frame | None = None
        self.dropped = 0
        self._closed = False

    def put(self, frame: Frame) -> bool:
        """Enqueue a frame, dropping the oldest if the buffer is full."""
        if self._closed:
            return False
        self._latest = frame
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait(frame)
                return True
            except (queue.Empty, queue.Full):  # pragma: no cover - race
                self.dropped += 1
                return False

    def get(self, timeout: float = 1.0) -> Frame | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self, timeout: float = 1.0) -> Frame | None:
        """Newest buffered frame, waiting up to ``timeout`` for the first one.

        Reading must NOT consume: the MJPEG stream, the snapshot endpoint and
        the health endpoint all read this buffer, and a destructive read lets
        the first caller starve the others (a snapshot taken while a stream is
        open returns 503 even though frames are arriving). So drain the queue
        only to learn the freshest frame, then always answer from a
        non-destructive copy of the newest frame seen.

        It must ALSO not keep waiting while frames are already flowing: draining
        the queue for the whole timeout window made every call take ~1 s, which
        capped the live stream at ~1 fps even with capture running at 25 fps.
        So drain only what is already queued (never blocking), and block just
        once — only when nothing has been seen yet.
        """
        while True:
            try:
                self._latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if self._latest is not None:
            return self._latest
        try:
            self._latest = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            pass
        return self._latest

    @property
    def latest(self) -> Frame | None:
        """Newest frame seen so far, without waiting for one. Never consumes."""
        return self._latest

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


def require_frame(source: FrameSource) -> FrameSource:
    """Type-narrowing helper that documents the SourceUnavailableError contract."""
    if not isinstance(source, FrameSource):
        raise SourceUnavailableError("Object is not a FrameSource.")
    return source