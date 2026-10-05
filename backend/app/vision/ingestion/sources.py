"""Concrete frame sources: video file, image, webcam, RTSP and web video links."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import SourceUnavailableError
from app.core.logging import get_logger
from app.vision.ingestion.base import FrameSource

camera_log = get_logger("camera")

# OpenCV/FFmpeg is noisy and sometimes prints banner text on stderr.
cv2.setNumThreads(1)


def open_capture(source: str | int, *, is_url: bool, timeout: float = 10.0) -> cv2.VideoCapture:
    """Open a VideoCapture with explicit validation and helpful errors."""
    if isinstance(source, int):
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise SourceUnavailableError(f"Cannot open capture device index {source}.")
        return capture

    if is_url:
        # RTSP over TCP avoids UDP loss on a LAN; fall back to the default if needed.
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise SourceUnavailableError(
                f"Cannot open network stream: {source}. "
                "Check the URL, credentials, port and that the camera is online."
            )
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout * 1000))
        return capture

    path = Path(source)
    if not path.exists():
        raise SourceUnavailableError(f"Media file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SourceUnavailableError(f"OpenCV could not decode the media file: {path.name}")
    return capture


def _read_fps(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return 0.0
    return float(fps)


class VideoFileSource(FrameSource):
    """A finite video file on disk (repeatable offline testing)."""

    source_type = "video_file"

    def __init__(self, path: str, *, target_fps: float = 5.0, frame_skip: int = 1, loop: bool = False) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self.path = Path(path)
        self.loop = loop
        self._capture: cv2.VideoCapture | None = None

    def is_finite_source(self) -> bool:
        return not self.loop

    def _open(self) -> None:
        self._capture = open_capture(str(self.path), is_url=False)
        self.status.source_fps = _read_fps(self._capture)
        self.status.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.status.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.status.source_fps <= 0:
            self.status.source_fps = 25.0
        if self.status.source_fps < self.target_fps:
            # Never request a rate the file cannot supply.
            self.target_fps = self.status.source_fps
        # Keep video time in step with wall-clock time (see
        # FrameSource.apply_realtime_pacing).
        self.apply_realtime_pacing()

    def _read(self) -> np.ndarray | None:
        if self._capture is None:
            raise SourceUnavailableError("Capture is closed.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if self.loop:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
            if not ok or frame is None:
                return None
        return frame

    def _advance(self) -> bool:
        """Skip one frame WITHOUT decoding it to BGR.

        ``cv2.grab`` advances the stream without the expensive colour
        conversion that ``retrieve``/``read`` perform. That is what makes 4K
        clips usable at all: decoding every frame of a 3840x2160 file costs far
        more than running the detector on the frames we keep.
        """
        if self._capture is None:
            raise SourceUnavailableError("Capture is closed.")
        if self._capture.grab():
            return True
        if self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return bool(self._capture.grab())
        return False

    def _close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class ImageSource(FrameSource):
    """A single still image.

    In ``loop`` mode the image is re-emitted at ``target_fps`` so it behaves
    like a static scene in Live Monitoring (useful for testing a camera
    position without hardware).  Otherwise the source ends after ``repeat``
    frames, which is what offline evaluation wants.
    """

    source_type = "image"

    def __init__(
        self,
        path: str,
        *,
        target_fps: float = 2.0,
        frame_skip: int = 1,
        repeat: int = 1,
        loop: bool = False,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self.path = Path(path)
        self.repeat = max(1, repeat)
        self.loop = loop
        self._image: np.ndarray | None = None
        self._emitted = 0

    def is_finite_source(self) -> bool:
        return not self.loop

    def _open(self) -> None:
        if not self.path.exists():
            raise SourceUnavailableError(f"Image not found: {self.path}")
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None:
            raise SourceUnavailableError(f"Could not decode image: {self.path.name}")
        self._image = image
        self._emitted = 0
        self.status.source_fps = 0.0

    def _read(self) -> np.ndarray | None:
        if self._image is None:
            return None
        if self._emitted >= self.repeat and not self.loop:
            return None
        self._emitted += 1
        return self._image.copy()

    def _close(self) -> None:
        self._image = None


class WebcamSource(FrameSource):
    """A local USB/built-in webcam by device index."""

    source_type = "webcam"

    def __init__(self, index: int = 0, *, target_fps: float = 5.0, frame_skip: int = 1) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self.index = index
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> None:
        self._capture = open_capture(self.index, is_url=False, timeout=5.0)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.status.source_fps = _read_fps(self._capture) or 30.0
        self.status.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.status.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _read(self) -> np.ndarray | None:
        if self._capture is None:
            raise SourceUnavailableError("Webcam capture is closed.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def _close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class RtspSource(FrameSource):
    """An RTSP / HTTP IP camera stream with bounded exponential backoff."""

    source_type = "rtsp"

    def __init__(
        self,
        url: str,
        *,
        target_fps: float = 5.0,
        frame_skip: int = 1,
        reconnect_delay: float | None = None,
        reconnect_max_delay: float | None = None,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self._url = url
        self._reconnect_delay = reconnect_delay or settings.rtsp_reconnect_delay
        self._reconnect_max_delay = reconnect_max_delay or settings.rtsp_reconnect_max_delay
        self._capture: cv2.VideoCapture | None = None

    @property
    def url(self) -> str:
        return self._url

    def backoff_bounds(self) -> tuple[float, float]:
        return self._reconnect_delay, self._reconnect_max_delay

    def _open(self) -> None:
        self._capture = open_capture(
            self._url, is_url=True, timeout=settings.rtsp_connect_timeout
        )
        self.status.source_fps = _read_fps(self._capture) or 25.0
        self.status.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.status.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.last_connected_at = time.time()

    def _read(self) -> np.ndarray | None:
        if self._capture is None:
            raise SourceUnavailableError("RTSP capture is closed.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            # A failed read from a live stream means the connection dropped.
            raise SourceUnavailableError("RTSP read returned no frame (connection lost).")
        return frame

    def _close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None


def normalise_ip_webcam_url(raw: str) -> str:
    """Return the base URL of a phone's IP Webcam server from loose input.

    The popular Android "IP Webcam" app exposes an MJPEG stream at
    ``http://<phone-ip>:8080/video`` and a snapshot at ``/shot.jpg``. Operators
    type the address in any of these forms, all meaning the same server::

        192.168.1.42:8080
        http://192.168.1.42:8080
        http://192.168.1.42:8080/video

    The result is always the bare ``scheme://host:port`` root; the stream path
    is appended by :class:`IpWebcamSource` so a mistyped ``/video`` suffix
    never produces ``/video/video``.
    """
    text = str(raw or "").strip()
    if not text:
        raise SourceUnavailableError(
            "An IP Webcam camera requires a stream_url (the phone's address, e.g. 192.168.1.42:8080)."
        )
    if "://" not in text:
        text = f"http://{text}"
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https"):
        raise SourceUnavailableError(
            f"An IP Webcam URL must start with http:// or https:// (got {parsed.scheme!r})."
        )
    if not parsed.hostname:
        raise SourceUnavailableError(f"Could not parse the IP Webcam address: {raw!r}")
    netloc = parsed.netloc
    if parsed.port is None:
        # The app's documented default port, so a bare IP still connects.
        netloc = f"{parsed.hostname}:{settings.ip_webcam_default_port}"
    return f"{parsed.scheme}://{netloc}"


class IpWebcamSource(FrameSource):
    """An Android phone camera streaming MJPEG over HTTP (the IP Webcam app).

    The MJPEG body is plain HTTP carrying concatenated JPEGs, so a tiny
    dedicated reader (stdlib ``urllib`` + JPEG marker scanning) is used instead
    of FFmpeg: connect/read failures then produce precise, actionable errors
    instead of codec-negotiation noise.

    A background thread continuously drains the socket and keeps only the
    NEWEST decoded frame. Draining matters: if the consumer read the socket
    synchronously at ``capture_fps`` while the phone sends 25+ fps, TCP
    backpressure would grow latency every second and the "live" feed would
    lag minutes behind reality. Freshest-frame-wins keeps it genuinely live.
    """

    source_type = "ip_webcam"
    #: The app serves the endless MJPEG stream here (snapshot is ``/shot.jpg``).
    stream_path = "/video"
    #: JPEG start/end-of-image markers (scan-based parsing, boundary-agnostic).
    _SOI = b"\xff\xd8\xff"
    _EOI = b"\xff\xd9"

    def __init__(
        self,
        address: str,
        *,
        target_fps: float = 5.0,
        frame_skip: int = 1,
        reconnect_delay: float | None = None,
        reconnect_max_delay: float | None = None,
        open_timeout: float | None = None,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self.address = address
        self._reconnect_delay = reconnect_delay or settings.rtsp_reconnect_delay
        self._reconnect_max_delay = reconnect_max_delay or settings.rtsp_reconnect_max_delay
        self._open_timeout = float(open_timeout or settings.rtsp_connect_timeout)
        self._read_timeout = max(2.0, self._open_timeout)
        self._base_url: str | None = None
        self._stream_url: str | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._frame_ready = threading.Event()
        self._latest: np.ndarray | None = None
        self._reader_error: Exception | None = None
        self._last_arrival = 0.0
        self._arrival_ema = 0.0

    @property
    def base_url(self) -> str | None:
        return self._base_url

    def backoff_bounds(self) -> tuple[float, float]:
        return self._reconnect_delay, self._reconnect_max_delay

    # ── reader thread ───────────────────────────────────────────────────────
    def _drain_stream(self) -> None:
        """Read the MJPEG body, decode frames, keep the newest. Never raises."""
        import urllib.error
        import urllib.request

        buffer = bytearray()
        try:
            request = urllib.request.Request(
                self._stream_url or "", headers={"User-Agent": "IRIS/1.0"}
            )
            with urllib.request.urlopen(request, timeout=self._open_timeout) as response:
                self.status.last_connected_at = time.time()
                # Only this source's own shutdown event ends the loop; the base
                # ``_stop`` flag is handled by the callers (see _open).
                while not self._shutdown.is_set():
                    chunk = response.read(4096)
                    if not chunk:
                        break  # stream ended cleanly
                    buffer.extend(chunk)
                    self._extract_jpegs(buffer)
        except Exception as exc:
            if not self._shutdown.is_set() and not self._stop.is_set():
                self._reader_error = exc
        finally:
            # Wake _open() immediately when the connection failed outright.
            self._frame_ready.set()

    def _extract_jpegs(self, buffer: bytearray) -> None:
        """Pop every complete JPEG out of the buffer and keep the newest."""
        while not self._shutdown.is_set():
            start = buffer.find(self._SOI)
            if start < 0:
                buffer.clear()
                return
            end = buffer.find(self._EOI, start + len(self._SOI))
            if end < 0:
                del buffer[:start]  # drop junk before the frame start
                return
            jpeg = bytes(buffer[start : end + len(self._EOI)])
            del buffer[: end + len(self._EOI)]
            self._accept(jpeg)

    def _accept(self, jpeg: bytes) -> None:
        import cv2

        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return  # a torn/partial frame is not fatal: the next one follows
        self._latest = frame
        self.status.width = int(frame.shape[1])
        self.status.height = int(frame.shape[0])
        self._measure_fps()
        self._frame_ready.set()

    def _measure_fps(self) -> None:
        """Estimate the phone's sending rate from inter-arrival times."""
        now = time.monotonic()
        if self._last_arrival > 0:
            delta = now - self._last_arrival
            if 0.001 < delta < 5.0:
                instant = 1.0 / delta
                self._arrival_ema = (
                    instant if self._arrival_ema <= 0 else 0.8 * self._arrival_ema + 0.2 * instant
                )
                self.status.source_fps = round(self._arrival_ema, 2)
        self._last_arrival = now

    def _friendly_error(self, exc: Exception) -> str:
        """Turn socket/HTTP failures into advice an operator can act on."""
        import urllib.error

        base = self._base_url or self.address
        if isinstance(exc, urllib.error.HTTPError):
            if exc.code == 404:
                return (
                    f"The server at {base} answered but {self.stream_path} was not "
                    "found. Start the IP Webcam app's video server on the phone "
                    "(tap 'Start server')."
                )
            return f"The phone camera server at {base} returned HTTP {exc.code}."
        return (
            f"Could not reach the phone camera at {base} ({type(exc).__name__}: {exc}). "
            "Check that the phone and this machine are on the same Wi-Fi network "
            "and that the IP Webcam server is running on the phone."
        )

    # ── FrameSource contract ────────────────────────────────────────────────
    def _open(self) -> None:
        self._base_url = normalise_ip_webcam_url(self.address)
        self._stream_url = f"{self._base_url}{self.stream_path}"
        # An explicit open() is also a request to run again: clearing ``_stop``
        # lets probe() re-open a source that was close()d earlier.
        self._stop.clear()
        self._shutdown.clear()
        self._frame_ready.clear()
        self._latest = None
        self._reader_error = None
        self._last_arrival = 0.0
        self._arrival_ema = 0.0
        self._thread = threading.Thread(
            target=self._drain_stream,
            name=f"iris-ipwebcam-{self._base_url.split('//')[-1]}",
            daemon=True,
        )
        self._thread.start()
        if not self._frame_ready.wait(self._open_timeout):
            self._close_reader()
            raise SourceUnavailableError(
                f"No frames received from the phone camera at {self._base_url} "
                f"within {self._open_timeout:.0f}s. Check the phone's IP and that "
                "the IP Webcam server is running."
            )
        if self._reader_error is not None:
            error, self._reader_error = self._reader_error, None
            self._close_reader()
            raise SourceUnavailableError(self._friendly_error(error))

    def _read(self) -> np.ndarray | None:
        if self._thread is None or not self._thread.is_alive():
            error = self._reader_error
            raise SourceUnavailableError(
                self._friendly_error(error) if error else "The phone camera connection is closed."
            )
        if not self._frame_ready.wait(self._read_timeout):
            raise SourceUnavailableError(
                f"No new frame from the phone camera at {self._base_url} within "
                f"{self._read_timeout:.0f}s; the connection looks dead."
            )
        frame = self._latest
        self._frame_ready.clear()
        if frame is None:
            raise SourceUnavailableError("The phone camera stream produced no decodable frame.")
        return frame

    def _close_reader(self) -> None:
        self._shutdown.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._latest = None

    def _close(self) -> None:
        self._close_reader()


#: Hosts whose watch page is not a media stream and must be resolved first.
_RESOLVABLE_HOSTS = (
    "youtube.com",
    "youtu.be",
    "www.youtube.com",
    "m.youtube.com",
    "vimeo.com",
    "dailymotion.com",
)


def is_resolvable_url(url: str) -> bool:
    """True when the URL is a web page/watch page rather than a media stream."""
    lowered = url.lower()
    return any(host in lowered for host in _RESOLVABLE_HOSTS)


def resolve_media_url(url: str, *, max_height: int = 720) -> str:
    """Resolve a watch-page URL to a direct, OpenCV-readable media URL.

    Uses yt-dlp, which is an optional dependency. When it is absent the caller
    gets an actionable error rather than a silent failure, because IRIS never
    pretends a source is working when it is not.
    """
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - depends on install
        raise SourceUnavailableError(
            "Reading a YouTube/web link needs the optional 'yt-dlp' package. "
            "Install it with: backend/.venv/Scripts/python.exe -m pip install yt-dlp"
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        # No "format" selector on purpose: letting yt-dlp pick can return
        # separate video-only and audio-only streams that only exist after a
        # merge, so there is no single URL for OpenCV to open. Ask for the
        # metadata and choose a progressive file ourselves below.
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except Exception as exc:
        raise SourceUnavailableError(
            f"Could not resolve that video link: {type(exc).__name__}: {exc}"
        ) from exc
    if not info:
        raise SourceUnavailableError("That video link resolved to no playable media.")

    direct = _pick_direct_url(info, max_height=max_height)
    if not direct:
        raise SourceUnavailableError("That video link has no directly playable stream.")
    return direct


def _pick_direct_url(info: dict, *, max_height: int) -> str | None:
    """Choose a single-file, playable media URL from yt-dlp's metadata.

    Must be PROGRESSIVE (video and audio in one file) and must actually carry
    video: the format list also contains storyboard images and video-only
    streams, neither of which OpenCV can play.
    """
    url = info.get("url")
    if url and info.get("vcodec") != "none":
        return str(url)
    for requested in info.get("requested_downloads") or []:
        nested = _pick_direct_url(requested, max_height=max_height)
        if nested:
            return nested

    def usable(formats: list[dict]) -> list[dict]:
        return [
            f
            for f in formats
            if f.get("url")
            and f.get("vcodec") not in (None, "none")
            and f.get("protocol") not in ("mhtml",)
        ]

    progressive = [
        f
        for f in usable(info.get("formats") or [])
        if f.get("acodec") not in (None, "none")
    ]
    candidates = progressive or usable(info.get("formats") or [])
    within_cap = [f for f in candidates if (f.get("height") or 0) <= max_height]
    candidates = within_cap or candidates
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: (
            1 if "mp4" in str(f.get("ext")) else 0,
            f.get("height") or 0,
            f.get("tbr") or 0,
        ),
        reverse=True,
    )
    return str(candidates[0]["url"])


class UrlVideoSource(FrameSource):
    """A video from a web link: YouTube, Vimeo, or any direct media URL.

    A watch page is not a media stream, so the URL is resolved to a direct one
    with yt-dlp on open, then read exactly like any other network capture. The
    original link is kept (encrypted at rest by the camera record) and the
    resolved URL is never persisted or logged, because signed CDN links carry
    temporary credentials.
    """

    source_type = "url"
    #: How many open failures before we stop retrying a permanently bad link.
    max_open_attempts = 3

    def __init__(
        self,
        url: str,
        *,
        target_fps: float = 5.0,
        frame_skip: int = 1,
        loop: bool = False,
        reconnect_delay: float | None = None,
        reconnect_max_delay: float | None = None,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self.page_url = url
        self.loop = loop
        self._reconnect_delay = reconnect_delay or settings.rtsp_reconnect_delay
        self._reconnect_max_delay = reconnect_max_delay or settings.rtsp_reconnect_max_delay
        self._capture: cv2.VideoCapture | None = None
        self._resolved: str | None = None
        self._attempts = 0

    def backoff_bounds(self) -> tuple[float, float]:
        return self._reconnect_delay, self._reconnect_max_delay

    def _open(self) -> None:
        self._attempts += 1
        if self._attempts > self.max_open_attempts:
            raise SourceUnavailableError(
                f"Gave up opening this video link after {self.max_open_attempts} "
                f"attempts: {self.page_url}"
            )
        media_url = (
            resolve_media_url(self.page_url) if is_resolvable_url(self.page_url)
            else self.page_url
        )
        self._resolved = media_url
        self._capture = open_capture(media_url, is_url=True, timeout=settings.rtsp_connect_timeout)
        self.status.source_fps = _read_fps(self._capture) or 25.0
        self.status.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.status.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.last_connected_at = time.time()
        self.apply_realtime_pacing()

    def _read(self) -> np.ndarray | None:
        if self._capture is None:
            raise SourceUnavailableError("Video-link capture is closed.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if self.loop:
                # Re-open rather than seek: a resolved CDN URL is often
                # range-limited, so restarting the request is more reliable.
                self._safe_close()
                self._attempts = 0
                self._open()
                ok, frame = self._capture.read() if self._capture else (False, None)
            if not ok or frame is None:
                raise SourceUnavailableError("Video-link read returned no frame (stream ended).")
        return frame

    def _close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None


def iter_video_frames(
    path: str, *, target_fps: float = 5.0, frame_skip: int = 1, limit: int | None = None
):
    """Convenience generator used by offline evaluation scripts."""
    source = VideoFileSource(path, target_fps=target_fps, frame_skip=frame_skip)
    count = 0
    for frame in source.frames():
        yield frame
        count += 1
        if limit is not None and count >= limit:
            break