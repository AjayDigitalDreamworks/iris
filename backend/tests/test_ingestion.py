"""Frame-source tests: sampling, skipping, pacing and end-of-clip behaviour.

These cover a real bug found during verification: ``frame_skip > 1`` used an
unbounded ``while`` loop that never terminated once a finite source ran dry,
spinning the CPU forever instead of ending the stream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.errors import ValidationError
from app.vision.ingestion import (
    UrlVideoSource,
    build_source,
    is_resolvable_url,
)
from app.services.camera_manager import collapse_tracks
from app.vision.ingestion.base import LatestFrameBuffer
from app.vision.types import Frame

CLIP = Path(__file__).resolve().parents[2] / "data" / "sample_videos" / "pan_0.mp4"


class _CountingSource:
    """A finite in-memory source used to test the base-class read loop."""

    def __init__(self, frames: int = 10, frame_skip: int = 1, target_fps: float = 1000.0):
        self.total = frames
        self.frame_skip = frame_skip
        self.target_fps = target_fps
        self.reads = 0
        self.closed = False
        self.source_type = "counting"

    def is_finite_source(self) -> bool:
        return True

    def open(self) -> None:  # pragma: no cover - not used
        pass

    def close(self) -> None:
        self.closed = True

    def _close(self) -> None:
        self.closed = True

    def _open(self) -> None:
        pass

    def _read(self):
        self.reads += 1
        if self.reads > self.total:
            return None
        return np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("frame_skip", [1, 2, 3, 5])
def test_frame_skip_terminates_at_end_of_a_finite_source(frame_skip):
    """Regression: an unbounded skip loop used to spin forever on a dry source."""
    from app.vision.ingestion.base import FrameSource

    class Source(_CountingSource, FrameSource):
        def __init__(self, **kwargs):
            _CountingSource.__init__(self, **kwargs)
            FrameSource.__init__(
                self, target_fps=self.target_fps, frame_skip=self.frame_skip
            )

    source = Source(frames=6, frame_skip=frame_skip)
    delivered = list(source.frames())

    assert source.closed, "a finite source must release its capture when it ends"
    assert len(delivered) >= 1, "a skip configuration must still deliver frames"
    # Skipping must never read more raw frames than the clip contains.
    assert source.reads <= 6 + frame_skip


def test_frame_skip_reduces_reads():
    from app.vision.ingestion.base import FrameSource

    class Source(_CountingSource, FrameSource):
        def __init__(self, **kwargs):
            _CountingSource.__init__(self, **kwargs)
            FrameSource.__init__(
                self, target_fps=self.target_fps, frame_skip=self.frame_skip
            )

    every = Source(frames=12, frame_skip=1)
    every_frames = list(every.frames())

    skipping = Source(frames=12, frame_skip=4)
    skipping_frames = list(skipping.frames())

    assert len(skipping_frames) < len(every_frames), "frame_skip must reduce delivered frames"


def test_a_source_that_does_not_support_skipping_still_ends():
    """frame_skip must never keep a dead source alive."""
    from app.vision.ingestion.base import FrameSource

    class Source(_CountingSource, FrameSource):
        def __init__(self, **kwargs):
            _CountingSource.__init__(self, **kwargs)
            FrameSource.__init__(
                self, target_fps=self.target_fps, frame_skip=self.frame_skip
            )

    delivered = list(Source(frames=3, frame_skip=4).frames())
    assert len(delivered) >= 1


# ── Construction validation ──────────────────────────────────────────────────
def test_rtsp_requires_a_stream_url():
    with pytest.raises(ValidationError):
        build_source({"source_type": "rtsp"})


def test_synthetic_source_needs_no_external_media():
    # A webcam source is deliberately not opened here: with no device attached
    # the base class retries with bounded backoff, which would make this test
    # hang rather than fail. Camera-manager tests cover the failure path.
    source = build_source({"source_type": "synthetic"})
    try:
        assert source.source_type == "synthetic"
    finally:
        source.close()


def test_webcam_source_is_constructible_without_a_device():
    source = build_source({"source_type": "webcam", "device_index": 0})
    try:
        assert source.source_type == "webcam"
    finally:
        source.close()


def test_video_file_source_rejects_a_missing_file_at_construction():
    with pytest.raises(ValidationError):
        build_source({"source_type": "video_file", "file_path": "definitely/missing.mp4"})


@pytest.mark.skipif(not CLIP.exists(), reason="demo clip not generated")
def test_real_clip_yields_increasing_indices():
    source = build_source(
        {"source_type": "video_file", "file_path": str(CLIP), "capture_fps": 0}
    )
    try:
        frames = []
        for index, frame in enumerate(source.frames()):
            frames.append(frame)
            if index >= 5:
                break
        assert len(frames) >= 2
        assert [f.index for f in frames] == sorted(f.index for f in frames)
        assert frames[0].image.size > 0
    finally:
        source.close()

# ── latest-frame buffer ───────────────────────────────────────────────────
def _frame(index: int) -> Frame:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return Frame(image=image, index=index, timestamp=float(index))


def test_reading_the_latest_frame_does_not_consume_it():
    """Regression: MJPEG, snapshots and health all read the same buffer.

    ``get_latest`` used to drain the queue, so the first caller starved the
    others and ``/snapshot.jpg`` answered 503 while frames were arriving.
    """
    buffer = LatestFrameBuffer(maxsize=4)
    buffer.put(_frame(1))
    buffer.put(_frame(2))

    first = buffer.get_latest(timeout=0.2)
    second = buffer.get_latest(timeout=0.2)

    assert first is not None and second is not None
    assert second.index == first.index


def test_latest_frame_returns_the_freshest_frame():
    buffer = LatestFrameBuffer(maxsize=4)
    for index in range(3):
        buffer.put(_frame(index))
    frame = buffer.get_latest(timeout=0.2)
    assert frame is not None
    assert frame.index == 2


def test_latest_frame_on_an_empty_buffer_times_out():
    buffer = LatestFrameBuffer(maxsize=4)
    assert buffer.get_latest(timeout=0.05) is None


# ── buffered track collapse ───────────────────────────────────────────────
class _Track:
    """Stand-in for a vision track: only ``track_id`` and confidence matter."""

    def __init__(self, track_id: int, confidence: float = 0.5) -> None:
        self.track_id = track_id
        self.class_name = "person"
        self.confidence = confidence


def test_collapse_tracks_keeps_the_newest_sample_per_track():
    """Regression: repeated buffered tracks used to violate a UNIQUE constraint.

    One buffered track per frame means the same track id appears many times
    between two database writes; persisting them all raised
    ``UNIQUE constraint failed: tracks.camera_id, tracks.track_uid,
    tracks.external_id`` and rolled back the whole frame's analytics.
    """
    collapsed = collapse_tracks(
        [_Track(7, 0.40), _Track(8, 0.60), _Track(7, 0.90), _Track(7, 0.95)]
    )
    assert set(collapsed) == {7, 8}
    # Newest sample wins, and the observation count is not lost.
    assert collapsed[7][0].confidence == pytest.approx(0.95)
    assert collapsed[7][1] == 3
    assert collapsed[8][1] == 1


def test_collapse_tracks_on_an_empty_buffer():
    assert collapse_tracks([]) == {}


# ── realtime pacing ───────────────────────────────────────────────────────
class TestRealtimePacing:
    """A prerecorded clip must play at wall-clock speed, not in slow motion."""

    def test_frame_skip_widens_to_track_the_source_fps(self):
        source = build_source(
            {"source_type": "video_file", "file_path": str(CLIP), "capture_fps": 5.0}
        )
        try:
            source.open()
        except Exception as exc:  # pragma: no cover - clip unreadable
            pytest.skip(f"clip unreadable: {exc}")
        try:
            source_fps = source.status.source_fps
            assert source_fps > 0
            # Each delivered frame must represent the video that really elapsed
            # while it was produced, otherwise a 25 fps clip analysed at 5 fps
            # plays back 5x slow.
            assert source.frame_skip == pytest.approx(
                round(source_fps / source.target_fps), abs=1
            )
            assert source.status.realtime is True
            assert source.status.realtime_ratio == pytest.approx(1.0, abs=0.2)
        finally:
            source.close()

    def test_pacing_never_shrinks_an_explicit_frame_skip(self):
        source = build_source(
            {
                "source_type": "video_file",
                "file_path": str(CLIP),
                "capture_fps": 30.0,
                "frame_skip": 4,
            }
        )
        try:
            source.open()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"clip unreadable: {exc}")
        try:
            assert source.frame_skip >= 4
        finally:
            source.close()


# ── web video links ──────────────────────────────────────────────────────
class TestUrlVideoSource:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "http://m.youtube.com/watch?v=abc123",
            "https://vimeo.com/12345",
        ],
    )
    def test_watch_pages_are_recognised_as_needing_resolution(self, url):
        assert is_resolvable_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://cdn.example.com/clip.mp4",
            "rtsp://camera.local/stream",
            "https://example.com/video.m3u8",
        ],
    )
    def test_direct_media_urls_are_left_alone(self, url):
        assert is_resolvable_url(url) is False

    def test_build_source_requires_an_http_url(self):
        with pytest.raises(ValidationError):
            build_source({"source_type": "url", "stream_url": "rtsp://camera/stream"})
        with pytest.raises(ValidationError):
            build_source({"source_type": "url"})

    def test_build_source_returns_a_url_source(self):
        source = build_source(
            {
                "source_type": "url",
                "stream_url": "https://www.youtube.com/watch?v=abc123",
                "capture_fps": 5.0,
                "loop": True,
            }
        )
        assert isinstance(source, UrlVideoSource)
        assert source.source_type == "url"
        assert source.loop is True

    def test_youtube_alias_is_accepted(self):
        source = build_source(
            {"source_type": "youtube", "stream_url": "https://youtu.be/abc123"}
        )
        assert isinstance(source, UrlVideoSource)

    def test_a_bad_link_fails_with_an_actionable_error_not_a_silent_one(self):
        # A closed local port: fails immediately and deterministically, with no
        # dependency on the public internet.
        source = build_source(
            {"source_type": "url", "stream_url": "http://127.0.0.1:9/nope.mp4"}
        )
        result = source.probe(max_frames=1, timeout=5)
        assert result["ok"] is False
        assert result["error"]
        # Never report success for a source that produced nothing.
        assert result["frames_read"] == 0

    def test_pick_direct_url_rejects_video_only_and_storyboards(self):
        from app.vision.ingestion.sources import _pick_direct_url

        info = {
            "formats": [
                {"url": "https://x/story.jpg", "ext": "jpg", "vcodec": "none"},
                {"url": "https://x/video-only.mp4", "ext": "mp4", "vcodec": "vp9",
                 "acodec": "none", "height": 720},
                {"url": "https://x/progressive.mp4", "ext": "mp4", "height": 360,
                 "vcodec": "avc1", "acodec": "mp4a"},
            ]
        }
        chosen = _pick_direct_url(info, max_height=720)
        # A progressive file OpenCV can actually play, not the higher-res
        # video-only stream.
        assert chosen == "https://x/progressive.mp4"

    def test_pick_direct_url_prefers_mp4_within_the_height_cap(self):
        from app.vision.ingestion.sources import _pick_direct_url

        info = {
            "formats": [
                {"url": "https://x/high.webm", "ext": "webm", "height": 1080,
                 "vcodec": "vp9", "acodec": "opus"},
                {"url": "https://x/low.mp4", "ext": "mp4", "height": 480,
                 "vcodec": "avc1", "acodec": "mp4a"},
            ]
        }
        # 1080p is over the cap, so the 480p mp4 wins.
        assert _pick_direct_url(info, max_height=720) == "https://x/low.mp4"

    def test_pick_direct_url_returns_none_when_nothing_is_playable(self):
        from app.vision.ingestion.sources import _pick_direct_url

        assert _pick_direct_url({"formats": []}, max_height=720) is None


# ── IP Webcam (phone camera over MJPEG/HTTP) ─────────────────────────────
def _jpeg_bytes(width: int = 8, height: int = 6) -> bytes:
    import cv2

    image = np.full((height, width, 3), (0, 128, 255), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


class _MjpegHandler:
    """Serves an endless multipart MJPEG stream, like the IP Webcam app."""

    boundary = "irisframe"

    def make_server(self, port: int = 0):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        handler = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/video":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={handler.boundary}"
                )
                self.end_headers()
                try:
                    for _ in range(200):
                        jpeg = _jpeg_bytes()
                        part = (
                            f"--{handler.boundary}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.write(part)
                        self.wfile.flush()
                        import time as _time

                        _time.sleep(0.01)
                except Exception:
                    pass

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread


mjpeg_server = _MjpegHandler()


class TestIpWebcamSource:
    def test_normalises_loose_addresses(self):
        from app.vision.ingestion import normalise_ip_webcam_url as norm

        assert norm("192.168.1.42:8080") == "http://192.168.1.42:8080"
        assert norm("192.168.1.42") == "http://192.168.1.42:8080"
        assert norm("http://192.168.1.42:8080/video") == "http://192.168.1.42:8080"
        assert norm("https://phone.local:9911") == "https://phone.local:9911"

    def test_build_source_requires_a_stream_url(self):
        with pytest.raises(ValidationError):
            build_source({"source_type": "ip_webcam"})

    def test_reads_frames_from_a_real_mjpeg_stream(self):
        from app.vision.ingestion import IpWebcamSource

        server, thread = mjpeg_server.make_server()
        try:
            source = build_source(
                {
                    "source_type": "ip_webcam",
                    "stream_url": f"http://127.0.0.1:{server.server_address[1]}",
                    "capture_fps": 30.0,
                    "open_timeout": 5.0,
                }
            )
            assert isinstance(source, IpWebcamSource)
            frames = []
            try:
                for frame in source.frames():
                    frames.append(frame)
                    if len(frames) >= 3:
                        break
            finally:
                source.close()
            assert len(frames) == 3
            assert frames[0].image.shape == (6, 8, 3)
            assert source.status.width == 8 and source.status.height == 6
            # probe() (the /cameras/test endpoint path) must succeed too.
            probe = source.probe(max_frames=2, timeout=5.0)
            assert probe["ok"] is True
            assert probe["frames_read"] == 2
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_an_unreachable_phone_fails_with_an_actionable_error(self):
        source = build_source(
            {
                "source_type": "ip_webcam",
                "stream_url": "127.0.0.1:9",
                "open_timeout": 2.0,
            }
        )
        try:
            result = source.probe(max_frames=1, timeout=5.0)
            assert result["ok"] is False
            assert result["frames_read"] == 0
            assert "Could not reach the phone camera" in str(result["error"])
        finally:
            source.close()
