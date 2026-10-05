"""Frame source implementations and the factory that builds them from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.errors import ValidationError
from app.vision.ingestion.base import FrameSource, LatestFrameBuffer
from app.vision.ingestion.sources import (
    ImageSource,
    IpWebcamSource,
    RtspSource,
    UrlVideoSource,
    VideoFileSource,
    WebcamSource,
    is_resolvable_url,
    iter_video_frames,
    normalise_ip_webcam_url,
    open_capture,
    resolve_media_url,
)
from app.vision.ingestion.synthetic import BrokenSource, SyntheticLayout, SyntheticSource

__all__ = [
    "FrameSource",
    "LatestFrameBuffer",
    "VideoFileSource",
    "ImageSource",
    "WebcamSource",
    "RtspSource",
    "IpWebcamSource",
    "UrlVideoSource",
    "is_resolvable_url",
    "resolve_media_url",
    "SyntheticSource",
    "SyntheticLayout",
    "BrokenSource",
    "iter_video_frames",
    "open_capture",
    "build_source",
    "parse_webcam_index",
    "normalise_ip_webcam_url",
]


def parse_webcam_index(raw: str | int | None) -> int:
    """Accept ``0``, ``"0"`` or ``"webcam:0"`` and return a device index."""
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.startswith("webcam:"):
        text = text.split(":", 1)[1]
    try:
        index = int(text)
    except ValueError as exc:
        raise ValidationError(f"Invalid webcam device index: {raw!r}") from exc
    if index < 0:
        raise ValidationError("Webcam device index must be >= 0.")
    return index


def build_source(spec: dict[str, Any]) -> FrameSource:
    """Construct a :class:`FrameSource` from a camera record.

    ``spec`` accepts the camera model's attributes (or a plain dict in tests)::

        {"source_type": "rtsp", "stream_url": "...", "capture_fps": 5, ...}
    """
    source_type = str(spec.get("source_type") or "").lower()
    target_fps = float(spec.get("capture_fps") or 5.0)
    frame_skip = int(spec.get("frame_skip") or 1)

    if source_type == "rtsp":
        url = spec.get("stream_url")
        if not url:
            raise ValidationError("An RTSP camera requires a stream_url.")
        return RtspSource(
            url,
            target_fps=target_fps,
            frame_skip=frame_skip,
            reconnect_delay=float(spec.get("reconnect_delay") or 5.0),
            reconnect_max_delay=float(spec.get("reconnect_max_delay") or 120.0),
        )

    if source_type in ("url", "youtube"):
        url = spec.get("stream_url")
        if not url:
            raise ValidationError(
                "A web-video camera requires a stream_url (a YouTube, Vimeo or "
                "direct media link)."
            )
        if not url.startswith(("http://", "https://")):
            raise ValidationError("A web-video stream_url must start with http:// or https://")
        return UrlVideoSource(
            url,
            target_fps=target_fps,
            frame_skip=frame_skip,
            loop=bool(spec.get("loop", False)),
            reconnect_delay=float(spec.get("reconnect_delay") or 5.0),
            reconnect_max_delay=float(spec.get("reconnect_max_delay") or 120.0),
        )

    if source_type == "webcam":
        return WebcamSource(
            parse_webcam_index(spec.get("device_index", 0)),
            target_fps=target_fps,
            frame_skip=frame_skip,
        )

    if source_type == "ip_webcam":
        url = spec.get("stream_url")
        if not url:
            raise ValidationError(
                "An IP Webcam camera requires a stream_url (the phone's address, "
                "e.g. 192.168.1.42:8080)."
            )
        return IpWebcamSource(
            url,
            target_fps=target_fps,
            frame_skip=frame_skip,
            reconnect_delay=float(spec.get("reconnect_delay") or 5.0),
            reconnect_max_delay=float(spec.get("reconnect_max_delay") or 120.0),
        )

    if source_type == "video_file":
        path = spec.get("file_path")
        if not path:
            raise ValidationError("A video-file camera requires a file_path.")
        resolved = Path(path)
        if not resolved.exists():
            raise ValidationError(f"Video file not found: {path}")
        return VideoFileSource(
            str(resolved), target_fps=target_fps, frame_skip=frame_skip,
            loop=bool(spec.get("loop", False)),
        )

    if source_type == "image":
        path = spec.get("image_path") or spec.get("file_path")
        if not path:
            raise ValidationError("An image camera requires an image_path.")
        resolved = Path(path)
        if not resolved.exists():
            raise ValidationError(f"Image not found: {path}")
        return ImageSource(
            str(resolved), target_fps=target_fps, frame_skip=frame_skip,
            repeat=int(spec.get("repeat", 1)),
            loop=bool(spec.get("loop", True)),
        )

    if source_type == "synthetic":
        # DEMO MODE
        return SyntheticSource(
            target_fps=target_fps,
            frame_skip=frame_skip,
            duration_seconds=float(spec.get("duration_seconds") or 0.0),
            speed=float(spec.get("speed") or 1.0),
        )

    raise ValidationError(f"Unsupported source_type: {source_type!r}")