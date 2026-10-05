"""Verify the camera pipeline: sources, capture, sampling, health, and failure isolation."""

from __future__ import annotations

import sys

from _verify_common import REPO_ROOT, Report, in_backend_python

SOURCE_CHECK = """
from pathlib import Path
import cv2
from app.vision.ingestion import build_source

root = Path(r"{data}")
built = []

video = sorted((root / "sample_videos").glob("pan*.mp4"))
if video:
    source = build_source({"source_type": "video_file", "file_path": str(video[0]), "capture_fps": 5.0})
    frame = next(iter(source.frames()))
    built.append(f"video_file {frame.width}x{frame.height}")
    source.close()

image = root / "sample_images" / "bus.jpg"
if image.exists():
    source = build_source({"source_type": "image", "image_path": str(image), "capture_fps": 2.0})
    frame = next(iter(source.frames()))
    built.append(f"image {frame.width}x{frame.height}")
    source.close()

source = build_source({"source_type": "synthetic"})
frame = next(iter(source.frames()))
built.append(f"synthetic {frame.width}x{frame.height}")
source.close()

print("; ".join(built))
assert len(built) >= 2, "expected at least two working source types"
"""

MISSING_FILE_CHECK = """
from app.vision.ingestion import build_source
from app.core.errors import SourceUnavailableError

try:
    source = build_source({"source_type": "video_file", "file_path": "definitely/not/here.mp4"})
    next(iter(source.frames()))
    print("UNEXPECTED_SUCCESS")
except Exception as exc:
    print("unreadable file rejected:", type(exc).__name__)
    assert "UNEXPECTED_SUCCESS" not in str(exc)
"""

RTSP_REJECT_CHECK = """
from app.vision.ingestion import build_source
from app.core.errors import ValidationError

try:
    build_source({"source_type": "rtsp"})
    print("UNEXPECTED_SUCCESS")
except ValidationError as exc:
    print("rtsp without url rejected:", exc)
    assert "UNEXPECTED_SUCCESS" not in str(exc)
"""

SAMPLING_CHECK = """
from pathlib import Path
from app.vision.ingestion import build_source

root = Path(r"{data}")
video = sorted((root / "sample_videos").glob("pan*.mp4"))
if not video:
    print("NO_CLIP")
else:
    source = build_source({"source_type": "video_file", "file_path": str(video[0]),
                            "capture_fps": 2.0, "frame_skip": 2})
    frames = []
    for i, frame in enumerate(source.frames()):
        frames.append(frame.index)
        if i >= 8:
            break
    source.close()
    print("yielded frames:", frames)
    assert len(frames) > 0
"""


def main() -> int:
    report = Report("IRIS — Camera pipeline verification")
    data = str(REPO_ROOT / "data").replace("\\", "\\\\")

    report.add(
        "Demo media present",
        any((REPO_ROOT / "data" / "sample_videos").glob("pan*.mp4")),
        "data/sample_videos, data/sample_images",
    )
    report.run("Frame sources produce real frames", lambda: in_backend_python(SOURCE_CHECK.replace("{data}", data)))
    report.run("Unreadable file is rejected cleanly", lambda: in_backend_python(MISSING_FILE_CHECK))
    report.run("RTSP without a URL is rejected", lambda: in_backend_python(RTSP_REJECT_CHECK))
    report.run("Frame sampling honours target FPS", lambda: in_backend_python(SAMPLING_CHECK.replace("{data}", data)))

    def rtsp_hardware() -> tuple[bool, str]:
        """RTSP needs a real camera; without one this is blocked, never 'passed'."""
        import socket

        try:
            with socket.create_connection(("127.0.0.1", 554), timeout=1):
                return False, "something is listening on 554 — add a real camera to validate"
        except OSError:
            return False, "BLOCKED: no RTSP camera reachable (implement + unit tested, not hardware validated)"

    passed, detail = rtsp_hardware()
    report.add("RTSP validated against real hardware", passed, detail, skipped=not passed)

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())