"""Generate a DEMO video clip by animating a REAL photograph.

Why this exists
---------------
Customer analytics (line crossing, dwell time, occupancy) need *motion*. A
still photograph legitimately produces zero crossings — that is correct
behaviour, not a bug — so it cannot demonstrate or regression-test the footfall
path. This script produces a short clip in which a real photograph is panned
across the frame, so the pixels genuinely move and the real detector and real
tracker see genuinely moving people.

What the clip is
----------------
* The imagery is a **real photograph** (``data/sample_images/*.jpg``).
* The motion is a synthetic pan and a watermark.
* Frames are therefore watermarked ``DEMO`` and every derived detection is
  stored with ``source='demo'``.

Accuracy note: a panned photo is not a real camera, so footfall accuracy
measured on this clip means nothing operationally. It exists to exercise and
regression-test the pipeline.

Usage::

    python scripts/make_demo_video.py --out data/sample_videos/pan.mp4 --seconds 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_SOURCES = (
    REPO_ROOT / "data" / "sample_images" / "wikimedia_store_interior.jpg",
    REPO_ROOT / "data" / "sample_images" / "bus.jpg",
)


def render_clip(
    image_path: Path,
    out_path: Path,
    *,
    seconds: float = 12.0,
    fps: float = 12.0,
    zoom: float = 1.25,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read source image: {image_path}")
    height, width = image.shape[:2]

    # Crop a window that pans left->right across the photo, with a slow zoom in.
    crop_w = int(width / zoom)
    crop_h = int(height / zoom)
    crop_h = min(crop_h, height)
    total_frames = max(1, int(seconds * fps))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h)
    )
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {out_path}")

    for index in range(total_frames):
        t = index / max(1, total_frames - 1)
        # Ease-in-out horizontal pan.
        eased = 0.5 - 0.5 * np.cos(np.pi * t)
        x0 = int(eased * max(0, width - crop_w))
        zoom_now = 1.0 + (zoom - 1.0) * t
        ch = int(height / zoom_now)
        cw = int(width / zoom_now)
        ch, cw = min(ch, height), min(cw, width)
        x0 = min(x0, width - cw)
        y0 = max(0, (height - ch) // 2)
        frame = image[y0 : y0 + ch, x0 : x0 + cw].copy()
        frame = cv2.resize(frame, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        banner = frame.shape[0] // 12
        cv2.rectangle(frame, (0, 0), (crop_w, banner), (32, 32, 32), -1)
        cv2.putText(
            frame,
            f"DEMO MODE — panned still ({image_path.name})  frame {index + 1}/{total_frames}",
            (8, max(14, banner - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 215, 255), 1, cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    return {
        "output": str(out_path),
        "frames": total_frames,
        "fps": fps,
        "seconds": round(total_frames / fps, 2),
        "resolution": f"{crop_w}x{crop_h}",
        "source_image": image_path.name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a DEMO video from real photographs.")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "sample_videos" / "pan.mp4"))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--zoom", type=float, default=1.25)
    parser.add_argument("--source", default=None, help="Explicit source image path")
    args = parser.parse_args(argv)

    if args.source:
        sources = [Path(args.source)]
    else:
        sources = [p for p in DEFAULT_SOURCES if p.exists()]

    if not sources:
        print("No source images found. Run: python scripts/fetch_models.py", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    results = []
    for index, source in enumerate(sources):
        target = out_root if len(sources) == 1 else out_root.with_name(f"{out_root.stem}_{index}{out_root.suffix}")
        results.append(
            render_clip(source, target, seconds=args.seconds, fps=args.fps, zoom=args.zoom)
        )

    print("=" * 88)
    print("DEMO VIDEO RENDERED")
    print("=" * 88)
    for result in results:
        print(
            f"  {result['output']}\n"
            f"      {result['frames']} frames @ {result['fps']}fps "
            f"({result['seconds']}s) {result['resolution']} from {result['source_image']}"
        )
    print()
    print("These clips are for pipeline testing and demonstration only.")
    print("Footfall accuracy measured on a panned still is NOT an operational result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())