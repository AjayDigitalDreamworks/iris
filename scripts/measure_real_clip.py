"""Measure what IRIS actually detects in a real video file.

This runs the real pipeline (real detector, real tracker, real recogniser) over
a real clip and prints only measured numbers: frames read, how many detections
and tracks each frame produced, the class histogram and the confidence
distribution. It deliberately reports NO accuracy figure, because accuracy
needs labelled ground truth that a random retail clip does not provide.

Usage::

    python scripts/measure_real_clip.py <video-path> [--max-frames N]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.camera_manager import camera_confidence, camera_manager  # noqa: E402
from app.services.pipeline import CameraConfig, PipelineRunner  # noqa: E402
from app.vision.ingestion import build_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a real video file")
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: no such file: {args.video}", file=sys.stderr)
        return 2

    config = CameraConfig(
        camera_id="measure",
        camera_name=args.video.name,
        store_id=None,
        role="other",
        source_type="video_file",
        is_demo=False,
        confidence_threshold=camera_confidence(),
    )
    recognizer = camera_manager.recognizer
    runner = PipelineRunner(config, recognizer=recognizer)

    source = build_source(
        {"source_type": "video_file", "file_path": str(args.video), "capture_fps": 0}
    )

    frames = 0
    people_per_frame: list[int] = []
    tracks_per_frame: list[int] = []
    classes: Counter[str] = Counter()
    confidences: list[float] = []
    skus: Counter[str] = Counter()
    linked = 0
    detections = 0
    latencies: list[float] = []
    try:
        for frame in source.frames():
            frames += 1
            result = runner.process(frame)
            people_per_frame.append(len(result.people))
            tracks_per_frame.append(len(result.tracks))
            latencies.append(result.latency_ms)
            for detection in result.detections:
                detections += 1
                classes[detection.class_name] += 1
                confidences.append(detection.confidence)
                if detection.sku:
                    skus[detection.sku] += 1
                if detection.track_id is not None:
                    linked += 1
            if frames >= args.max_frames:
                break
    finally:
        source.close()

    print(f"file                : {args.video}")
    print(f"frames processed    : {frames}")
    print(f"detections total    : {detections}")
    print(f"mean detections/frame: {detections / frames:.2f}" if frames else "no frames")
    print(f"mean persons/frame  : {statistics.fmean(people_per_frame):.2f}")
    print(f"max persons in frame : {max(people_per_frame)}")
    print(f"mean tracks/frame   : {statistics.fmean(tracks_per_frame):.2f}")
    print(f"frames with a track : {sum(1 for n in tracks_per_frame if n)}/{frames}")
    print(f"detections linked to a track: {linked}/{detections}")
    print(f"mean latency ms     : {statistics.fmean(latencies):.1f}")
    print(f"max latency ms      : {max(latencies):.1f}")
    print("class histogram     :")
    for name, count in classes.most_common():
        share = 100.0 * count / detections if detections else 0.0
        print(f"    {name:20s} {count:6d}  ({share:5.1f}%)")
    if confidences:
        ordered = sorted(confidences)
        print(
            "confidence          : "
            f"min={ordered[0]:.3f} median={statistics.median(ordered):.3f} "
            f"p95={ordered[int(0.95 * (len(ordered) - 1))]:.3f} max={ordered[-1]:.3f}"
        )
    print(f"SKU matches         : {dict(skus) if skus else 'none (catalog has no reference image for these items)'}")
    print(
        "\nNOTE: these are raw detection counts, not accuracy. No accuracy claim is\n"
        "made here because this clip has no labelled ground truth."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
