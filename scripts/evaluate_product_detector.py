"""Evaluate retail product detector per docs/DATASET_GUIDE.md Section 8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.vision.models_registry import detect_hardware


def main():
    parser = argparse.ArgumentParser(description="Evaluate IRIS retail product detector.")
    parser.add_argument("--config", default="configs/detector.yaml", help="Path to dataset config YAML")
    parser.add_argument("--weights", default="models/detection/retail_yolo11n/weights/best.pt", help="Weights to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    weights_path = REPO_ROOT / args.weights

    if not weights_path.exists():
        # Fallback to base weights if fine-tuned weights have not been trained yet
        weights_path = REPO_ROOT / "models" / "yolo11n.pt"
        print(f"Fine-tuned weights not found. Evaluating base checkpoint: {weights_path}")

    from ultralytics import YOLO

    hw = detect_hardware()
    device = "cuda" if hw.cuda_available else "cpu"

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(config_path),
        imgsz=args.imgsz,
        device=device,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

