"""Export retail product detector to ONNX / TorchScript per DATASET_GUIDE.md Section 8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description="Export IRIS retail product detector.")
    parser.add_argument("--weights", default="models/yolo11n.pt", help="Weights to export")
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image resolution")
    args = parser.parse_args()

    weights_path = REPO_ROOT / args.weights
    if not weights_path.exists():
        print(f"Error: Weights file {weights_path} does not exist.")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    exported_path = model.export(format=args.format, imgsz=args.imgsz)
    print(f"\nSuccessfully exported model to {exported_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

