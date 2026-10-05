"""Train retail product detector on store shelf ground truth dataset.

Follows the training directives in docs/DATASET_GUIDE.md Section 8:
Loads configuration from YAML, fine-tunes Ultralytics YOLO11 on the
annotated shelf ground-truth dataset, records training metrics,
and saves weights to models/detection/retail_yolo11n/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.vision.models_registry import detect_hardware


def main():
    parser = argparse.ArgumentParser(description="Train IRIS retail shelf product detector.")
    parser.add_argument("--config", default="configs/detector.yaml", help="Path to dataset config YAML")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--weights", default="models/yolo11n.pt", help="Base model weights")
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    base_weights = REPO_ROOT / args.weights

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return 1

    if not base_weights.exists():
        print(f"Error: Base weights file not found at {base_weights}")
        return 1

    from ultralytics import YOLO

    hw = detect_hardware()
    device = "cuda" if hw.cuda_available else "cpu"

    print("=" * 60)
    print("IRIS Retail Product Detector Training")
    print("=" * 60)
    print(f"Dataset Config: {config_path}")
    print(f"Base Checkpoint: {base_weights}")
    print(f"Device: {device} (CUDA Available: {hw.cuda_available})")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}")

    model = YOLO(str(base_weights))

    project_dir = REPO_ROOT / "models" / "detection"
    run_name = "retail_yolo11n"

    started = time.perf_counter()
    results = model.train(
        data=str(config_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
        workers=2,
    )
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)
    print(f"Elapsed Time: {elapsed:.1f} s")
    print(f"Trained weights saved to: {project_dir / run_name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

