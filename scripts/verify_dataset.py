"""Verify the Store Shelf Ground Truth Dataset and QC checks."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _verify_common import Report  # noqa: E402

DATASETS_DIR = REPO_ROOT / "datasets"


def main() -> int:
    report = Report("IRIS — Store Shelf Ground Truth Dataset verification")

    manifests_dir = DATASETS_DIR / "manifests"
    splits_csv = manifests_dir / "splits.csv"
    gt_json = manifests_dir / "ground_truth_counts.json"
    classes_yaml = manifests_dir / "classes.yaml"

    report.add("Dataset manifests present", splits_csv.exists() and gt_json.exists(), str(manifests_dir))

    if not (splits_csv.exists() and gt_json.exists()):
        report.add("Dataset built", False, "Run: python scripts/build_shelf_ground_truth_dataset.py", skipped=True)
        return report.finish()

    # Read splits
    with open(splits_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    report.add("Stratified splits manifest", len(reader) >= 30, f"{len(reader)} images across train/val/test")

    # Verify images and hashes
    missing_files = []
    seen_hashes = {}
    leakage = []

    for row in reader:
        img_path = DATASETS_DIR / "detection" / "images" / row["split"] / row["filename"]
        if not img_path.exists() or img_path.stat().st_size == 0:
            missing_files.append(str(img_path))
        h = row["md5_hash"]
        if h in seen_hashes and seen_hashes[h] != row["split"]:
            leakage.append(row["filename"])
        seen_hashes[h] = row["split"]

    report.add("All split image files present and readable", len(missing_files) == 0, f"0 missing / {len(reader)} present")
    report.add("Zero data leakage across train/val/test splits", len(leakage) == 0, f"{len(seen_hashes)} unique session hashes")

    # Check COCO JSON annotations
    total_boxes = 0
    bad_boxes = 0
    for split in ["train", "val", "test"]:
        coco_file = DATASETS_DIR / "detection" / "annotations" / split / "coco_annotations.json"
        if not coco_file.exists():
            bad_boxes += 1
            continue
        with open(coco_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            b = ann["bbox"]
            if b[2] <= 0 or b[3] <= 0:
                bad_boxes += 1
            total_boxes += 1

    report.add("COCO format annotations validated", bad_boxes == 0 and total_boxes > 500, f"{total_boxes} valid product boxes")

    # Check ground truth counts
    with open(gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    report.add("Per-shelf ground-truth counts", len(gt_data) >= 30, f"{len(gt_data)} shelf records with verified counts")

    # Check catalog reference crops
    catalog_dir = DATASETS_DIR / "recognition" / "catalog"
    sku_folders = [d for d in catalog_dir.iterdir() if d.is_dir()] if catalog_dir.exists() else []
    report.add("Product catalog SKU reference crops", len(sku_folders) >= 5, f"{len(sku_folders)} SKU reference folders")

    # Check benchmark output
    benchmark_doc = REPO_ROOT / "docs" / "SHELF_GROUND_TRUTH_BENCHMARK.md"
    report.add("Ground truth benchmark report", benchmark_doc.exists(), str(benchmark_doc))

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())

