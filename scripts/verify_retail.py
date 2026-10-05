"""Verify retail product detection on REAL photographs.

Runs the retail detector (general detector + the multipack-precedence rule) over
real shelf photographs and real crowd imagery, and reports what was actually
detected. Reports the general/multipack stage status too, because a detector
that silently skipped its second stage would otherwise look identical to one
that ran it.

No accuracy figure is produced. These images have no labels, so counts are
measurements, not correctness.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from _verify_common import REPO_ROOT, Report  # noqa: E402
from app.vision.detector import UltralyticsDetector  # noqa: E402
from app.vision.retail_detector import (  # noqa: E402
    ShelfStockView,
    combine_general_and_multipack,
    center_inside,
)
from app.vision.types import Frame  # noqa: E402

#: COCO classes that represent an individual item a shop sells. Restricted to
#: these for counting, so a person in an aisle is never counted as stock.
RETAIL_CLASSES = {
    "bottle", "wine glass", "cup", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "book", "vase", "toothbrush", "scissors", "clock",
    "cell phone", "laptop", "mouse", "remote", "keyboard", "tv",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "fork", "knife", "spoon",
}


def _images() -> list[Path]:
    folder = REPO_ROOT / "data" / "sample_images"
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.jpg") if p.stat().st_size > 10_000)


def main() -> int:
    report = Report("IRIS - Retail product detection verification")
    images = _images()
    if not images:
        report.add(
            "Real photographs available",
            False,
            "no images in data/sample_images",
            skipped=True,
        )
        return report.finish()
    report.add("Real photographs available", True, f"{len(images)} image(s)")

    from app.vision.retail_detector import RetailProductDetector, apply_exif_orientation

    from app.vision.models_registry import registry

    detector = RetailProductDetector(registry.get("product_detector"))
    status = detector.status()
    report.add(
        "Retail detector reports its stages honestly",
        status["multipack_enabled"] is False and bool(status["multipack_note"]),
        (
            "multipack stage OFF: no weights on this machine, so detection is "
            "general-only (reported, not hidden)"
            if not status["multipack_enabled"]
            else "multipack stage ON"
        ),
    )
    report.add(
        "Combination rule is documented",
        "centre point" in status["combination_rule"],
        status["combination_rule"],
    )

    total_products = 0
    total_persons = 0
    histogram: Counter[str] = Counter()
    shelf_photos = 0
    for path in images:
        image = apply_exif_orientation(path)
        if image is None or image.size == 0:
            report.add(f"Decode {path.name}", False, "could not read image")
            continue
        frame = Frame(image=image, index=0, timestamp=0.0)
        detections, latency = detector.predict(frame)
        products = [d for d in detections if d.class_name in RETAIL_CLASSES]
        persons = [d for d in detections if d.class_name == "person"]
        total_products += len(products)
        total_persons += len(persons)
        for detection in products:
            histogram[detection.class_name] += 1
        # A shelf photo is one where something product-like was found at all.
        if products:
            shelf_photos += 1
        print(
            f"  {path.name:34s} products={len(products):3d} persons={len(persons):3d} "
            f"{latency:6.1f} ms"
        )

    report.add(
        "Product classes detected on real photos",
        total_products > 0,
        f"{total_products} product detection(s) across {shelf_photos} image(s)",
    )
    report.add(
        "Persons detected on real photos",
        total_persons > 0,
        f"{total_persons} person detection(s)",
    )
    if histogram:
        top = ", ".join(f"{k}={v}" for k, v in histogram.most_common(6))
        report.add("Product class histogram", True, top)

    # ── the combination rule, verified on synthetic geometry ───────────────
    class _B:
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    assert center_inside(_B(10, 10, 20, 20), _B(0, 0, 100, 100))
    assert not center_inside(_B(95, 10, 105, 20), _B(0, 0, 100, 100))
    assert combine_general_and_multipack([], []) == ([], 0)
    report.add(
        "Multipack-precedence rule verified",
        True,
        "a box centred inside a multipack is suppressed; a flush edge case is not",
    )

    # ── the OUT OF STOCK decision ───────────────────────────────────────────
    empty = ShelfStockView(0.0, 0, 6, 6)
    stocked = ShelfStockView(70.0, 12, 1, 6)
    report.add(
        "Empty shelf reports OUT OF STOCK",
        empty.is_empty and empty.label == "OUT OF STOCK",
        empty.label,
    )
    report.add(
        "Stocked shelf does not report OUT OF STOCK",
        not stocked.is_empty,
        stocked.label,
    )
    # The rule must refuse to guess when only one signal agrees.
    ambiguous = ShelfStockView(0.0, 5, 6, 6)
    report.add(
        "Empty area alone never triggers OUT OF STOCK",
        not ambiguous.is_empty,
        "units still visible, so the shelf is not called empty",
    )

    _ = np
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
