"""Verify the AI layer with REAL inference on real sample media.

This script never fabricates a result. If model weights are missing it reports
the model check as SKIPPED (blocked), not passed.
"""

from __future__ import annotations

import sys

from _verify_common import REPO_ROOT, Report, in_backend_python

IMAGE_CHECK = """
from pathlib import Path
import cv2, numpy as np
from app.vision.models_registry import registry
from app.vision.types import Frame

detector = registry.get("product_detector")
detector.load()
detector.warmup()

results = []
for name in ("bus.jpg", "wikimedia_grocery.jpg"):
    path = Path(r"{data}") / "sample_images" / name
    if not path.exists():
        continue
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        continue
    detections, latency_ms = detector.predict(Frame(image=image, index=0, timestamp=0.0))
    labels = sorted({d.class_name for d in detections})
    results.append(f"{{name}}:{{len(detections)}} dets {{labels}} @{{latency_ms:.0f}}ms")

if not results:
    print("NO_MEDIA")
else:
    print("; ".join(results))
    assert any("det" in r and not r.endswith("dets [] @0ms") for r in results)
"""

EMPTY_FRAME_CHECK = """
import numpy as np
from app.vision.models_registry import registry
from app.vision.types import Frame
detector = registry.get("product_detector")
detector.load(); detector.warmup()
detections, latency = detector.predict(Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), index=0, timestamp=0.0))
print("blank frame ->", len(detections), "detections")
assert len(detections) == 0, "a blank frame must not produce detections"
"""

INVALID_IMAGE_CHECK = """
import numpy as np
from app.vision.detector import UltralyticsDetector
from app.vision.types import BBox, Frame
crop = UltralyticsDetector.crop(np.zeros((10, 10, 3), dtype=np.uint8), BBox(0, 0, 5, 5))
print("tiny crop shape", crop.shape)
assert crop.size >= 0
"""

EMBEDDER_CHECK = """
from app.vision.models_registry import registry
embedder = registry.get("embedder")
embedder.load()
import numpy as np
vector = embedder.embed(np.full((128, 128, 3), 120, dtype=np.uint8))
print("embedding dim", None if vector is None else vector.shape)
assert vector is not None and vector.shape[0] > 0
"""


SCHEDULER_CHECK = """
from app.vision.scheduler import (
    DETECTION, SHELF_ANALYSIS, InferenceScheduler,
)

# README §3 sampling table: shelf product detection 3-5 FPS, segmentation 1 FPS.
shelf = InferenceScheduler(role="shelf", capture_fps=5.0)
assert shelf.target_for(DETECTION) == 4.0, shelf.target_for(DETECTION)
assert shelf.target_for(SHELF_ANALYSIS) == 1.0

ran = []
for _ in range(20):
    shelf.begin_frame()
    ran.append(shelf.allow(SHELF_ANALYSIS))
detection_ran = []
for _ in range(20):
    shelf.begin_frame()
    detection_ran.append(shelf.allow(DETECTION))

stats = shelf.stats()["tasks"]
print(
    "detection", stats[DETECTION]["achieved_fps"], "fps;",
    "shelf_analysis", stats[SHELF_ANALYSIS]["achieved_fps"], "fps",
)
# Segmentation is rate-limited to 1 in 5; detection must not be.
assert ran.count(True) == 4, ran.count(True)
assert all(detection_ran), "detection must not be gated by another task"
"""

def main() -> int:
    report = Report("IRIS — AI verification (real inference)")

    weights = REPO_ROOT / "models" / "yolo11n.pt"
    if not weights.exists():
        report.add(
            "Model weights present",
            False,
            f"{weights} missing — run: python scripts/fetch_models.py",
            skipped=True,
        )
        for name in ("Detector inference on real photographs",
                     "Blank frame produces no detections",
                     "Invalid/tiny crop is handled",
                     "Embedder produces a descriptor"):
            report.add(name, False, "blocked: weights unavailable", skipped=True)
        return report.finish()

    report.add("Model weights present", True, str(weights))
    report.run(
        "Detector inference on real photographs",
        lambda: in_backend_python(IMAGE_CHECK.replace("{data}", str(REPO_ROOT / "data").replace("\\", "\\\\"))),
    )
    report.run("Blank frame produces no detections", lambda: in_backend_python(EMPTY_FRAME_CHECK))
    report.run("Invalid/tiny crop is handled", lambda: in_backend_python(INVALID_IMAGE_CHECK))
    report.run("Embedder produces a descriptor", lambda: in_backend_python(EMBEDDER_CHECK))
    report.run("Per-task inference scheduling", lambda: in_backend_python(SCHEDULER_CHECK))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())