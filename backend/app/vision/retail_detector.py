"""Retail product detection for dense shelf images.

This ports the approach from the `Retail-product-detection` project found in
`Retail-product-detection-main/`, whose README documents the problem it solves:

> The general detector outputs bounding boxes for all products including those
> inside a multipack. The multipack detector outputs multipack boxes but misses
> many single packs. Therefore, bounding box coordinates from the 2 detectors
> are combined with the Multipack detector taking precedence.

The rule is precise and worth stating exactly, because "multipack takes
precedence" is ambiguous on its own. A general box is **discarded when its
centre point falls strictly inside a multipack box**. Surviving general boxes
are then concatenated after the multipack boxes.

**What could and could not be taken from that repository.** The notebooks
document the two training runs (a YOLOv5L fine-tuned on SKU-110K for general
detection, and a second detector trained on a private multipack dataset), but
the trained weight files live on the original author's Google Drive and are not
present in the folder. So this module ships the *algorithm* — which is the
reusable part — and drives it with models that actually exist on this machine.
The multipack stage is optional: without multipack weights the detector runs
general-only and reports that it is doing so, rather than silently pretending
the second stage ran.

The repository's reported mAP@0.5 of 89% belongs to *their* fine-tune on their
data. It is not a property of this system and is never claimed for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.core.config import settings
from app.core.errors import SourceUnavailableError
from app.core.logging import get_logger
from app.vision.types import BBox, Detection, Frame

ai_log = get_logger("ai")

#: Shelf images photographed on phones carry an EXIF orientation tag. OpenCV's
#: YOLO path has historically ignored it, so a portrait shelf arrives sideways
#: and the detector returns nonsense. See `apply_exif_orientation`.
EXIF_SOURCE = "Retail-product-detection-main (multipack-precedence combination)"


@dataclass
class DetectorStage:
    """One detector in the chain, plus what it is allowed to claim."""

    name: str
    detector: Any
    enabled: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        self.frames = 0
        self.detections = 0
        self.suppressed = 0


def apply_exif_orientation(image_path: str | Path) -> "np.ndarray | None":  # noqa: F821
    """Load an image with its EXIF orientation already applied.

    Ported from the `rotate_jpeg` helper in the reference repository, which
    exists for exactly this reason. PIL does the transpose natively, so no
    `piexif` dependency is needed.

    Returns BGR (OpenCV order), matching every other image path in IRIS.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    path = Path(image_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as handle:
            fixed = ImageOps.exif_transpose(handle)
            rgb = np.asarray(fixed.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pragma: no cover - corrupt file
        ai_log.warning(
            "Could not read image with EXIF correction",
            extra={"filename": path.name, "error": f"{type(exc).__name__}: {exc}"},
        )
        return None


def center_inside(inner: BBox, outer: BBox) -> bool:
    """True when ``inner``'s centre lies strictly inside ``outer``.

    Strict comparisons, matching the reference implementation
    (``row_m['xmin'] < cx and row_m['xmax'] > cx``). A centre exactly on a
    multipack edge is NOT suppressed, so a single pack standing flush against a
    multipack is still counted.
    """
    cx = inner.x1 + (inner.x2 - inner.x1) / 2.0
    cy = inner.y1 + (inner.y2 - inner.y1) / 2.0
    return outer.x1 < cx < outer.x2 and outer.y1 < cy < outer.y2


def combine_general_and_multipack(
    general: Sequence[Detection],
    multipack: Sequence[Detection],
) -> tuple[list[Detection], int]:
    """Apply the multipack-precedence rule.

    Returns ``(final_detections, suppressed_count)``. Multipack boxes come
    first, then the surviving general boxes, matching the reference notebook.
    """
    if not multipack:
        return list(general), 0
    multipack_boxes = [d.bbox for d in multipack]
    surviving = [
        detection
        for detection in general
        if not any(center_inside(detection.bbox, box) for box in multipack_boxes)
    ]
    suppressed = len(general) - len(surviving)
    return list(multipack) + surviving, suppressed


class RetailProductDetector:
    """General product detector, optionally chained with a multipack detector.

    ``general`` is any object exposing ``predict(frame) -> (detections,
    latency_ms)`` — in IRIS that is :class:`UltralyticsDetector`.

    The multipack stage is constructed from ``settings.retail_multipack_weights``
    when that file exists. When it does not, the detector runs general-only and
    ``stages`` makes that visible instead of hiding it.
    """

    def __init__(
        self,
        general: Any,
        *,
        multipack: Any = None,
        min_confidence: float | None = None,
    ) -> None:
        self.general = general
        self.min_confidence = (
            min_confidence if min_confidence is not None else settings.confidence_threshold
        )
        multipack_weights = settings.retail_multipack_weights
        available = bool(multipack_weights and Path(multipack_weights).exists())
        if multipack is not None:
            self.multipack = multipack
        elif available:
            self.multipack = self._load(multipack_weights)
        else:
            self.multipack = None

        self.stages: dict[str, DetectorStage] = {
            "general": DetectorStage(
                name="general",
                detector=general,
                notes="All individual products, including those inside multipacks.",
            )
        }
        if self.multipack is not None:
            self.stages["multipack"] = DetectorStage(
                name="multipack",
                detector=self.multipack,
                notes="Multipack boxes. Takes precedence over the general detector.",
            )
        else:
            ai_log.info(
                "Retail multipack stage disabled: no weights available",
                extra={"expected_path": multipack_weights or "(unset)"},
            )
        #: Frames processed and how the multipack rule behaved, for health output.
        self.frames = 0
        self.suppressed_total = 0

    @staticmethod
    def _load(weights: str) -> Any:
        """Load a standalone detector from a weights file.

        Built through the registry so it inherits the same hardware selection,
        image size and error handling as every other model - a second loading
        path would be a second set of bugs.
        """
        from app.vision.detector import UltralyticsDetector
        from app.vision.models_registry import ModelSpec, registry

        spec = ModelSpec(
            key="retail_multipack_detector",
            name="Multipack detector",
            task="detect",
            version="multipack",
            weights=weights,
            source=f"retail multipack weights ({weights})",
            provider="ultralytics",
            status="baseline",
            input_size=settings.detector_imgsz,
            confidence_threshold=settings.confidence_threshold,
        )
        try:
            return UltralyticsDetector(spec, weights)
        except SourceUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - bad weights file
            raise SourceUnavailableError(
                f"Multipack weights could not be loaded: {weights}: {exc}"
            ) from exc
        _ = registry

    # ── inference ──────────────────────────────────────────────────────────
    def predict(self, frame: Frame) -> tuple[list[Detection], float]:
        general, general_ms = self.general.predict(frame)
        general = [d for d in general if d.confidence >= self.min_confidence]

        total_ms = general_ms
        multipack: list[Detection] = []
        if self.multipack is not None:
            multipack, multipack_ms = self.multipack.predict(frame)
            multipack = [d for d in multipack if d.confidence >= self.min_confidence]
            total_ms += multipack_ms

        combined, suppressed = combine_general_and_multipack(general, multipack)
        for detection in combined:
            detection.extra["retail"] = {
                "source": EXIF_SOURCE,
                "multipack": detection.extra.get("retail", {}).get("multipack", False),
            }
        for detection in multipack:
            detection.extra["retail"]["multipack"] = True

        self.frames += 1
        self.suppressed_total += suppressed
        self.stages["general"].frames += 1
        self.stages["general"].detections += len(general)
        if "multipack" in self.stages:
            self.stages["multipack"].frames += 1
            self.stages["multipack"].detections += len(multipack)
            self.stages["multipack"].suppressed += suppressed
        return combined, total_ms

    # ── introspection ──────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """What this detector is, honestly. Used by the Models page."""
        return {
            "combination_rule": (
                "Multipack takes precedence: a general box is discarded when its "
                "centre point falls strictly inside a multipack box."
            ),
            "multipack_enabled": self.multipack is not None,
            "multipack_weights": settings.retail_multipack_weights or None,
            "multipack_note": (
                None
                if self.multipack is not None
                else (
                    "No multipack weights on this machine. The reference "
                    "repository keeps them on an external drive. Detection is "
                    "running general-only, so boxes inside multipacks are "
                    "counted individually rather than as one pack."
                )
            ),
            "frames": self.frames,
            "suppressed_by_multipack": self.suppressed_total,
            "stages": {
                name: {
                    "frames": stage.frames,
                    "detections": stage.detections,
                    "suppressed": stage.suppressed,
                    "notes": stage.notes,
                }
                for name, stage in self.stages.items()
            },
        }

    def model_id(self) -> str:
        base = getattr(self.general, "model_id", "detector")
        if self.multipack is None:
            return f"{base} (retail general-only)"
        return f"{base} + multipack"


@dataclass
class ShelfStockView:
    """Per-shelf view used by the OUT OF STOCK badge.

    Kept separate from the detector so the "empty shelf" decision is one small
    testable function rather than something buried in a page.
    """

    occupancy_pct: float
    detected_units: int
    empty_positions: int
    total_positions: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when the shelf shows no product at all.

        Requires BOTH no detected units and a negligible occupied area. Either
        signal alone is unreliable on its own: a dark shelf can read as empty
        area while units are still present, and a few missed detections can read
        as zero units on a full shelf.
        """
        return self.detected_units <= 0 and self.occupancy_pct < 5.0

    @property
    def label(self) -> str:
        return "OUT OF STOCK" if self.is_empty else f"{self.detected_units} unit(s)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "occupancy_pct": round(self.occupancy_pct, 2),
            "detected_units": self.detected_units,
            "empty_positions": self.empty_positions,
            "total_positions": self.total_positions,
            "is_empty": self.is_empty,
            "label": self.label,
        }


def shelf_stock_view(observation: Any) -> ShelfStockView:
    """Build a :class:`ShelfStockView` from a ``ShelfObservation``."""
    return ShelfStockView(
        occupancy_pct=float(getattr(observation, "occupancy_pct", 0.0)),
        detected_units=int(getattr(observation, "detected_units", 0)),
        empty_positions=int(getattr(observation, "empty_positions", 0)),
        total_positions=int(getattr(observation, "total_positions", 0)),
        notes=list(getattr(observation, "notes", []) or []),
    )
