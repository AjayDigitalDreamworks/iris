"""Shelf occupancy and empty-space analysis.

**What this measures:** the fraction of a configured shelf region covered by
product-looking foreground, plus a slot-level count of visually empty positions.

**What this does NOT measure:** the true number of units on the shelf.  Empty
*area* is not empty *items*; two facings of a wide box and six facings of a
narrow box cover the same area.  README §4.2 gives the same warning.

The baseline implementation uses a running background model of the shelf plus
edge/texture energy, because no trained shelf-segmentation model exists yet
(``status='baseline'`` in the registry).  A trained U-Net/YOLO-seg can replace
:class:`SlotOccupancyEstimator` without changing its interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.logging import get_logger
from app.vision.tracker import polygon_area, scaled_polygon
from app.vision.types import BBox, Detection

ai_log = get_logger("ai")


@dataclass
class SlotOccupancy:
    index: int
    bbox: BBox
    occupied: bool
    fill_ratio: float
    detection_count: int = 0
    sku: str | None = None
    confidence: float = 0.0


@dataclass
class ShelfObservation:
    """The result of analysing one shelf region on one frame."""

    shelf_id: str | None
    zone_id: str | None
    camera_id: str | None
    occupancy_pct: float
    empty_positions: int
    total_positions: int
    detected_units: int
    area_coverage_pct: float
    confidence: float
    uncertain: bool = False
    slots: list[SlotOccupancy] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class SlotOccupancyEstimator:
    """Divides a shelf polygon into slots and estimates occupancy per slot."""

    def __init__(
        self,
        slots_per_row: int = 12,
        rows: int = 4,
        *,
        min_fill_ratio: float = 0.18,
        background_alpha: float = 0.02,
    ) -> None:
        self.slots_per_row = max(1, slots_per_row)
        self.rows = max(1, rows)
        self.min_fill_ratio = min_fill_ratio
        self.background_alpha = background_alpha
        self._background: np.ndarray | None = None
        self._initialized = False

    def reset(self) -> None:
        self._background = None
        self._initialized = False

    def build_slots(self, polygon: list[list[float]], width: int, height: int) -> list[BBox]:
        """Split the shelf polygon into a grid of normalised slot boxes."""
        if len(polygon) < 3:
            return []
        pts = np.array(polygon, dtype=np.float32) * np.array([width, height], dtype=np.float32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        box_width = (x_max - x_min) / self.slots_per_row
        box_height = (y_max - y_min) / self.rows
        slots: list[BBox] = []
        for row in range(self.rows):
            for col in range(self.slots_per_row):
                x1 = x_min + col * box_width
                y1 = y_min + row * box_height
                slots.append(BBox(float(x1), float(y1), float(x1 + box_width), float(y1 + box_height)))
        return slots

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """Foreground mask via a running background model."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if not self._initialized or self._background is None:
            self._background = gray.astype(np.float32)
            self._initialized = True
            # First frame: no reliable foreground yet.
            return np.zeros_like(gray, dtype=np.uint8)
        diff = cv2.absdiff(gray.astype(np.float32), self._background)
        # Textured/edge areas count as product presence as well as colour change.
        edges = cv2.Canny(gray, 60, 180)
        mask = ((diff > 22) | (edges > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Slowly update the background model (adapt to slow lighting changes).
        self._background = (
            1.0 - self.background_alpha
        ) * self._background + self.background_alpha * gray.astype(np.float32)
        return mask

    def analyse(
        self,
        image: np.ndarray,
        *,
        shelf_id: str | None,
        zone_id: str | None,
        camera_id: str | None,
        polygon_norm: list[list[float]],
        detections: list[Detection] | None = None,
        capacity_units: int = 0,
    ) -> ShelfObservation:
        """Estimate occupancy for one shelf region on one frame."""
        height, width = image.shape[:2]
        polygon_px = scaled_polygon(polygon_norm, width, height)
        notes: list[str] = []
        uncertain = False

        if len(polygon_px) < 3:
            return ShelfObservation(
                shelf_id=shelf_id, zone_id=zone_id, camera_id=camera_id,
                occupancy_pct=0.0, empty_positions=0, total_positions=0,
                detected_units=0, area_coverage_pct=0.0, confidence=0.0,
                uncertain=True, notes=["Shelf polygon is not configured."],
            )

        # _foreground_mask() sets _initialized on its very first call, so capture
        # the state before running it — otherwise the first observation can never
        # be reported as uncertain, which is exactly when it IS uncertain.
        first_observation = not self._initialized
        mask = self._foreground_mask(image)
        shelf_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(shelf_mask, [np.array(polygon_px, dtype=np.int32)], 255)
        shelf_area_px = polygon_area(polygon_px)
        shelf_mask_bool = shelf_mask > 0
        total_area = int(shelf_mask_bool.sum())
        if total_area == 0:
            return ShelfObservation(
                shelf_id=shelf_id, zone_id=zone_id, camera_id=camera_id,
                occupancy_pct=0.0, empty_positions=0, total_positions=0,
                detected_units=0, area_coverage_pct=0.0, confidence=0.0,
                uncertain=True, notes=["Shelf polygon covers no pixels."],
            )

        covered = int(((mask > 0) & shelf_mask_bool).sum())
        area_coverage_pct = 100.0 * covered / total_area

        slots = self.build_slots(polygon_norm, width, height)
        slot_results: list[SlotOccupancy] = []
        detections = detections or []
        for index, slot in enumerate(slots):
            x1, y1 = int(slot.x1), int(slot.y1)
            x2, y2 = int(min(width, slot.x2)), int(min(height, slot.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            slot_mask = mask[y1:y2, x1:x2]
            in_shelf = shelf_mask_bool[y1:y2, x1:x2]
            slot_area = int(in_shelf.sum())
            fill = float(((slot_mask > 0) & in_shelf).sum() / slot_area) if slot_area else 0.0

            hits = [
                d for d in detections
                if _box_in_slot(d.bbox, BBox(float(x1), float(y1), float(x2), float(y2)))
            ]
            best_sku = None
            confidence = 0.0
            for hit in hits:
                if hit.confidence > confidence:
                    confidence = hit.confidence
                    best_sku = hit.sku
            occupied = fill >= self.min_fill_ratio or bool(hits)
            slot_results.append(
                SlotOccupancy(
                    index=index, bbox=slot, occupied=occupied,
                    fill_ratio=round(fill, 4), detection_count=len(hits),
                    sku=best_sku, confidence=round(confidence, 4),
                )
            )

        empty = sum(1 for s in slot_results if not s.occupied)
        total_slots = len(slot_results)
        occupancy_pct = (
            100.0 * (total_slots - empty) / total_slots if total_slots else 0.0
        )
        detected_units = sum(s.detection_count for s in slot_results)

        # Confidence blends how much of the frame the analyser could actually see
        # with how confident the individual detections were.
        visibility = min(1.0, shelf_area_px / max(1.0, width * height) * 6.0)
        mean_detection_conf = (
            float(np.mean([s.confidence for s in slot_results if s.confidence]))
            if any(s.confidence for s in slot_results)
            else 0.5
        )
        confidence = round(min(1.0, 0.5 * visibility + 0.5 * mean_detection_conf), 4)

        if first_observation:
            uncertain = True
            notes.append("First observation: background model not yet established.")
        if total_slots == 0:
            uncertain = True
            notes.append("No slots resolved from the shelf polygon.")

        return ShelfObservation(
            shelf_id=shelf_id, zone_id=zone_id, camera_id=camera_id,
            occupancy_pct=round(occupancy_pct, 3),
            empty_positions=empty, total_positions=total_slots,
            detected_units=detected_units,
            area_coverage_pct=round(area_coverage_pct, 3),
            confidence=confidence, uncertain=uncertain, slots=slot_results, notes=notes,
        )


def _box_in_slot(box: BBox, slot: BBox, overlap: float = 0.35) -> bool:
    """True when enough of a detection box falls inside a slot."""
    ix1, iy1 = max(box.x1, slot.x1), max(box.y1, slot.y1)
    ix2, iy2 = min(box.x2, slot.x2), min(box.y2, slot.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    box_area = box.area
    return bool(box_area) and (inter / box_area) >= overlap


def summarise_shelf_observations(observations: list[ShelfObservation]) -> ShelfObservation | None:
    """Median-pool a batch of per-frame observations into one shelf reading."""
    if not observations:
        return None
    occupancy = float(np.median([o.occupancy_pct for o in observations]))
    empty = int(np.median([o.empty_positions for o in observations]))
    total = int(np.median([o.total_positions for o in observations]))
    coverage = float(np.median([o.area_coverage_pct for o in observations]))
    confidence = float(np.median([o.confidence for o in observations]))
    units = int(round(float(np.median([o.detected_units for o in observations]))))
    return ShelfObservation(
        shelf_id=observations[0].shelf_id,
        zone_id=observations[0].zone_id,
        camera_id=observations[0].camera_id,
        occupancy_pct=round(occupancy, 3),
        empty_positions=empty, total_positions=total, detected_units=units,
        area_coverage_pct=round(coverage, 3), confidence=round(confidence, 4),
        uncertain=any(o.uncertain for o in observations),
        notes=sorted({n for o in observations for n in o.notes}),
    )