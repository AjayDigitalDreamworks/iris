"""Visible shelf stock estimation.

Pipeline (README §4.3):

    detections + shelf capacity + planogram + recent history
        -> estimated visible stock + confidence

Three quantities are deliberately kept apart and never conflated:

``observed_quantity``   raw count of recognised detections right now
``estimated_quantity``  smoothed value used for business decisions
``registered_quantity`` the authoritative figure from POS/ERP/manual count

A camera estimate **never** overwrites the registered quantity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.vision.rules import StockState, StockThresholds, classify_stock, stock_percentage
from app.vision.shelf_analysis import ShelfObservation
from app.vision.types import Detection

ai_log = get_logger("ai")

# Detections at or below this confidence are not counted as a confirmed unit.
DEFAULT_MIN_UNIT_CONFIDENCE = 0.35


@dataclass
class StockEstimate:
    """One SKU's estimated visible stock on one shelf."""

    sku: str
    product_id: str | None
    product_name: str | None
    shelf_id: str | None
    camera_id: str | None
    observed_quantity: int
    estimated_quantity: float
    capacity_units: int
    stock_percentage: float
    confidence: float
    observation_count: int
    method: str
    occupancy_pct: float | None = None
    state: StockState = StockState.UNKNOWN
    notes: list[str] = field(default_factory=list)

    def as_payload(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "product_id": self.product_id,
            "observed_quantity": self.observed_quantity,
            "estimated_quantity": round(self.estimated_quantity, 3),
            "capacity_units": self.capacity_units,
            "stock_percentage": round(self.stock_percentage, 3),
            "confidence": round(self.confidence, 4),
            "observation_count": self.observation_count,
            "method": self.method,
            "state": self.state.value,
        }


class StockEstimator:
    """Turns per-frame detections into smoothed, confidence-aware estimates.

    Smoothing uses an exponential moving average with a bounded history so a
    single noisy frame cannot swing the estimate or spam an alert.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.4,
        history_size: int = 8,
        min_unit_confidence: float = DEFAULT_MIN_UNIT_CONFIDENCE,
    ) -> None:
        self.alpha = min(0.95, max(0.05, alpha))
        self.history_size = max(2, history_size)
        self.min_unit_confidence = min_unit_confidence
        self._history: dict[str, deque[float]] = {}
        self._smoothed: dict[str, float] = {}
        self._observations: dict[str, int] = {}

    def reset(self, sku: str | None = None) -> None:
        if sku is None:
            self._history.clear()
            self._smoothed.clear()
            self._observations.clear()
        else:
            self._history.pop(sku, None)
            self._smoothed.pop(sku, None)
            self._observations.pop(sku, None)

    # ── counting ────────────────────────────────────────────────────────────
    def count_detections(
        self, detections: list[Detection], *, sku: str | None = None
    ) -> tuple[int, float, list[str]]:
        """Count confident detections. Returns (count, mean_confidence, unknowns)."""
        selected = [d for d in detections if sku is None or d.sku == sku]
        confirmed = [
            d for d in selected if d.confidence >= self.min_unit_confidence and d.sku is not None
        ]
        unknown_count = sum(1 for d in selected if d.sku is None)
        if not confirmed:
            return 0, 0.0, (
                [f"{unknown_count} unrecognised object(s) not counted"] if unknown_count else []
            )
        return len(confirmed), float(sum(d.confidence for d in confirmed) / len(confirmed)), (
            [f"{unknown_count} unrecognised object(s) not counted"] if unknown_count else []
        )

    # ── estimation ──────────────────────────────────────────────────────────
    def estimate(
        self,
        *,
        sku: str,
        detections: list[Detection],
        capacity_units: int,
        product_id: str | None = None,
        product_name: str | None = None,
        shelf_id: str | None = None,
        camera_id: str | None = None,
        occupancy_pct: float | None = None,
        thresholds: StockThresholds | None = None,
        shelf_observation: ShelfObservation | None = None,
        product_capacity: int | None = None,
    ) -> StockEstimate:
        """Produce a smoothed stock estimate for one SKU."""
        notes: list[str] = []
        observed, mean_conf, detect_notes = self.count_detections(detections, sku=sku)
        notes.extend(detect_notes)

        history = self._history.setdefault(sku, deque(maxlen=self.history_size))
        history.append(float(observed))
        observation_count = self._observations.get(sku, 0) + 1
        self._observations[sku] = observation_count

        method = "detection_count"
        if shelf_observation is not None and shelf_observation.total_positions:
            method = "detection_count+shelf_occupancy"
        if product_capacity and capacity_units <= 0:
            capacity_units = product_capacity
            notes.append("Capacity taken from the product's facing_capacity.")

        if sku not in self._smoothed:
            # Seed the EMA with the first raw count — the only value we have.
            self._smoothed[sku] = float(observed)
            estimated = float(observed)
            notes.append("First observation: estimate equals the raw count.")
        else:
            # Recursive EMA: s_t = alpha * x_t + (1 - alpha) * s_{t-1}.
            # A single empty frame therefore moves the estimate by
            # alpha * 0 and never collapses it straight to zero.
            previous = self._smoothed[sku]
            estimated = round(self.alpha * float(observed) + (1.0 - self.alpha) * previous, 3)
            self._smoothed[sku] = estimated

        # Confidence: mean detection confidence, damped while the estimate is
        # still moving (first few observations).
        stability = 1.0 if len(history) < 2 else _stability(history)
        sample_factor = min(1.0, observation_count / 3.0)
        confidence = round(max(0.0, min(1.0, mean_conf * (0.6 + 0.4 * stability) * sample_factor)), 4)

        pct = stock_percentage(estimated, capacity_units)
        thresholds = thresholds or StockThresholds()
        state = classify_stock(estimated, capacity_units, confidence, thresholds)

        if capacity_units <= 0:
            notes.append("Shelf capacity is not configured; percentage is unknown.")
        if not any(d.sku for d in detections):
            notes.append("No SKU was recognised on this frame; the estimate is a floor, not a count.")
        if occupancy_pct is not None:
            notes.append(f"Shelf area occupancy {occupancy_pct:.1f}% (area ratio, not an item count).")

        return StockEstimate(
            sku=sku,
            product_id=product_id,
            product_name=product_name,
            shelf_id=shelf_id,
            camera_id=camera_id,
            observed_quantity=observed,
            estimated_quantity=estimated,
            capacity_units=capacity_units,
            stock_percentage=pct,
            confidence=confidence,
            observation_count=observation_count,
            method=method,
            occupancy_pct=occupancy_pct,
            state=state,
            notes=notes,
        )

    def history(self, sku: str) -> list[float]:
        return list(self._history.get(sku, []))


def _stability(history: list[float]) -> float:
    """1.0 when the recent series is flat, →0 when it swings."""
    if len(history) < 2:
        return 0.0
    mean = sum(history) / len(history)
    if mean <= 0:
        return 1.0 if max(history) - min(history) < 0.5 else 0.0
    spread = (max(history) - min(history)) / mean
    return round(max(0.0, min(1.0, 1.0 - spread)), 4)


def compare_to_registered(
    estimated_quantity: float, registered_quantity: int | None
) -> dict[str, Any]:
    """Difference between the camera estimate and the registered figure."""
    if registered_quantity is None:
        return {"discrepancy": None, "discrepancy_pct": None, "comparable": False}
    discrepancy = round(estimated_quantity - float(registered_quantity), 3)
    base = max(1.0, float(registered_quantity))
    return {
        "discrepancy": discrepancy,
        "discrepancy_pct": round(100.0 * discrepancy / base, 2),
        "comparable": True,
        "note": (
            "A camera cannot see hidden or backroom stock; a negative discrepancy "
            "does not prove stock is missing from the store."
        ),
    }