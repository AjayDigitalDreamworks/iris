"""Visual anomaly detection for shelf regions.

**What this does:** flag shelf conditions that ordinary inventory detection can
miss — an unusual gap, a sudden change in how a shelf looks, items that appear
to have been moved, or a shelf that stays obscured. Each candidate carries a
score, a severity and the evidence frame it was judged from, so a staff member
can look at the picture and decide.

**What this does NOT do:** it does not detect theft, and it cannot. A camera
picture shows that a shelf *looks different*; it cannot establish that anything
was taken, by whom, or with what intent. Callers and the UI must present these
as *candidates for review*, never as confirmed incidents.

How it works, per shelf, with no trained model:

1. **Descriptor.** The shelf ROI is reduced to a small normalised grayscale
   vector. This is deliberately cheap: it must run on a CPU, on a live feed.
2. **Reference appearance.** The first N observations establish a baseline by
   running *median* per component, which ignores a single odd frame. Nothing is
   flagged until the baseline exists — otherwise the first observation of a
   shelf is always an "anomaly".
3. **Change score.** ``1 - cosine_similarity(current, reference)``. A large
   value means the shelf looks nothing like its established baseline.
4. **Gap score.** Occupancy falling far below its own baseline is the
   "unusual gap" signal. This is deliberately kept separate from the appearance
   score because a shelf can look different without being emptier.
5. **Persistence filter.** A candidate is only emitted after the same condition
   holds for several consecutive observations. One odd frame — a customer
   walking past, a light flicker — is filtered out rather than alerted on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from app.core.logging import get_logger
from app.vision.types import BBox

ai_log = get_logger("ai")

#: Side length of the square grayscale descriptor. 32x32 = 1024 dims, small
#: enough to keep cosine similarity cheap enough for every persisted frame.
DESCRIPTOR_SIZE = 32


class AnomalyKind(str, Enum):
    UNUSUAL_GAP = "unusual_gap"
    APPEARANCE_CHANGE = "appearance_change"
    DISPLACED_ITEMS = "displaced_items"
    PERSISTENT_OCCLUSION = "persistent_occlusion"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AnomalyCandidate:
    """One shelf condition worth a human's attention."""

    kind: AnomalyKind
    severity: AnomalySeverity
    score: float
    message: str
    camera_id: str | None = None
    shelf_id: str | None = None
    zone_id: str | None = None
    bbox: BBox | None = None
    #: Consecutive observations that produced this condition. Always >= the
    #: configured persistence requirement when a candidate is emitted.
    consecutive: int = 0
    #: Free-form measured context, stored with the alert as evidence.
    context: dict[str, float] = field(default_factory=dict)


def describe_roi(image: np.ndarray, size: int = DESCRIPTOR_SIZE) -> np.ndarray:
    """Reduce a shelf ROI to a normalised grayscale descriptor vector."""
    if image is None or getattr(image, "size", 0) == 0:
        return np.zeros(size * size, dtype=np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    # A perfectly uniform shelf has no structure to compare; the zero vector is
    # the honest answer, and cosine similarity against it is handled below.
    return (small / norm).reshape(-1) if norm > 1e-6 else np.zeros(size * size, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity clamped to [0, 1]; 0.0 if either vector is degenerate."""
    if a.size == 0 or b.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / (na * nb), 0.0, 1.0))


@dataclass
class ShelfBaseline:
    """Running, robust baseline for one shelf."""

    samples: list[np.ndarray] = field(default_factory=list)
    occupancies: list[float] = field(default_factory=list)
    units: list[float] = field(default_factory=list)

    @property
    def reference(self) -> np.ndarray | None:
        if not self.samples:
            return None
        return np.median(np.stack(self.samples), axis=0)

    @property
    def occupancy_reference(self) -> float | None:
        return float(np.median(self.occupancies)) if self.occupancies else None

    @property
    def units_reference(self) -> float | None:
        return float(np.median(self.units)) if self.units else None


class ShelfAnomalyDetector:
    """Stateful per-shelf anomaly detector.

    One instance is shared by every camera (the state is keyed by shelf id), the
    same way the tracker is per-camera: shelf identity must not leak between
    cameras.
    """

    def __init__(
        self,
        *,
        baseline_observations: int = 8,
        persistence_required: int = 3,
        gap_tolerance_pct: float = 12.0,
        change_threshold: float = 0.35,
        gap_threshold: float = 0.5,
        units_drop_ratio: float = 0.45,
        occlusion_ratio: float = 0.35,
        history_limit: int = 120,
    ) -> None:
        self.baseline_observations = max(2, baseline_observations)
        self.persistence_required = max(1, persistence_required)
        self.gap_tolerance_pct = gap_tolerance_pct
        self.change_threshold = change_threshold
        self.gap_threshold = gap_threshold
        self.units_drop_ratio = units_drop_ratio
        self.occlusion_ratio = occlusion_ratio
        self.history_limit = history_limit
        self._baselines: dict[str, ShelfBaseline] = {}
        self._streaks: dict[tuple[str, str], int] = {}

    # ── introspection (used by the API and the dashboard) ───────────────────
    def is_calibrated(self, shelf_id: str) -> bool:
        baseline = self._baselines.get(shelf_id)
        return bool(baseline and len(baseline.samples) >= self.baseline_observations)

    def tracked_shelves(self) -> list[str]:
        return sorted(self._baselines)

    def forget(self, shelf_id: str) -> None:
        self._baselines.pop(shelf_id, None)
        for key in [k for k in self._streaks if k[0] == shelf_id]:
            self._streaks.pop(key, None)

    # ── main entry point ───────────────────────────────────────────────────
    def update(
        self,
        shelf_id: str,
        roi: np.ndarray,
        *,
        occupancy_pct: float,
        detected_units: int,
        camera_id: str | None = None,
        zone_id: str | None = None,
        bbox: BBox | None = None,
        occluder_ratio: float = 0.0,
    ) -> AnomalyCandidate | None:
        """Record one observation and return a candidate if one is warranted.

        ``occluder_ratio`` is the fraction of the shelf covered by person boxes
        on this frame. It is used to tell "the shelf changed" apart from "some-
        one is standing in front of it", so a busy aisle does not raise
        appearance-change alerts all day.
        """
        descriptor = describe_roi(roi)
        baseline = self._baselines.setdefault(shelf_id, ShelfBaseline())
        baseline.samples.append(descriptor)
        baseline.occupancies.append(float(occupancy_pct))
        baseline.units.append(float(detected_units))
        # Keep memory bounded on a camera that runs for days.
        if len(baseline.samples) > self.history_limit:
            del baseline.samples[: len(baseline.samples) - self.history_limit]
            del baseline.occupancies[: len(baseline.occupancies) - self.history_limit]
            del baseline.units[: len(baseline.units) - self.history_limit]

        if len(baseline.samples) < self.baseline_observations:
            return None  # still learning what "normal" looks like

        # ── a shelf hidden behind a person is an occlusion, not a stock event ──
        if occluder_ratio >= self.occlusion_ratio:
            streak = self._bump(shelf_id, AnomalyKind.PERSISTENT_OCCLUSION.value)
            if streak < self.persistence_required:
                return None
            return self._emit(
                shelf_id,
                AnomalyKind.PERSISTENT_OCCLUSION,
                score=min(1.0, occluder_ratio),
                message=(
                    f"Shelf has been obscured by people for {streak} consecutive "
                    f"observations ({occluder_ratio:.0%} of the region covered)."
                ),
                camera_id=camera_id,
                zone_id=zone_id,
                bbox=bbox,
                streak=streak,
                context={"occluder_ratio": round(occluder_ratio, 4)},
            )

        candidates: list[AnomalyCandidate] = []

        # ── 1. how different does it look from its own baseline? ─────────────
        reference = baseline.reference
        similarity = cosine_similarity(descriptor, reference) if reference is not None else 1.0
        change_score = 1.0 - similarity
        if change_score >= self.change_threshold:
            streak = self._bump(shelf_id, AnomalyKind.APPEARANCE_CHANGE.value)
            if streak >= self.persistence_required:
                candidates.append(
                    self._emit(
                        shelf_id,
                        AnomalyKind.APPEARANCE_CHANGE,
                        score=change_score,
                        message=(
                            f"Shelf appearance differs from its established baseline "
                            f"({change_score:.0%} change) for {streak} consecutive observations."
                        ),
                        camera_id=camera_id,
                        zone_id=zone_id,
                        bbox=bbox,
                        streak=streak,
                        context={
                            "change_score": round(change_score, 4),
                            "baseline_similarity": round(similarity, 4),
                        },
                    )
                )
        else:
            self._clear(shelf_id, AnomalyKind.APPEARANCE_CHANGE.value)

        # ── 2. is it emptier than it normally is? ───────────────────────────
        occupancy_reference = baseline.occupancy_reference
        if occupancy_reference is not None:
            drop = occupancy_reference - float(occupancy_pct)
            # Tolerance is expressed in percentage points so it means the same
            # thing regardless of how big the number is.
            gap_score = max(0.0, drop - self.gap_tolerance_pct) / max(
                1.0, self.gap_tolerance_pct
            )
            if gap_score >= self.gap_threshold:
                streak = self._bump(shelf_id, AnomalyKind.UNUSUAL_GAP.value)
                if streak >= self.persistence_required:
                    candidates.append(
                        self._emit(
                            shelf_id,
                            AnomalyKind.UNUSUAL_GAP,
                            score=min(1.0, gap_score),
                            message=(
                                f"Occupancy {occupancy_pct:.0f}% is {drop:.0f} points below "
                                f"the {occupancy_reference:.0f}% baseline for {streak} "
                                f"consecutive observations."
                            ),
                            camera_id=camera_id,
                            zone_id=zone_id,
                            bbox=bbox,
                            streak=streak,
                            context={
                                "gap_score": round(gap_score, 4),
                                "occupancy_pct": round(float(occupancy_pct), 2),
                                "baseline_occupancy_pct": round(occupancy_reference, 2),
                            },
                        )
                    )
            else:
                self._clear(shelf_id, AnomalyKind.UNUSUAL_GAP.value)

        # ── 3. fewer visible units without a matching occupancy drop ─────────
        #    Something is blocking or obscuring individual items rather than the
        #    shelf simply running down. Reported as a candidate, not as a theft.
        units_reference = baseline.units_reference
        if units_reference and units_reference >= 2:
            drop_ratio = (units_reference - float(detected_units)) / units_reference
            if drop_ratio >= self.units_drop_ratio:
                streak = self._bump(shelf_id, AnomalyKind.DISPLACED_ITEMS.value)
                if streak >= self.persistence_required:
                    candidates.append(
                        self._emit(
                            shelf_id,
                            AnomalyKind.DISPLACED_ITEMS,
                            score=min(1.0, drop_ratio),
                            message=(
                                f"{detected_units} unit(s) visible versus a baseline of "
                                f"{units_reference:.0f} for {streak} consecutive observations."
                            ),
                            camera_id=camera_id,
                            zone_id=zone_id,
                            bbox=bbox,
                            streak=streak,
                            context={
                                "units_drop_ratio": round(drop_ratio, 4),
                                "detected_units": float(detected_units),
                                "baseline_units": round(units_reference, 2),
                            },
                        )
                    )
            else:
                self._clear(shelf_id, AnomalyKind.DISPLACED_ITEMS.value)

        if not candidates:
            return None
        # Report the most serious condition, and keep the rest as context so a
        # reviewer sees the whole picture from one alert.
        candidates.sort(key=lambda c: c.score, reverse=True)
        strongest = candidates[0]
        strongest.context["additional_conditions"] = [c.kind.value for c in candidates[1:]]
        ai_log.info(
            "Shelf anomaly candidate",
            extra={
                "shelf_id": shelf_id,
                "camera_id": camera_id,
                "kind": strongest.kind.value,
                "severity": strongest.severity.value,
                "score": round(strongest.score, 4),
            },
        )
        return strongest

    # ── streak bookkeeping ─────────────────────────────────────────────────
    def _bump(self, shelf_id: str, kind: str) -> int:
        key = (shelf_id, kind)
        streak = self._streaks.get(key, 0) + 1
        self._streaks[key] = streak
        return streak

    def _clear(self, shelf_id: str, kind: str) -> None:
        self._streaks.pop((shelf_id, kind), None)

    def _emit(
        self,
        shelf_id: str,
        kind: AnomalyKind,
        *,
        score: float,
        message: str,
        camera_id: str | None,
        zone_id: str | None,
        bbox: BBox | None,
        streak: int,
        context: dict[str, float],
    ) -> AnomalyCandidate:
        return AnomalyCandidate(
            kind=kind,
            severity=severity_for(score),
            score=round(float(min(1.0, max(0.0, score))), 4),
            message=message,
            camera_id=camera_id,
            zone_id=zone_id,
            shelf_id=shelf_id,
            bbox=bbox,
            consecutive=streak,
            context=context,
        )


def severity_for(score: float) -> AnomalySeverity:
    """Map a 0..1 score onto a severity band.

    Bands are deliberately fixed constants rather than anything derived from the
    data, so the same score is always reported as the same severity.
    """
    if score >= 0.75:
        return AnomalySeverity.HIGH
    if score >= 0.5:
        return AnomalySeverity.MEDIUM
    return AnomalySeverity.LOW


def save_evidence(
    directory: str,
    candidate: AnomalyCandidate,
    frame: np.ndarray,
) -> str | None:
    """Write the judged frame next to its alert and return the stored path.

    Evidence is what turns "the shelf looks odd" into something a staff member
    can act on, so the image the decision was made from is kept, not a
    reconstruction of it.
    """
    import os
    import time

    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    try:
        shelf = (candidate.shelf_id or "shelf")[:16]
        camera = (candidate.camera_id or "camera")[:16]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{camera}_{shelf}_{candidate.kind.value}_{stamp}.jpg")
        cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return path
    except Exception as exc:  # pragma: no cover - evidence is best-effort
        ai_log.warning("Could not save anomaly evidence", exc_info=exc)
        return None


def haversine_like_ratio(a: float, b: float) -> float:
    """Relative difference between two non-negative values, 0..1."""
    if a <= 0 and b <= 0:
        return 0.0
    larger = max(abs(a), abs(b))
    return min(1.0, abs(a - b) / larger) if larger > 0 else 0.0


_ = math  # kept for callers that extend the scoring with distance terms
