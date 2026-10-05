"""Visual anomaly detection tests (README §2, Module 7).

These assert the behaviour that makes the module trustworthy rather than
noisy: nothing fires before a baseline exists, nothing fires for a shelf that
is simply steady, a single odd frame never raises an alert, and a person's body
blocking the view is reported as occlusion rather than as missing stock.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.pipeline import person_cover_ratio
from app.vision.anomaly import (
    DESCRIPTOR_SIZE,
    AnomalyKind,
    AnomalySeverity,
    ShelfAnomalyDetector,
    cosine_similarity,
    describe_roi,
    save_evidence,
    severity_for,
)
from app.vision.types import BBox, Detection


def shelf_roi(fill: int, seed: int = 0) -> np.ndarray:
    """A synthetic shelf ROI: a bright band of goods on a dark shelf."""
    rng = np.random.default_rng(seed)
    roi = np.full((120, 300, 3), 30, dtype=np.uint8)
    roi[20:100, 10:290] = fill
    noise = rng.integers(0, 12, size=roi.shape, dtype=np.uint8)
    return np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def calibrate(detector, roi, n: int = 8, **kwargs):
    """Feed `n` observations so the detector learns this shelf's baseline."""
    for _ in range(n):
        detector.update("shelf-1", roi, occupancy_pct=80.0, detected_units=20, **kwargs)
    return detector


class TestCalibration:
    def test_nothing_is_flagged_before_the_baseline_exists(self):
        detector = ShelfAnomalyDetector(baseline_observations=8, persistence_required=3)
        for i in range(7):
            candidate = detector.update(
                "shelf-1", shelf_roi(200, seed=i), occupancy_pct=80.0, detected_units=20
            )
            assert candidate is None, f"flagged too early on observation {i}"
        assert detector.is_calibrated("shelf-1") is False

    def test_a_steady_shelf_is_never_flagged(self):
        detector = calibrate(ShelfAnomalyDetector(), shelf_roi(200, seed=1))
        for i in range(20):
            candidate = detector.update(
                "shelf-1", shelf_roi(200, seed=i), occupancy_pct=80.0, detected_units=20
            )
            assert candidate is None

    def test_shelves_do_not_share_state(self):
        detector = calibrate(ShelfAnomalyDetector(), shelf_roi(200))
        for _ in range(3):
            detector.update("shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
        # A brand-new shelf must not be judged against shelf-1's baseline.
        assert detector.is_calibrated("shelf-2") is False
        assert (
            detector.update("shelf-2", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
            is None
        )
        assert detector.tracked_shelves() == ["shelf-1", "shelf-2"]

    def test_forget_drops_a_shelf(self):
        detector = calibrate(ShelfAnomalyDetector(), shelf_roi(200))
        detector.forget("shelf-1")
        assert detector.is_calibrated("shelf-1") is False
        assert detector.tracked_shelves() == []


class TestPersistenceFilter:
    def test_a_single_odd_frame_is_filtered_out(self):
        """Pipeline D step 6: ignore temporary occlusions and uncertain results."""
        detector = calibrate(ShelfAnomalyDetector(persistence_required=3), shelf_roi(200))
        candidate = detector.update(
            "shelf-1", shelf_roi(20), occupancy_pct=5.0, detected_units=1
        )
        assert candidate is None, "one odd frame must not raise an alert"

    def test_a_sustained_change_is_flagged_with_evidence(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=3), shelf_roi(200))
        candidates = [
            detector.update("shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
            for _ in range(4)
        ]
        flagged = [c for c in candidates if c is not None]
        assert flagged, "a persistent visual change must eventually be flagged"
        strongest = flagged[0]
        assert strongest.shelf_id == "shelf-1"
        assert strongest.consecutive >= 3
        assert 0.0 <= strongest.score <= 1.0
        assert strongest.severity in set(AnomalySeverity)
        assert strongest.message
        # Measured context is stored so a reviewer can see why it fired.
        assert strongest.context
        assert {"change_score", "gap_score", "units_drop_ratio"} & set(
            strongest.context
        )
        # All three conditions were true at once; the strongest is reported and
        # the rest are kept so one alert tells the reviewer the whole story.
        others = strongest.context.get("additional_conditions", [])
        assert set(others) <= {
            AnomalyKind.UNUSUAL_GAP.value,
            AnomalyKind.APPEARANCE_CHANGE.value,
            AnomalyKind.DISPLACED_ITEMS.value,
        }

    def test_persistence_requirement_is_respected(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=5), shelf_roi(200))
        emitted = [
            detector.update("shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
            for _ in range(4)
        ]
        assert all(c is None for c in emitted), "fired before 5 consecutive observations"
        assert (
            detector.update("shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
            is not None
        )

    def test_a_streak_resets_when_the_shelf_returns_to_normal(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=3), shelf_roi(200))
        for _ in range(2):
            detector.update("shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
        for _ in range(5):
            detector.update(
                "shelf-1", shelf_roi(200), occupancy_pct=80.0, detected_units=20
            )
        # Back to the changed state: the full persistence run is required again.
        emitted = [
            detector.update("shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1)
            for _ in range(2)
        ]
        assert all(c is None for c in emitted)


class TestConditions:
    def test_an_emptied_shelf_is_reported_as_an_unusual_gap(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=1), shelf_roi(200))
        candidate = detector.update(
            "shelf-1", shelf_roi(200), occupancy_pct=10.0, detected_units=20
        )
        assert candidate is not None
        assert candidate.kind is AnomalyKind.UNUSUAL_GAP
        assert candidate.context["occupancy_pct"] == pytest.approx(10.0)

    def test_small_occupancy_jitter_does_not_count_as_a_gap(self):
        """A 5-point wobble is normal noise, not an unusual gap."""
        detector = calibrate(ShelfAnomalyDetector(persistence_required=1), shelf_roi(200))
        assert (
            detector.update(
                "shelf-1", shelf_roi(200), occupancy_pct=74.0, detected_units=20
            )
            is None
        )

    def test_fewer_visible_units_is_reported_as_displaced_items(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=1), shelf_roi(200))
        candidate = detector.update(
            "shelf-1", shelf_roi(200), occupancy_pct=80.0, detected_units=2
        )
        assert candidate is not None
        assert candidate.kind is AnomalyKind.DISPLACED_ITEMS
        assert candidate.context["baseline_units"] == pytest.approx(20.0)

    def test_a_person_in_front_is_occlusion_not_a_stock_event(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=2), shelf_roi(200))
        candidates = [
            detector.update(
                "shelf-1",
                shelf_roi(200),
                occupancy_pct=80.0,
                detected_units=20,
                occluder_ratio=0.9,
            )
            for _ in range(3)
        ]
        flagged = [c for c in candidates if c is not None]
        assert flagged
        assert flagged[0].kind is AnomalyKind.PERSISTENT_OCCLUSION
        # Critically: it must NOT also be reported as a gap or displaced stock.
        assert "displaced_items" not in flagged[0].context.get(
            "additional_conditions", []
        )
        assert "unusual_gap" not in flagged[0].context.get("additional_conditions", [])

    def test_a_brief_occlusion_does_not_fire(self):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=3), shelf_roi(200))
        assert (
            detector.update(
                "shelf-1",
                shelf_roi(200),
                occupancy_pct=80.0,
                detected_units=20,
                occluder_ratio=0.9,
            )
            is None
        )


class TestEvidence:
    def test_evidence_image_is_written_and_readable(self, tmp_path):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=1), shelf_roi(200))
        candidate = detector.update(
            "shelf-1",
            shelf_roi(20),
            occupancy_pct=8.0,
            detected_units=1,
            camera_id="cam-abc",
        )
        assert candidate is not None
        path = save_evidence(str(tmp_path), candidate, shelf_roi(20))
        assert path and path.endswith(".jpg")
        assert cv2.imread(path) is not None

    def test_evidence_returns_none_for_an_empty_frame(self, tmp_path):
        detector = calibrate(ShelfAnomalyDetector(persistence_required=1), shelf_roi(200))
        candidate = detector.update(
            "shelf-1", shelf_roi(20), occupancy_pct=8.0, detected_units=1
        )
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert save_evidence(str(tmp_path), candidate, empty) is None


class TestHelpers:
    def test_descriptor_is_normalised_and_stable(self):
        a = describe_roi(shelf_roi(200, seed=3))
        b = describe_roi(shelf_roi(200, seed=3))
        assert a.shape == (DESCRIPTOR_SIZE * DESCRIPTOR_SIZE,)
        assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-3)
        assert np.allclose(a, b)

    def test_a_uniform_roi_yields_a_zero_descriptor(self):
        uniform = np.full((50, 50, 3), 128, dtype=np.uint8)
        assert np.linalg.norm(describe_roi(uniform)) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_is_clamped_and_safe(self):
        v = describe_roi(shelf_roi(200))
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-3)
        assert cosine_similarity(v, np.zeros_like(v)) == 0.0
        assert cosine_similarity(np.array([]), v) == 0.0

    def test_severity_bands_are_fixed(self):
        assert severity_for(0.9) is AnomalySeverity.HIGH
        assert severity_for(0.6) is AnomalySeverity.MEDIUM
        assert severity_for(0.1) is AnomalySeverity.LOW
        # Out-of-range input is clamped, never raises.
        assert severity_for(5.0) is AnomalySeverity.HIGH
        assert severity_for(-1.0) is AnomalySeverity.LOW

    def test_person_cover_ratio_measures_overlap(self):
        polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        assert person_cover_ratio([], polygon) == 0.0
        far = [Detection(bbox=BBox(500, 500, 600, 600), class_id=0, class_name="person", confidence=0.9)]
        assert person_cover_ratio(far, polygon) == 0.0
        covering = [Detection(bbox=BBox(0, 0, 200, 200), class_id=0, class_name="person", confidence=0.9)]
        assert person_cover_ratio(covering, polygon) == pytest.approx(1.0)
        half = [Detection(bbox=BBox(0, 0, 100, 50), class_id=0, class_name="person", confidence=0.9)]
        assert person_cover_ratio(half, polygon) == pytest.approx(0.5, abs=0.05)
