"""Retail product detector tests.

The behaviour under test is the combination rule ported from
`Retail-product-detection-main/`: multipack boxes take precedence, and a general
box is discarded exactly when its centre falls strictly inside a multipack box.
That boundary matters — a pack standing flush against a multipack must still be
counted — so the strict comparisons are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.vision.retail_detector import (
    ShelfStockView,
    center_inside,
    combine_general_and_multipack,
    shelf_stock_view,
)
from app.vision.types import BBox, Detection


def box(x1, y1, x2, y2, conf=0.9, cls="product") -> Detection:
    return Detection(
        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
        class_id=1,
        class_name=cls,
        confidence=conf,
    )


class _StubDetector:
    """Returns a fixed detection list; no model, no inference."""

    def __init__(self, detections, latency_ms=5.0, model_id="stub") -> None:
        self._detections = detections
        self._latency_ms = latency_ms
        self.model_id = model_id
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return list(self._detections), self._latency_ms


class TestCenterInside:
    def test_centre_inside_is_true(self):
        assert center_inside(BBox(10, 10, 20, 20), BBox(0, 0, 100, 100)) is True

    def test_centre_outside_is_false(self):
        # Centre to the right of the multipack.
        assert center_inside(BBox(90, 10, 130, 20), BBox(0, 0, 100, 100)) is False

    def test_centre_exactly_on_the_edge_is_not_inside(self):
        """Strict comparison: a flush pack must not be swallowed."""
        # Centre x = 100, multipack xmax = 100.
        assert center_inside(BBox(95, 10, 105, 20), BBox(0, 0, 100, 100)) is False

    def test_overlap_alone_does_not_suppress(self):
        """Overlap is not the rule - centre containment is."""
        # Overlaps heavily but its centre is outside.
        assert center_inside(BBox(80, 80, 200, 200), BBox(0, 0, 100, 100)) is False


class TestCombinationRule:
    def test_boxes_inside_a_multipack_are_dropped(self):
        multipack = [box(0, 0, 100, 100)]
        general = [
            box(10, 10, 30, 30),   # centre inside -> dropped
            box(200, 10, 220, 30), # centre outside -> kept
        ]
        combined, suppressed = combine_general_and_multipack(general, multipack)
        assert suppressed == 1
        assert len(combined) == 2

    def test_multipack_boxes_come_first(self):
        multipack = [box(0, 0, 100, 100)]
        general = [box(200, 10, 220, 30)]
        combined, _ = combine_general_and_multipack(general, multipack)
        assert combined[0] is multipack[0]

    def test_no_multipack_stage_keeps_everything(self):
        general = [box(10, 10, 30, 30), box(40, 10, 60, 30)]
        combined, suppressed = combine_general_and_multipack(general, [])
        assert len(combined) == 2
        assert suppressed == 0

    def test_one_multipack_can_absorb_several_packs(self):
        multipack = [box(0, 0, 100, 100)]
        general = [box(5, 5, 15, 15), box(20, 20, 30, 30), box(40, 40, 50, 50)]
        combined, suppressed = combine_general_and_multipack(general, multipack)
        assert suppressed == 3
        assert len(combined) == 1

    def test_each_pack_is_matched_against_every_multipack(self):
        multipack = [box(0, 0, 40, 40), box(200, 200, 300, 300)]
        general = [box(10, 10, 20, 20), box(240, 240, 260, 260)]
        _, suppressed = combine_general_and_multipack(general, multipack)
        assert suppressed == 2

    def test_an_empty_general_list_still_returns_the_multipacks(self):
        combined, suppressed = combine_general_and_multipack([], [box(0, 0, 10, 10)])
        assert len(combined) == 1
        assert suppressed == 0


class TestRetailProductDetector:
    def _detector(self, general, multipack=None, min_confidence=0.3):
        from app.vision.retail_detector import RetailProductDetector

        return RetailProductDetector(
            _StubDetector(general),
            multipack=_StubDetector(multipack) if multipack is not None else None,
            min_confidence=min_confidence,
        )

    def test_runs_general_only_when_no_multipack_weights_exist(self, monkeypatch):
        monkeypatch.setattr(
            "app.vision.retail_detector.settings.retail_multipack_weights", "", raising=False
        )
        detector = self._detector([box(10, 10, 30, 30)])
        from app.vision.types import Frame

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detections, _ = detector.predict(Frame(image=image, index=1, timestamp=0.0))
        assert len(detections) == 1
        status = detector.status()
        assert status["multipack_enabled"] is False
        # The missing stage must be reported, never hidden.
        assert status["multipack_note"]
        assert "general-only" in detector.model_id()

    def test_multipack_stage_is_used_when_present(self):
        detector = self._detector(
            [box(10, 10, 30, 30), box(200, 10, 220, 30)],
            multipack=[box(0, 0, 100, 100)],
        )
        from app.vision.types import Frame

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        detections, _ = detector.predict(Frame(image=image, index=1, timestamp=0.0))
        assert detector.status()["multipack_enabled"] is True
        assert detector.status()["suppressed_by_multipack"] == 1
        assert len(detections) == 2  # 1 multipack + 1 surviving general

    def test_low_confidence_detections_are_filtered(self):
        detector = self._detector([box(10, 10, 30, 30, conf=0.1)], min_confidence=0.5)
        from app.vision.types import Frame

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        detections, _ = detector.predict(Frame(image=image, index=1, timestamp=0.0))
        assert detections == []

    def test_both_stages_contribute_to_latency(self):
        detector = self._detector([box(10, 10, 30, 30)], multipack=[box(0, 0, 5, 5)])
        from app.vision.types import Frame

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        _, latency = detector.predict(Frame(image=image, index=1, timestamp=0.0))
        assert latency == pytest.approx(10.0)

    def test_every_detection_records_its_provenance(self):
        detector = self._detector(
            [box(200, 10, 220, 30)], multipack=[box(0, 0, 100, 100)]
        )
        from app.vision.types import Frame

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        detections, _ = detector.predict(Frame(image=image, index=1, timestamp=0.0))
        assert all("retail" in d.extra for d in detections)
        multipack_flags = [d.extra["retail"]["multipack"] for d in detections]
        assert True in multipack_flags and False in multipack_flags


class TestOutOfStockDecision:
    def test_a_shelf_with_nothing_on_it_is_out_of_stock(self):
        view = ShelfStockView(
            occupancy_pct=0.0, detected_units=0, empty_positions=6, total_positions=6
        )
        assert view.is_empty is True
        assert view.label == "OUT OF STOCK"

    def test_units_present_means_not_out_of_stock(self):
        view = ShelfStockView(
            occupancy_pct=70.0, detected_units=12, empty_positions=1, total_positions=6
        )
        assert view.is_empty is False
        assert "12" in view.label

    def test_empty_area_alone_is_not_enough(self):
        """A dark shelf can read as empty area while units are still there."""
        view = ShelfStockView(
            occupancy_pct=0.0, detected_units=5, empty_positions=6, total_positions=6
        )
        assert view.is_empty is False

    def test_zero_units_alone_is_not_enough(self):
        """A few missed detections must not be reported as an empty shelf."""
        view = ShelfStockView(
            occupancy_pct=64.0, detected_units=0, empty_positions=0, total_positions=6
        )
        assert view.is_empty is False

    def test_view_is_built_from_a_shelf_observation(self):
        class FakeObservation:
            occupancy_pct = 0.0
            detected_units = 0
            empty_positions = 5
            total_positions = 5
            notes = ["first observation is uncertain"]

        view = shelf_stock_view(FakeObservation())
        assert view.is_empty is True
        assert view.notes == ["first observation is uncertain"]
        assert view.as_dict()["label"] == "OUT OF STOCK"


class TestExifOrientation:
    def test_a_missing_image_returns_none(self, tmp_path):
        from app.vision.retail_detector import apply_exif_orientation

        assert apply_exif_orientation(tmp_path / "nope.jpg") is None

    def test_a_normal_image_loads_as_bgr(self, tmp_path):
        import cv2

        from app.vision.retail_detector import apply_exif_orientation

        path = tmp_path / "shelf.jpg"
        source = np.zeros((40, 80, 3), dtype=np.uint8)
        source[:, 40:, 2] = 255  # red on the right half, in BGR
        assert cv2.imwrite(str(path), source)

        loaded = apply_exif_orientation(path)
        assert loaded is not None
        assert loaded.shape[:2] == (40, 80)
        # Red in BGR is the low byte of channel 2 being high.
        assert loaded[:, 60, 2].mean() > 200
