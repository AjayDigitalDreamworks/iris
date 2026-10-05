"""Unit tests for the vision layer: geometry, stock estimation, shelf analysis,
tracking, recognition and planogram compliance."""

from __future__ import annotations

import numpy as np
import pytest

from app.vision.embedder import CatalogEmbedding, ProductRecognizer
from app.vision.footfall import CustomerAnalyticsEngine
from app.vision.planogram import PlanogramChecker
from app.vision.queue_analysis import QueueAnalyzer, QueueMeasurement
from app.vision.rules import QueueThresholds
from app.vision.shelf_analysis import SlotOccupancyEstimator
from app.vision.stock import StockEstimator, compare_to_registered
from app.services.pipeline import stamp_track_ids
from app.vision.tracker import (
    line_crossed,
    point_in_polygon,
    polygon_area,
    scaled_line,
    scaled_polygon,
)
from app.vision.types import BBox, Detection, Frame, TrackedObject


# ── Geometry ─────────────────────────────────────────────────────────────────
class TestGeometry:
    SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]

    @pytest.mark.parametrize(
        ("point", "expected"),
        [((50, 50), True), ((0, 0), False), ((150, 50), False), ((50, -5), False)],
    )
    def test_point_in_polygon(self, point, expected):
        assert point_in_polygon(point, self.SQUARE) is expected

    def test_degenerate_polygon_is_never_inside(self):
        assert point_in_polygon((5, 5), [(0, 0), (1, 1)]) is False

    def test_polygon_area(self):
        assert polygon_area(self.SQUARE) == pytest.approx(10000.0)
        assert polygon_area([(0, 0), (1, 1)]) == 0.0

    def test_scaled_polygon_maps_to_pixels(self):
        assert scaled_polygon([[0.5, 0.25]], 200, 100) == [[100.0, 25.0]]

    def test_scaled_line_requires_two_points(self):
        assert scaled_line([[0.5, 0.5]], 100, 100) is None
        assert scaled_line([[0.5, 0.5], [0.5, 1.0]], 100, 100) == [[50.0, 50.0], [50.0, 100.0]]


class TestLineCrossing:
    VERTICAL = [(50, 0), (50, 100)]
    HORIZONTAL = [(0, 50), (100, 50)]

    def test_horizontal_motion_crosses_a_vertical_line(self):
        crossed, direction = line_crossed((70, 50), (30, 50), self.VERTICAL)
        assert crossed is True and direction in {"+", "-"}

    def test_motion_staying_on_one_side_does_not_cross(self):
        assert line_crossed((70, 50), (60, 50), self.VERTICAL)[0] is False

    def test_reverse_motion_gives_the_opposite_direction(self):
        forward = line_crossed((70, 50), (30, 50), self.VERTICAL)[1]
        backward = line_crossed((30, 50), (70, 50), self.VERTICAL)[1]
        assert forward != backward

    def test_vertical_motion_crosses_a_horizontal_line(self):
        assert line_crossed((50, 70), (50, 30), self.HORIZONTAL)[0] is True

    def test_degenerate_line_never_crosses(self):
        assert line_crossed((0, 0), (1, 1), [(5, 5), (5, 5)])[0] is False


class TestBBox:
    def test_normalised_conversion(self):
        box = BBox(50, 25, 150, 75).normalized(200, 100)
        assert (box.x1, box.y1, box.x2, box.y2) == (0.25, 0.25, 0.75, 0.75)

    def test_iou_of_identical_boxes_is_one(self):
        box = BBox(0, 0, 10, 10)
        assert box.iou(BBox(0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert BBox(0, 0, 10, 10).iou(BBox(50, 50, 60, 60)) == 0.0

    def test_bottom_center(self):
        assert BBox(10, 20, 30, 60).bottom_center == (20.0, 60.0)


# ── Stock estimation ─────────────────────────────────────────────────────────
def make_detection(sku="SKU-1", confidence=0.9, box=(0, 0, 10, 10)) -> Detection:
    return Detection(
        bbox=BBox(*box), class_id=1, class_name="bottle", confidence=confidence,
        product_id=f"p-{sku}", sku=sku, similarity=0.8,
    )


class TestStockEstimator:
    def test_counts_only_confident_recognised_detections(self):
        estimator = StockEstimator(min_unit_confidence=0.35)
        detections = [
            make_detection("A"), make_detection("B"),
            make_detection("C", confidence=0.10),          # below confidence floor
            make_detection("D", confidence=0.9, box=(0, 0, 5, 5)),
        ]
        detections[3].sku = None                            # unrecognised
        count, confidence, notes = estimator.count_detections(detections)
        assert count == 2
        assert 0.0 < confidence <= 1.0
        assert notes, "unrecognised objects must be reported, not silently dropped"

    def test_first_estimate_equals_raw_count(self):
        estimator = StockEstimator()
        estimate = estimator.estimate(
            sku="A", detections=[make_detection("A"), make_detection("A")],
            capacity_units=10,
        )
        assert estimate.estimated_quantity == pytest.approx(2.0)
        assert any("First observation" in note for note in estimate.notes)

    def test_estimate_is_smoothed_not_jumpy(self):
        estimator = StockEstimator(alpha=0.3)
        estimator.estimate(sku="A", detections=[make_detection("A")] * 5, capacity_units=10)
        estimate = estimator.estimate(sku="A", detections=[], capacity_units=10)
        # One empty frame must not collapse the estimate straight to zero.
        assert estimate.estimated_quantity == pytest.approx(3.5)

    def test_capacity_falls_back_to_product_facing_capacity(self):
        estimator = StockEstimator()
        estimate = estimator.estimate(
            sku="A", detections=[], capacity_units=0, product_capacity=20
        )
        assert estimate.capacity_units == 20
        assert estimate.stock_percentage == 0.0

    def test_unknown_capacity_reports_unknown_not_zero_percent(self):
        estimator = StockEstimator()
        estimate = estimator.estimate(sku="A", detections=[], capacity_units=0)
        assert estimate.stock_percentage == -1.0
        assert any("percentage is unknown" in note for note in estimate.notes)

    def test_confidence_is_damped_during_early_observations(self):
        estimator = StockEstimator()
        first = estimator.estimate(
            sku="A", detections=[make_detection("A", 0.95)], capacity_units=10
        )
        for _ in range(4):
            last = estimator.estimate(
                sku="A", detections=[make_detection("A", 0.95)], capacity_units=10
            )
        assert first.confidence < last.confidence

    def test_history_is_bounded(self):
        estimator = StockEstimator(history_size=4)
        for _ in range(20):
            estimator.estimate(sku="A", detections=[make_detection("A")], capacity_units=10)
        assert len(estimator.history("A")) == 4


class TestRegisteredComparison:
    def test_discrepancy_against_registered_stock(self):
        result = compare_to_registered(20.0, 24)
        assert result["discrepancy"] == pytest.approx(-4.0)
        assert result["comparable"] is True

    def test_no_registered_figure_is_not_comparable(self):
        result = compare_to_registered(20.0, None)
        assert result["comparable"] is False
        assert result["discrepancy"] is None


# ── Shelf analysis ───────────────────────────────────────────────────────────
class TestShelfAnalysis:
    POLYGON = [[0.05, 0.1], [0.95, 0.1], [0.95, 0.9], [0.05, 0.9]]

    def _frame(self, value: int = 200) -> Frame:
        image = np.full((360, 640, 3), value, dtype=np.uint8)
        return Frame(image=image, index=1, timestamp=0.0)

    def test_reports_uncertainty_on_first_observation(self):
        estimator = SlotOccupancyEstimator(slots_per_row=6, rows=2)
        observation = estimator.analyse(
            self._frame().image, shelf_id="s", zone_id=None, camera_id="c",
            polygon_norm=self.POLYGON, detections=[], capacity_units=48,
        )
        assert observation.uncertain is True
        assert observation.total_positions == 12

    def test_rejects_an_unconfigured_polygon(self):
        estimator = SlotOccupancyEstimator()
        observation = estimator.analyse(
            self._frame().image, shelf_id="s", zone_id=None, camera_id="c",
            polygon_norm=[], detections=[],
        )
        assert observation.uncertain is True
        assert observation.total_positions == 0

    def test_detections_mark_their_slot_occupied(self):
        estimator = SlotOccupancyEstimator(slots_per_row=6, rows=2)
        estimator.analyse(
            self._frame().image, shelf_id="s", zone_id=None, camera_id="c",
            polygon_norm=self.POLYGON, detections=[],
        )
        detection = make_detection("A", box=(100, 100, 150, 200))
        observation = estimator.analyse(
            self._frame().image, shelf_id="s", zone_id=None, camera_id="c",
            polygon_norm=self.POLYGON, detections=[detection],
        )
        assert observation.detected_units >= 1
        assert observation.empty_positions <= observation.total_positions

    def test_occupancy_is_a_percentage(self):
        estimator = SlotOccupancyEstimator(slots_per_row=4, rows=1)
        observation = estimator.analyse(
            self._frame().image, shelf_id="s", zone_id=None, camera_id="c",
            polygon_norm=self.POLYGON, detections=[],
        )
        assert 0.0 <= observation.occupancy_pct <= 100.0


# ── Customer analytics ───────────────────────────────────────────────────────
class TestCustomerAnalytics:
    def _frame(self) -> Frame:
        return Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), index=1, timestamp=0.0)

    def _states(self, track_id: int, foot: tuple[float, float]) -> dict:
        from app.vision.tracker import TrackState

        state = TrackState(
            external_id=track_id, class_name="person", first_seen=0.0,
            last_seen=1.0, bbox=BBox(foot[0] - 10, 0, foot[0] + 10, foot[1]),
        )
        state.centroids = [foot]
        return {track_id: state}

    def _track(self, track_id: int, foot: tuple[float, float]) -> TrackedObject:
        return TrackedObject(
            track_id=track_id, bbox=BBox(foot[0] - 10, 0, foot[0] + 10, foot[1]),
            class_id=0, class_name="person", confidence=0.9,
        )

    def test_counts_one_crossing_not_one_per_frame(self):
        engine = CustomerAnalyticsEngine("cam1")
        engine.configure(
            lines=[{"zone_id": "z1", "line": [[0.5, 0.0], [0.5, 1.0]], "entry_on_positive": True}]
        )
        frame = self._frame()
        crossings = 0
        for step in range(12):
            foot = (600 - step * 40, 300)
            result = engine.update(
                frame, [self._track(1, foot)], self._states(1, foot), now=float(step)
            )
            crossings += len(result.crossings)
        assert crossings == 1, "jitter across a line must be counted once"
        assert engine.entries == 1

    def test_reverse_crossing_counts_as_an_exit(self):
        engine = CustomerAnalyticsEngine("cam1")
        engine.configure(
            lines=[{"zone_id": "z1", "line": [[0.5, 0.0], [0.5, 1.0]], "entry_on_positive": True}]
        )
        frame = self._frame()
        for step in range(12):
            foot = (600 - step * 40, 300)
            engine.update(frame, [self._track(1, foot)], self._states(1, foot), now=float(step))
        for step in range(12):
            foot = (100 + step * 40, 300)
            engine.update(frame, [self._track(1, foot)], self._states(1, foot), now=float(100 + step))
        assert engine.entries == 1
        assert engine.exits == 1

    def test_two_people_are_counted_separately(self):
        engine = CustomerAnalyticsEngine("cam1")
        engine.configure(
            lines=[{"zone_id": "z1", "line": [[0.5, 0.0], [0.5, 1.0]], "entry_on_positive": True}]
        )
        frame = self._frame()
        for step in range(12):
            a = (600 - step * 40, 200)
            b = (600 - step * 40, 380)
            engine.update(
                frame, [self._track(1, a), self._track(2, b)],
                {**self._states(1, a), **self._states(2, b)}, now=float(step),
            )
        assert engine.entries == 2

    def test_occupancy_reflects_current_people(self):
        engine = CustomerAnalyticsEngine("cam1")
        frame = self._frame()
        foot = (100.0, 300.0)
        result = engine.update(frame, [self._track(1, foot)], self._states(1, foot))
        assert result.current_occupancy == 1
        assert result.active_tracks == 1

    def test_non_person_tracks_are_ignored(self):
        engine = CustomerAnalyticsEngine("cam1")
        frame = self._frame()
        box = BBox(10, 10, 50, 50)
        engine.update(frame, [TrackedObject(1, box, 7, "bottle", 0.9)], {})
        assert engine.entries == 0

    def test_zone_occupancy_is_counted(self):
        engine = CustomerAnalyticsEngine("cam1")
        engine.configure(
            occupancy_zones=[{"zone_id": "z", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        )
        frame = self._frame()
        foot = (300.0, 300.0)
        result = engine.update(frame, [self._track(1, foot)], self._states(1, foot))
        assert result.zone_occupancy["z"] == 1


# ── Queue analytics ──────────────────────────────────────────────────────────
class TestQueueAnalysis:
    QUEUE = {"zone_id": "z", "name": "Queue", "zone_type": "queue",
             "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
             "service_stations": 2}

    def _frame(self) -> Frame:
        return Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), index=1, timestamp=0.0)

    def _people(self, count: int) -> list[TrackedObject]:
        return [
            TrackedObject(
                track_id=i, bbox=BBox(100 + i * 40, 100, 140 + i * 40, 400),
                class_id=0, class_name="person", confidence=0.9,
            )
            for i in range(count)
        ]

    def test_empty_queue(self):
        analyzer = QueueAnalyzer("cam", QueueThresholds(busy=2, critical=4), zones=[self.QUEUE])
        result = analyzer.analyse(self._frame(), [])
        assert result[0].queue_length == 0
        assert result[0].status.value == "normal"
        assert result[0].estimated_wait_seconds == 0.0

    def test_queue_status_follows_configured_thresholds(self):
        analyzer = QueueAnalyzer("cam", QueueThresholds(busy=2, critical=4), zones=[self.QUEUE])
        assert analyzer.analyse(self._frame(), self._people(1))[0].status.value == "normal"
        assert analyzer.analyse(self._frame(), self._people(2))[0].status.value == "busy"
        assert analyzer.analyse(self._frame(), self._people(4))[0].status.value == "critical"

    def test_wait_estimate_is_labelled_as_estimate(self):
        analyzer = QueueAnalyzer("cam", QueueThresholds(busy=2, critical=4), zones=[self.QUEUE])
        measurement = analyzer.analyse(self._frame(), self._people(3))[0]
        assert measurement.estimated_wait_seconds and measurement.estimated_wait_seconds > 0
        assert "ESTIMATE" in measurement.note

    def test_zone_without_a_polygon_is_skipped(self):
        analyzer = QueueAnalyzer("cam", zones=[{"zone_id": "z", "zone_type": "queue", "polygon": []}])
        assert analyzer.analyse(self._frame(), self._people(3)) == []

    def test_payload_is_serialisable(self):
        analyzer = QueueAnalyzer("cam", zones=[self.QUEUE])
        payload = analyzer.analyse(self._frame(), self._people(1))[0].as_payload()
        assert set(payload) >= {"zone_id", "queue_length", "status", "confidence"}


# ── Planogram ────────────────────────────────────────────────────────────────
class TestPlanogram:
    PLANOGRAM = {
        "id": "p1",
        "shelf_id": "s1",
        "version": 1,
        "items": [
            {"product_id": "a", "sku": "SKU-A", "section_code": "L1", "expected_facings": 2},
            {"product_id": "b", "sku": "SKU-B", "section_code": "L1", "expected_facings": 1},
        ],
    }

    def test_no_planogram_means_no_compliance_number(self):
        result = PlanogramChecker().check(None, [])
        assert result.configured is False
        assert result.compliance_pct == 0.0
        assert "not a failure" in result.note

    def test_matching_layout_is_compliant(self):
        detections = [
            make_detection("SKU-A", box=(0, 0, 10, 10)),
            make_detection("SKU-A", box=(20, 0, 30, 10)),
            make_detection("SKU-B", box=(40, 0, 50, 10)),
        ]
        result = PlanogramChecker().check(self.PLANOGRAM, detections)
        assert result.compliance_pct == 100.0
        assert all(row.status == "compliant" for row in result.rows)

    def test_missing_sku_is_reported(self):
        result = PlanogramChecker().check(self.PLANOGRAM, [make_detection("SKU-A")])
        statuses = {row.sku: row.status for row in result.rows}
        assert statuses["SKU-B"] == "missing"
        assert result.compliance_pct < 100.0

    def test_unplanned_sku_is_misplaced(self):
        detections = [make_detection("SKU-Z")]
        result = PlanogramChecker().check(self.PLANOGRAM, detections)
        row = next(r for r in result.rows if r.sku == "SKU-Z")
        assert row.status == "misplaced"

    def test_over_facing_is_detected(self):
        detections = [make_detection("SKU-A", box=(i * 12, 0, i * 12 + 10, 10)) for i in range(9)]
        result = PlanogramChecker().check(self.PLANOGRAM, detections)
        row = next(r for r in result.rows if r.sku == "SKU-A")
        assert row.status == "over_faced"

    def test_low_confidence_detections_do_not_create_compliance(self):
        detections = [make_detection("SKU-A", confidence=0.05)]
        result = PlanogramChecker().check(self.PLANOGRAM, detections)
        assert all(row.observed_facings == 0 for row in result.rows)

    def test_note_always_warns_about_confidence(self):
        result = PlanogramChecker().check(self.PLANOGRAM, [make_detection("SKU-A")])
        assert "confidence" in result.note.lower()

# ── detection ↔ track linking ─────────────────────────────────────────────
class TestStampTrackIds:
    """A stored detection must be attributable to the person-track it belongs to.

    Regression: ByteTrack returns its own box copies, so `detections.track_id`
    stayed NULL even though the `tracks` table was being filled. Nothing could
    correlate the two.
    """

    @staticmethod
    def _person(box) -> Detection:
        return Detection(
            bbox=BBox(*box), class_id=0, class_name="person", confidence=0.8
        )

    @staticmethod
    def _track(track_id: int, box) -> TrackedObject:
        return TrackedObject(
            track_id=track_id,
            bbox=BBox(*box),
            class_id=0,
            class_name="person",
            confidence=0.8,
        )

    def test_matching_detection_gets_the_track_id(self):
        detection = self._person((10, 10, 50, 100))
        linked = stamp_track_ids([detection], [self._track(7, (12, 12, 52, 102))])
        assert linked == 1
        assert detection.track_id == 7

    def test_unrelated_track_is_not_linked(self):
        detection = self._person((0, 0, 20, 20))
        linked = stamp_track_ids([detection], [self._track(9, (400, 400, 440, 460))])
        assert linked == 0
        assert detection.track_id is None

    def test_one_track_claims_at_most_one_detection(self):
        first = self._person((0, 0, 40, 40))
        second = self._person((2, 2, 42, 42))
        linked = stamp_track_ids([first, second], [self._track(3, (1, 1, 41, 41))])
        assert linked == 1
        assert {first.track_id, second.track_id} == {3, None}

    def test_two_people_get_two_different_track_ids(self):
        first = self._person((0, 0, 40, 40))
        second = self._person((200, 0, 240, 40))
        linked = stamp_track_ids(
            [first, second],
            [self._track(1, (0, 0, 40, 40)), self._track(2, (200, 0, 240, 40))],
        )
        assert linked == 2
        assert first.track_id == 1
        assert second.track_id == 2

    def test_no_tracks_leaves_detections_untouched(self):
        detection = self._person((0, 0, 10, 10))
        assert stamp_track_ids([detection], []) == 0
        assert detection.track_id is None


# ── recognition allow-list ────────────────────────────────────────────────
class _StubEmbedder:
    """Returns a fixed unit vector so retrieval is fully deterministic."""

    def __init__(self, vector=None) -> None:
        self.vector = (
            vector if vector is not None else np.array([1.0, 0.0], dtype=np.float32)
        )

    def embed(self, image):
        return self.vector


class TestRecognitionAllowList:
    """Recognition must run on sellable items only.

    Regression measured on a real crowd clip: with only `person` denied, a
    backpack crop retrieved a demo apple's SKU (similarity above the floor), so
    a shopper's bag was reported as shelf stock. Recognition is therefore an
    allow-list of retail item classes.
    """

    @staticmethod
    def _recognizer_with_one_product() -> ProductRecognizer:
        recognizer = ProductRecognizer(_StubEmbedder(), min_similarity=0.3)
        recognizer.load_entries(
            [
                CatalogEmbedding(
                    product_id="p1",
                    sku="SKU-APPLE",
                    name="Apple",
                    vector=np.array([1.0, 0.0], dtype=np.float32),
                    model_key="embedder",
                )
            ]
        )
        return recognizer

    @staticmethod
    def _detection(class_name: str) -> Detection:
        return Detection(
            bbox=BBox(0, 0, 20, 20),
            class_id=1,
            class_name=class_name,
            confidence=0.9,
        )

    def _annotate(self, class_name: str) -> Detection:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        detection = self._detection(class_name)
        self._recognizer_with_one_product().annotate([detection], image)
        return detection

    def test_a_product_class_gets_the_sku(self):
        detection = self._annotate("bottle")
        assert detection.sku == "SKU-APPLE"
        assert detection.extra["recognition"]["state"] == "matched"

    def test_a_backpack_never_gets_a_sku(self):
        detection = self._annotate("backpack")
        assert detection.sku is None
        assert detection.product_id is None
        assert detection.extra["recognition"] == {
            "state": "skipped",
            "reason": "class_not_a_product",
        }

    def test_a_person_never_gets_a_sku(self):
        detection = self._annotate("person")
        assert detection.sku is None
        assert detection.extra["recognition"]["reason"] == "not_a_product"

    @pytest.mark.parametrize("class_name", ["handbag", "suitcase", "umbrella", "tie"])
    def test_belongings_never_get_a_sku(self, class_name):
        assert self._annotate(class_name).sku is None
