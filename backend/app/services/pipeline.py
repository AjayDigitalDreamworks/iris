"""Per-camera processing pipeline.

One :class:`PipelineRunner` exists per active camera and holds that camera's
stateful components (tracker, stock estimator, background model, crossing
memory).  It is invoked once per sampled frame:

    Frame
      -> preprocess (inside the detector)
      -> detect
      -> recognise (SKU via embedding retrieval)
      -> track (ByteTrack)
      -> shelf occupancy / void analysis
      -> stock estimation
      -> customer analytics (line crossing, dwell)
      -> queue analytics
      -> business rules
      -> result object  (persisted + broadcast by the caller)

Everything is synchronous and CPU-bound by design: it runs on the camera's own
worker thread, keeping the asyncio event loop free for the API and WebSocket.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.alert_service import AlertService, RuleOutcome
from app.services.db_service import settings_service
from app.vision.anomaly import AnomalyCandidate, ShelfAnomalyDetector
from app.vision.embedder import ProductRecognizer
from app.vision.footfall import CustomerAnalytics, CustomerAnalyticsEngine
from app.vision.queue_analysis import QueueAnalyzer, QueueMeasurement
from app.vision.retail_detector import RetailProductDetector
from app.vision.scheduler import (
    DETECTION as SCHED_DETECTION,
)
from app.vision.scheduler import (
    FOOTFALL as SCHED_FOOTFALL,
)
from app.vision.scheduler import (
    PERSON_TRACKING as SCHED_TRACKING,
)
from app.vision.scheduler import (
    QUEUE as SCHED_QUEUE,
)
from app.vision.scheduler import (
    RECOGNITION as SCHED_RECOGNITION,
)
from app.vision.scheduler import (
    SHELF_ANALYSIS as SCHED_SHELF,
)
from app.vision.scheduler import InferenceScheduler
from app.vision.shelf_analysis import ShelfObservation, SlotOccupancyEstimator
from app.vision.stock import StockEstimate, StockEstimator
from app.vision.tracker import polygon_area, scaled_line, scaled_polygon
from app.vision.types import Detection, Frame, InferenceResult

logger = get_logger("app")
ai_log = get_logger("ai")

PRODUCT_TASK = "product"
PERSON_TASK = "person"


@dataclass
class ShelfConfig:
    shelf_id: str
    code: str
    capacity_units: int
    polygon: list[list[float]]
    zone_id: str | None = None
    camera_id: str | None = None
    store_id: str | None = None
    skus: tuple[str, ...] = ()


@dataclass
class CameraConfig:
    """Everything the runner needs, loaded from the database before it starts."""

    camera_id: str
    camera_name: str
    store_id: str | None
    role: str
    source_type: str
    is_demo: bool
    #: Frame delivery rate for this camera. Drives the inference scheduler's
    #: per-task frame intervals.
    capture_fps: float = 5.0
    shelves: list[ShelfConfig] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    queue_zones: list[dict[str, Any]] = field(default_factory=list)
    occupancy_zones: list[dict[str, Any]] = field(default_factory=list)
    detection_tasks: tuple[str, ...] = (PRODUCT_TASK, PERSON_TASK)
    confidence_threshold: float = settings.confidence_threshold
    run_recognition: bool = True


@dataclass
class PipelineResult:
    """Everything one processed frame produced."""

    camera_id: str
    frame_index: int
    captured_at: datetime
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    people: list[Detection] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    stock_estimates: list[StockEstimate] = field(default_factory=list)
    shelf_observations: list[ShelfObservation] = field(default_factory=list)
    customer: CustomerAnalytics | None = None
    queues: list[QueueMeasurement] = field(default_factory=list)
    #: Shelf conditions flagged for human review. Candidates, never verdicts.
    anomalies: list[AnomalyCandidate] = field(default_factory=list)
    #: What the scheduler asked each AI task to do on this frame, and what it did.
    scheduling: dict[str, Any] = field(default_factory=dict)
    rule_outcomes: list[RuleOutcome] = field(default_factory=list)
    latency_ms: float = 0.0
    inference_fps: float = 0.0
    model_id: str | None = None
    errors: list[str] = field(default_factory=list)
    is_demo: bool = False


class PipelineRunner:
    """Stateful per-camera processor."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        detector: Any = None,
        tracker: Any = None,
        recognizer: ProductRecognizer | None = None,
        alert_service: AlertService | None = None,
        anomaly_detector: ShelfAnomalyDetector | None = None,
        scheduler: InferenceScheduler | None = None,
    ) -> None:
        from app.vision.models_registry import registry
        from app.vision.tracker import ByteTrackWrapper

        self.config = config
        self.detector = detector if detector is not None else registry.get("product_detector")
        # Product detection runs through the retail chain (general detector +
        # the multipack-precedence rule from Retail-product-detection-main).
        # The multipack stage is skipped automatically when its weights are
        # absent, and the detector reports that it is general-only.
        self.retail_detector = RetailProductDetector(self.detector)
        # A tracker MUST be per camera: ByteTrack keeps track state internally,
        # so sharing one instance would mix identities across cameras and let a
        # camera with no detections expire another camera's tracks.
        self.tracker = (
            tracker if tracker is not None
            else ByteTrackWrapper(registry.spec("tracker"))
        )
        self.recognizer = recognizer
        # A fine-tuned detector names its own classes ("chips", "namkeen", …).
        # COCO has exactly 80 classes, so anything else is a custom model and
        # its classes must be treated as products, or every crop it produces
        # would be skipped as "not a product".
        if self.recognizer is not None:
            try:
                names = list(getattr(self.detector, "names", {}).values())
                if names and len(names) != 80:
                    self.recognizer.set_custom_classes(names)
            except Exception:  # pragma: no cover - defensive
                pass
        self.alert_service = alert_service
        self.stock_estimator = StockEstimator()
        # Shelf identity must not leak between cameras, so the detector is
        # per-camera like the tracker — its state is keyed by shelf id.
        self.anomaly_detector = (
            anomaly_detector if anomaly_detector is not None else ShelfAnomalyDetector()
        )
        # Per-camera task routing, from the camera's role and delivery rate.
        self.scheduler = (
            scheduler
            if scheduler is not None
            else InferenceScheduler(
                role=config.role,
                capture_fps=config.capture_fps or settings.default_capture_fps,
            )
        )
        # Configuration drives the task set as well as the role: a camera that
        # has a queue zone, an entrance line or a shelf must run the matching
        # task even when its role profile would otherwise skip it.
        if config.queue_zones:
            self.scheduler.ensure_task(SCHED_QUEUE, min(config.capture_fps or 5.0, 5.0))
        if config.lines:
            self.scheduler.ensure_task(SCHED_FOOTFALL, config.capture_fps or 5.0)
        if config.shelves:
            self.scheduler.ensure_task(SCHED_SHELF, min(config.capture_fps or 5.0, 5.0))
            self.scheduler.ensure_task(SCHED_RECOGNITION, min(config.capture_fps or 5.0, 2.0))

        self.shelf_estimator = SlotOccupancyEstimator()
        self.customer_engine = CustomerAnalyticsEngine(
            config.camera_id, occupancy_zones=config.occupancy_zones
        )
        self.customer_engine.configure(config.lines, config.occupancy_zones)
        self.queue_analyzer = QueueAnalyzer(config.camera_id, settings_service.queue_thresholds())
        self.queue_analyzer.configure(config.queue_zones)
        self._latencies: list[float] = []
        self._frames = 0
        self._started = time.time()
        self.errors: list[str] = []
        #: Latest frame's detections as plain dicts, for the Live overlay. This
        #: is the ONLY correct source for on-screen boxes: reading the stored
        #: detection history instead stacks boxes from many past frames onto one
        #: image, which looks like the model is detecting far more people than
        #: are actually visible.
        self.live_detections: list[dict[str, Any]] = []
        self.live_frame: dict[str, Any] = {}

    # ── main entry point ────────────────────────────────────────────────────
    def process(self, frame: Frame) -> PipelineResult:
        started = time.perf_counter()
        result = PipelineResult(
            camera_id=self.config.camera_id,
            frame_index=frame.index,
            captured_at=frame.captured_at,
            width=frame.width,
            height=frame.height,
            is_demo=frame.is_demo,
        )
        frame.camera_id = self.config.camera_id

        # 0) Task scheduling (README §3, Pipeline A step 5). Decided once per
        # delivered frame so every stage below sees the same decision.
        frame_index = self.scheduler.begin_frame()
        result.scheduling = self.scheduler.stats()

        # 1) Detection (real model inference). Detection is the base signal
        #    every other task consumes, so it is only ever skipped when the
        #    camera's own delivery rate is below the task's target.
        detection_ran = False
        if self.scheduler.allow(SCHED_DETECTION):
            detections, latency_ms = self._detect(frame, retail=True)
            result.detections = detections
            result.latency_ms = latency_ms
            result.model_id = getattr(self.detector, "model_id", None)
            self._latencies.append(latency_ms)
            detection_ran = True
        else:
            detections, latency_ms = [], result.latency_ms
        self._frames += 1

        # 2) Recognition (SKU assignment via catalog embeddings). Expensive, and
        #    a shelf changes slowly, so it runs on the scheduler's rate.
        if (
            self.scheduler.allow(SCHED_RECOGNITION)
            and self.config.run_recognition
            and self.recognizer
            and self.recognizer.is_ready
        ):
            try:
                self.recognizer.annotate(detections, frame.image)
            except Exception as exc:  # pragma: no cover - defensive
                self._note_error(f"recognition: {exc}")

        people = [d for d in detections if d.class_name == "person"]
        result.people = people

        # 3) Tracking: track all detected objects (both retail products and persons)
        if self.tracker is not None and self.scheduler.allow(SCHED_TRACKING):
            try:
                result.tracks = self.tracker.update(
                    detections, frame, camera_id=self.config.camera_id,
                    class_names=getattr(self.detector, "names", {}),
                )
            except Exception as exc:
                self._note_error(f"tracking: {exc}")
                result.tracks = []
        stamp_track_ids(detections, result.tracks)

        # 4) Shelf occupancy + stock estimation + anomaly detection.
        #    Grouped deliberately: occupancy, stock and anomalies must all be
        #    judged on the SAME observation, so they share one schedule slot.
        if self.config.shelves and self.scheduler.allow(SCHED_SHELF):
            self._process_shelves(frame, detections, people, result)

        # 5) Customer analytics (operates on person tracks only)
        person_tracks = [t for t in result.tracks if t.class_name == "person"]
        if self.scheduler.allow(SCHED_FOOTFALL):
            try:
                result.customer = self.customer_engine.update(
                    frame, person_tracks, self.tracker.states if self.tracker else {}
                )
            except Exception as exc:
                self._note_error(f"customer analytics: {exc}")

        # 6) Queue analytics (operates on person tracks only)
        if self.scheduler.allow(SCHED_QUEUE):
            try:
                result.queues = self.queue_analyzer.analyse(frame, person_tracks)
            except Exception as exc:
                self._note_error(f"queue analytics: {exc}")

        # 7) Business rules
        self._apply_rules(result)

        result.inference_fps = self.inference_fps
        result.errors = list(self.errors[-5:])
        self.publish_live(result, detection_ran=detection_ran)
        logger.debug(
            "Frame processed",
            extra={
                "camera_id": self.config.camera_id,
                "detections": len(detections),
                "latency_ms": round(latency_ms, 1),
            },
        )
        _ = time.perf_counter() - started
        return result

    # ── stages ──────────────────────────────────────────────────────────────
    def publish_live(self, result: PipelineResult, *, detection_ran: bool = True) -> None:
        """Record this frame's detections for the Live overlay endpoints.

        When the scheduler gated detection on this frame the overlay keeps the
        previous boxes instead of blanking: the boxes would otherwise flicker
        off on every gated frame.
        """
        boxes: list[dict[str, Any]] = []
        for detection in result.detections:
            retail = detection.extra.get("retail") if isinstance(detection.extra, dict) else None
            boxes.append(
                {
                    "class_name": detection.class_name,
                    "task": detection.task,
                    "confidence": round(float(detection.confidence), 4),
                    "x1": round(float(detection.bbox.x1), 2),
                    "y1": round(float(detection.bbox.y1), 2),
                    "x2": round(float(detection.bbox.x2), 2),
                    "y2": round(float(detection.bbox.y2), 2),
                    "track_id": detection.track_id,
                    "sku": detection.sku,
                    "is_person": detection.class_name == "person",
                    "is_multipack": bool(retail.get("multipack")) if retail else False,
                }
            )
        if detection_ran or not self.live_detections:
            self.live_detections = boxes
        self.live_frame = {
            "width": result.width,
            "height": result.height,
            "frame_index": result.frame_index,
            "captured_at": result.captured_at.isoformat(),
            "latency_ms": round(result.latency_ms, 2),
            "detections_fresh": detection_ran,
        }

    def _detect(self, frame: Frame, *, retail: bool = True) -> tuple[list[Detection], float]:
        engine = self.retail_detector if retail else self.detector
        try:
            detections, latency_ms = engine.predict(frame)
        except Exception as exc:  # pragma: no cover - detector already swallows
            self._note_error(f"detection: {exc}")
            return [], 0.0
        floor = max(self.config.confidence_threshold, settings.min_detection_confidence)
        return [d for d in detections if d.confidence >= floor], latency_ms

    def _process_shelves(
        self,
        frame: Frame,
        detections: list[Detection],
        people: list[Detection],
        result: PipelineResult,
    ) -> None:
        thresholds = settings_service.stock_thresholds()
        for shelf in self.config.shelves:
            try:
                observation = self.shelf_estimator.analyse(
                    frame.image,
                    shelf_id=shelf.shelf_id,
                    zone_id=shelf.zone_id,
                    camera_id=self.config.camera_id,
                    polygon_norm=shelf.polygon,
                    detections=detections,
                    capacity_units=shelf.capacity_units,
                )
            except Exception as exc:
                self._note_error(f"shelf analysis: {exc}")
                continue
            result.shelf_observations.append(observation)
            self._process_anomalies(frame, shelf, observation, people, result)

            for sku in shelf.skus:
                estimate = self.stock_estimator.estimate(
                    sku=sku,
                    detections=detections,
                    capacity_units=shelf.capacity_units,
                    shelf_id=shelf.shelf_id,
                    camera_id=self.config.camera_id,
                    occupancy_pct=observation.occupancy_pct,
                    thresholds=thresholds,
                    shelf_observation=observation,
                )
                result.stock_estimates.append(estimate)

    def _process_anomalies(
        self,
        frame: Frame,
        shelf: ShelfConfig,
        observation: Any,
        people: list[Detection],
        result: PipelineResult,
    ) -> None:
        """Stage 4b: flag shelf conditions that stock counting alone misses.

        Runs on the shelf ROI only, and reports candidates — a shelf that looks
        different is not proof that anything was taken.
        """
        try:
            polygon = scaled_polygon(shelf.polygon, frame.width, frame.height)
            if len(polygon) < 3:
                return
            mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
            roi = cv2.bitwise_and(frame.image, frame.image, mask=mask)
            # Keep only the shelf area so the descriptor describes the shelf and
            # not the aisle around it.
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                return
            roi = roi[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            occluder = person_cover_ratio(people, polygon)
            candidate = self.anomaly_detector.update(
                shelf.shelf_id,
                roi,
                occupancy_pct=observation.occupancy_pct,
                detected_units=observation.detected_units,
                camera_id=self.config.camera_id,
                zone_id=shelf.zone_id,
                occluder_ratio=occluder,
            )
            if candidate is not None:
                result.anomalies.append(candidate)
        except Exception as exc:
            self._note_error(f"anomaly detection: {exc}")

    def _apply_rules(self, result: PipelineResult) -> None:
        service = self.alert_service
        if service is None:
            return
        for estimate in result.stock_estimates:
            outcome = service.evaluate_stock(
                sku=estimate.sku,
                estimated_quantity=estimate.estimated_quantity,
                capacity_units=estimate.capacity_units,
                confidence=estimate.confidence,
                stock_percentage=estimate.stock_percentage,
                state=estimate.state,
                camera_id=self.config.camera_id,
                shelf_id=estimate.shelf_id,
            )
            if outcome.action != "none":
                result.rule_outcomes.append(outcome)

        for measurement in result.queues:
            # One reading can raise, escalate AND resolve (list): e.g. a queue
            # returning to normal resolves both its busy and critical alerts.
            for outcome in service.evaluate_queue(
                zone_id=measurement.zone_id,
                zone_name=measurement.zone_name,
                queue_length=measurement.queue_length,
                status=measurement.status.value,
                estimated_wait_seconds=measurement.estimated_wait_seconds,
                camera_id=self.config.camera_id,
                busy_threshold=measurement.busy_threshold,
                critical_threshold=measurement.critical_threshold,
            ):
                if outcome.action != "none":
                    result.rule_outcomes.append(outcome)

    # ── metrics ─────────────────────────────────────────────────────────────
    @property
    def inference_fps(self) -> float:
        elapsed = time.time() - self._started
        if elapsed <= 0:
            return 0.0
        return round(self._frames / elapsed, 2)

    @property
    def mean_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return round(sum(self._latencies) / len(self._latencies), 2)

    def p95_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return round(ordered[index], 2)

    def snapshot(self) -> dict[str, Any]:
        return {
            "camera_id": self.config.camera_id,
            "camera_name": self.config.camera_name,
            "frames": self._frames,
            "inference_fps": self.inference_fps,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms(),
            "model_id": result_model_id(self.detector),
            "device": getattr(self.detector, "device", "cpu"),
            "customer": {
                "entries": self.customer_engine.entries,
                "exits": self.customer_engine.exits,
                "lines": len(self.config.lines),
                "queue_zones": len(self.queue_analyzer.zones),
                "shelves": len(self.config.shelves),
            },
            # What each AI task was asked to do, and what it actually achieved.
            # Reported rather than assumed: the targets are configuration, the
            # achieved rates are measurements.
            "scheduling": self.scheduler.stats(),
            "errors": list(self.errors[-5:]),
            "started_at": datetime.fromtimestamp(self._started, tz=timezone.utc).isoformat(),
        }

    def reset(self) -> None:
        self._latencies.clear()
        self._frames = 0
        self._started = time.time()
        self.errors.clear()
        self.shelf_estimator.reset()
        self.stock_estimator.reset()
        self.customer_engine.reset()

    def _note_error(self, message: str) -> None:
        logger.warning(
            "Pipeline stage error",
            extra={
                "camera_id": self.config.camera_id,
                "camera_name": self.config.camera_name,
                "stage": message,
            },
        )
        self.errors.append(f"{datetime.now(tz=timezone.utc).isoformat()} {message}")


#: Minimum IoU for a tracked box to be considered the same object as a detection.
TRACK_LINK_IOU = 0.3


def _iou(a: Any, b: Any) -> float:
    """Intersection-over-union of two boxes with x1y1x2y2 corners."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stamp_track_ids(detections: Any, tracks: Any, min_iou: float = TRACK_LINK_IOU) -> int:
    """Write each track's id onto the detection it belongs to. Returns the count.

    Each track claims at most one detection (the best-matching one), so two
    overlapping people can never be given the same track id.
    """
    if not tracks:
        return 0
    claimed: set[int] = set()
    linked = 0
    for detection in detections:
        best_iou, best = 0.0, None
        for track in tracks:
            if track.track_id in claimed:
                continue
            score = _iou(detection.bbox, track.bbox)
            if score > best_iou:
                best_iou, best = score, track
        if best is not None and best_iou >= min_iou:
            detection.track_id = best.track_id
            claimed.add(best.track_id)
            linked += 1
    return linked


def person_cover_ratio(people: list[Any], polygon: list[tuple[float, float]]) -> float:
    """Fraction of a shelf polygon hidden behind person boxes on this frame.

    Used so that "the shelf looks different" and "somebody is standing in front
    of it" stay separate signals. Without this, a busy aisle raises appearance
    alerts all day.
    """
    if not people or len(polygon) < 3:
        return 0.0
    poly = np.array(polygon, dtype=np.float32)
    shape = (int(poly[:, 1].max()) + 2, int(poly[:, 0].max()) + 2)
    shelf_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(shelf_mask, [poly.astype(np.int32)], 1)
    shelf_pixels = int(shelf_mask.sum())
    if shelf_pixels == 0:
        return 0.0
    # Paint the people onto a COPY, using a DIFFERENT value (2) so the covered
    # pixels can be told apart from the shelf itself.
    covered = shelf_mask.copy()
    for person in people:
        box = person.bbox
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2 = min(shape[1] - 1, int(box.x2))
        y2 = min(shape[0] - 1, int(box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(covered, (x1, y1), (x2, y2), 2, thickness=-1)
    hidden = int(np.count_nonzero((covered == 2) & (shelf_mask == 1)))
    return min(1.0, hidden / float(shelf_pixels))


def result_model_id(detector: Any) -> str | None:
    return getattr(detector, "model_id", None)


def load_shelf_configs(
    camera_id: str, session: Any = None
) -> tuple[list["ShelfConfig"], str | None]:
    """Load shelf configurations (with their product SKUs) for a camera.

    Pass ``session`` to reuse an existing one; opening a second session inside
    an open transaction would contend for the same SQLite write lock.
    """
    from app.db.session import SessionLocal
    from app.models import Camera, Product, Shelf

    owns = session is None
    session = session or SessionLocal()
    try:
        camera = session.get(Camera, camera_id)
        if camera is None:
            return [], None
        store_id = camera.store_id
        zone_by_id = {zone.id: zone for zone in camera.zones}
        shelves: list[ShelfConfig] = []
        for shelf in session.query(Shelf).filter(Shelf.camera_id == camera_id).all():
            polygon = shelf.polygon
            zone_id = shelf.zone_id
            if not polygon and zone_id in zone_by_id:
                polygon = zone_by_id[zone_id].polygon
            products = (
                session.query(Product)
                .filter(Product.shelf_id == shelf.id, Product.is_active.is_(True))
                .all()
            )
            shelves.append(
                ShelfConfig(
                    shelf_id=shelf.id,
                    code=shelf.code,
                    capacity_units=shelf.capacity_units,
                    polygon=polygon or [],
                    zone_id=zone_id,
                    camera_id=camera_id,
                    store_id=shelf.store_id or store_id,
                    skus=tuple(p.sku for p in products),
                )
            )
        return shelves, store_id
    finally:
        if owns:
            session.close()


def scale_line_config(line: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Convenience used by the camera manager when building zone configs."""
    result = dict(line)
    scaled = scaled_line(line.get("line"), width, height)
    if scaled:
        result["line_px"] = scaled
    return result


def line_config_from_zone(zone: Any) -> dict[str, Any]:
    return {
        "zone_id": zone.id,
        "name": zone.name,
        "zone_type": zone.zone_type,
        "line": zone.line or [],
        "entry_on_positive": zone.zone_type != "exit_line",
    }


def queue_config_from_zone(zone: Any) -> dict[str, Any]:
    return {
        "zone_id": zone.id,
        "name": zone.name,
        "zone_type": zone.zone_type,
        "polygon": zone.polygon or [],
        "service_stations": zone.service_stations or 1,
    }


def zone_config_from_zone(zone: Any) -> dict[str, Any]:
    return {
        "zone_id": zone.id,
        "name": zone.name,
        "zone_type": zone.zone_type,
        "polygon": zone.polygon or [],
    }


_ = InferenceResult  # re-exported type used by API schemas