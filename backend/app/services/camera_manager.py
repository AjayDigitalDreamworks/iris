"""Camera manager: stream lifecycle, worker threads, health and streaming.

One :class:`CameraWorker` thread per active camera.  Capture is separated from
inference so a slow model never blocks frame reading, and every queue is bounded
so memory cannot grow without limit.

A failing camera is isolated: exceptions are caught, logged, published as a
``camera.status`` event and retried with bounded exponential backoff.  It can
never take the backend down.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import NotFoundError, SourceUnavailableError, ValidationError
from app.core.logging import get_logger, set_camera_id
from app.core.security import decrypt_secret, redact_url
from app.db.session import SessionLocal
from app.models import (
    AlertType,
    Camera,
    CameraStatus,
    Detection,
    DetectionSource,
    FootfallEvent,
    Product,
    QueueSnapshot,
    ZoneType,
)
from app.services.alert_service import alert_service
from app.services.db_service import run_in_session, settings_service
from app.services.pipeline import (
    CameraConfig,
    PipelineRunner,
    line_config_from_zone,
    queue_config_from_zone,
    zone_config_from_zone,
)
from app.vision.ingestion import LatestFrameBuffer, build_source
from app.vision.models_registry import detect_hardware, registry
from app.vision.types import Frame

camera_log = get_logger("camera")
app_log = get_logger("app")

#: The multipart boundary used for MJPEG streams. The part delimiter written by
#: :func:`build_multipart_frame` MUST match the boundary declared in the HTTP
#: ``Content-Type``; a mismatch makes the browser treat the whole body as a
#: single part and the feed freezes on its first frame.
MJPEG_BOUNDARY = "irisframe"

#: Caps how many cameras run model inference at once. On a CPU box, letting
#: every camera use all cores makes them ALL slow; a small cap raises the total
#: number of frames analysed per second. Streaming is unaffected — capture runs
#: on its own thread and never waits for this gate.
_INFERENCE_GATE = threading.Semaphore(max(1, settings.inference_max_concurrency))


# ── Health ───────────────────────────────────────────────────────────────────
@dataclass
class CameraHealth:
    """Live health for one camera, merged from source + pipeline."""

    camera_id: str
    name: str
    status: str = CameraStatus.OFFLINE.value
    is_enabled: bool = True
    seconds_since_last_frame: float | None = None
    capture_fps: float = 0.0
    inference_fps: float = 0.0
    inference_latency_ms: float | None = None
    queue_depth: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    width: int | None = None
    height: int | None = None
    last_error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None

    @property
    def seconds_since_start(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, time.time() - self.started_at)

    @property
    def is_stale(self) -> bool:
        stale_after = settings_service.get_float("camera_offline_alert_seconds", 60.0)
        if self.started_at is None:
            return False
        if self.seconds_since_last_frame is None:
            return True
        return self.seconds_since_last_frame > stale_after

    @property
    def is_frozen(self) -> bool:
        return bool(self.detail.get("is_frozen"))

    @property
    def drop_ratio(self) -> float:
        total = self.frames_processed + self.frames_dropped
        return round(self.frames_dropped / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "status": self.status,
            "is_enabled": self.is_enabled,
            "is_stale": self.is_stale,
            "is_frozen": self.is_frozen,
            "seconds_since_last_frame": (
                round(self.seconds_since_last_frame, 1)
                if self.seconds_since_last_frame is not None
                else None
            ),
            "capture_fps": round(self.capture_fps, 2),
            "inference_fps": round(self.inference_fps, 2),
            "inference_latency_ms": self.inference_latency_ms,
            "queue_depth": self.queue_depth,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "drop_ratio": self.drop_ratio,
            "reconnect_count": self.reconnect_count,
            "width": self.width,
            "height": self.height,
            "last_error": self.last_error,
            "detail": self.detail,
        }


#: Upper bound on buffered point-in-time analytics awaiting a DB write.
_MAX_PENDING_ANALYTICS = 2000

#: How long a worker may run without ever delivering a frame before its health
#: is reported as an error rather than `online`. A webcam with no device
#: attached retries forever and never regains control to correct itself, so
#: without this it would keep reporting a stale `online`.
NO_FRAME_GRACE_SECONDS = 20.0


# ── Configuration loading ────────────────────────────────────────────────────
def load_camera_config(camera_id: str, session: Any = None) -> CameraConfig:
    """Build a :class:`CameraConfig` from persisted configuration."""
    from app.services.pipeline import load_shelf_configs

    owns = session is None
    session = session or SessionLocal()
    try:
        camera = session.get(Camera, camera_id)
        if camera is None:
            raise NotFoundError(f"Camera {camera_id!r} does not exist.")
        zones = [z for z in camera.zones if z.is_active]
        shelves, store_id = load_shelf_configs(camera_id, session=session)
        lines = [line_config_from_zone(z) for z in zones
                 if z.zone_type in (ZoneType.ENTRANCE_LINE.value, ZoneType.EXIT_LINE.value)]
        queue_zones = [queue_config_from_zone(z) for z in zones if z.zone_type == ZoneType.QUEUE.value]
        occupancy_zones = [zone_config_from_zone(z) for z in zones if z.zone_type == ZoneType.ZONE.value]
        return CameraConfig(
            camera_id=camera_id,
            camera_name=camera.name,
            store_id=store_id,
            role=camera.role,
            source_type=camera.source_type,
            is_demo=camera.source_type == "synthetic",
            capture_fps=camera.capture_fps or settings.default_capture_fps,
            shelves=shelves,
            lines=lines,
            queue_zones=queue_zones,
            occupancy_zones=occupancy_zones,
            confidence_threshold=max(camera_confidence(), settings.confidence_threshold),
        )
    finally:
        if owns:
            session.close()


def camera_confidence() -> float:
    return settings_service.get_float("detection_confidence", settings.confidence_threshold)


def resolve_stream_url(camera: Camera) -> str | None:
    """Decrypt a camera's stream URL. Never logged, never returned by the API."""
    return decrypt_secret(camera.stream_url_encrypted)


def build_source_for_camera(camera: Camera):
    """Instantiate the :class:`FrameSource` for a camera record."""
    spec: dict[str, Any] = {
        "source_type": camera.source_type,
        "capture_fps": camera.capture_fps or settings.default_capture_fps,
        "frame_skip": max(1, camera.frame_skip or 1),
        "reconnect_delay": camera.reconnect_delay,
        "reconnect_max_delay": camera.reconnect_max_delay,
        "file_path": camera.file_path,
        "image_path": camera.image_path,
    }
    if camera.source_type in ("rtsp", "url", "youtube", "ip_webcam"):
        url = resolve_stream_url(camera)
        if not url:
            raise ValidationError(
                f"This {camera.source_type} camera has no stored stream URL."
            )
        spec["stream_url"] = url
    if camera.source_type in ("url", "youtube", "video_file"):
        # A prerecorded clip should LOOP and play at wall-clock speed, not end
        # and reopen every few seconds (which made the feed flicker and re-decode
        # from frame 0), nor play in slow motion.
        spec["loop"] = True
    if camera.source_type == "webcam":
        # Device index comes from ``file_path`` so the UI has one field for it:
        # "0", "1", "2" … (a bare value means the default camera, index 0).
        # parse_webcam_index raises a clear error for a non-numeric value.
        spec["device_index"] = camera.file_path or 0
    return build_source(spec)


def prepare_frame_for_pipeline(frame: Frame) -> None:
    """Downscale an oversized frame in place, once, at capture.

    Phone/board clips are often 2.7K or larger. JPEG-encoding that for the live
    feed is far too slow on CPU to feel live, and the detector resizes to 640
    internally anyway. Downscaling here speeds up streaming AND inference, and
    keeps every downstream coordinate (detections, overlay, health) in one
    consistent space.
    """
    limit = settings.frame_max_width
    if not limit or limit <= 0:
        return
    height, width = frame.image.shape[:2]
    if width <= limit or height <= 0:
        return
    scale = limit / float(width)
    size = (limit, max(1, int(round(height * scale))))
    frame.image = cv2.resize(frame.image, size, interpolation=cv2.INTER_LINEAR)


# ── Worker ───────────────────────────────────────────────────────────────────
class CameraWorker:
    """Runs one camera end-to-end on its own thread."""

    def __init__(self, camera_id: str, manager: "CameraManager") -> None:
        self.camera_id = camera_id
        self.manager = manager
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.frame_buffer = LatestFrameBuffer(maxsize=settings.frame_queue_size)
        self.health = CameraHealth(camera_id=camera_id, name=camera_id)
        self.runner: PipelineRunner | None = None
        self.source_status: dict[str, Any] = {}
        self._last_signature: float | None = None
        self._last_signature_change: float | None = None
        self._last_persist = 0.0
        self.persist_interval = 5.0
        # Transient, point-in-time analytics (crossings, queue readings, track
        # rows) are produced on EVERY frame but only flushed every
        # ``persist_interval`` seconds. Buffering them here is what stops a
        # crossing that happened between two writes from being lost forever.
        self._pending_crossings: list[Any] = []
        self._pending_queues: list[Any] = []
        self._pending_tracks: list[Any] = []
        self._offline_alerted = False
        self._was_offline = False
        self._latest_annotated_frame: Frame | None = None
        self._latest_analysed_frame: Frame | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self.thread and self.thread.is_alive():
            return False
        self.stop_event.clear()
        # stop() closes the frame buffer, and a restart reuses this same
        # worker — so without a fresh buffer a restarted camera would feed zero
        # frames forever (the MJPEG stream stayed on its last frame).
        self.frame_buffer = LatestFrameBuffer(maxsize=settings.frame_queue_size)
        self._last_signature = None
        self._last_signature_change = None
        self.thread = threading.Thread(
            target=self._run, name=f"iris-camera-{self.camera_id[:8]}", daemon=True
        )
        self.thread.start()
        self.health.started_at = time.time()
        camera_log.info("Camera worker started", extra={"camera_id": self.camera_id})
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.frame_buffer.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        self.health.status = "stopped"
        camera_log.info("Camera worker stopped", extra={"camera_id": self.camera_id})

    @property
    def is_running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    # ── main loop ───────────────────────────────────────────────────────────
    def _run(self) -> None:
        set_camera_id(self.camera_id)
        session_factory = SessionLocal
        while not self.stop_event.is_set():
            try:
                session = session_factory()
                try:
                    camera = session.get(Camera, self.camera_id)
                    if camera is None or not camera.is_enabled:
                        self.health.status = CameraStatus.DISABLED.value
                        if self.stop_event.wait(2.0):
                            return
                        continue
                    config = load_camera_config(self.camera_id, session=session)
                    source = build_source_for_camera(camera)
                finally:
                    # The configuration is a plain snapshot; never hold a session
                    # for the lifetime of a stream.
                    session.close()

                self.health.name = camera.name
                self.health.is_enabled = True
                runner = PipelineRunner(
                    config, recognizer=self.manager.recognizer, alert_service=alert_service
                )
                self.runner = runner
                self._consume(source, runner, camera, session_factory)
                if self.stop_event.is_set():
                    return
                if not source.is_finite_source():
                    self.health.status = CameraStatus.ERROR.value
                    if self.stop_event.wait(3.0):
                        return
            except Exception as exc:  # pragma: no cover - last-resort isolation
                self._handle_failure(exc)
                if self.stop_event.wait(3.0):
                    return
        self.health.status = "stopped"

    def _consume(self, source: Any, runner: PipelineRunner, camera: Camera, session_factory: Any) -> None:
        """Stream frames and analyse them on independent threads.

        Capture must never wait for inference. A 270 ms detector on a 5 fps
        stream would otherwise cap the *stream* at the detector's rate and make
        a live feed look like a slideshow. So one thread reads frames and keeps
        the freshest one for the MJPEG/snapshot endpoints, while this thread
        analyses whatever frame is newest. Inference falls behind the stream
        instead of holding it up; stale frames are dropped, never queued, so
        latency cannot grow.
        """
        self.health.status = CameraStatus.CONNECTING.value
        self._persist_status(camera, CameraStatus.CONNECTING.value, session_factory)

        inference_queue: LatestFrameBuffer = LatestFrameBuffer(maxsize=1)
        capture_done = threading.Event()

        def _capture() -> None:
            set_camera_id(self.camera_id)
            try:
                for frame in source.frames():
                    if self.stop_event.is_set():
                        break
                    frame.is_demo = camera.source_type == "synthetic"
                    prepare_frame_for_pipeline(frame)
                    self._update_health(source, frame, runner)
                    # Publish for streaming FIRST, before any analysis, so the
                    # dashboard always shows the freshest pixel available.
                    self.frame_buffer.put(frame)
                    inference_queue.put(frame)
            except Exception as exc:  # pragma: no cover - defensive
                self.health.detail["capture_error"] = f"{type(exc).__name__}: {exc}"
                camera_log.error(
                    "Capture thread failed",
                    extra={"camera_id": self.camera_id, "error": str(exc)},
                )
            finally:
                capture_done.set()

        capture_thread = threading.Thread(
            target=_capture, name=f"iris-capture-{self.camera_id[:8]}", daemon=True
        )
        capture_thread.start()

        try:
            while not self.stop_event.is_set():
                frame = inference_queue.get(timeout=0.5)
                if frame is None:
                    if capture_done.is_set():
                        break
                    continue
                self._analyse(frame, runner, camera, source, session_factory)
        finally:
            capture_thread.join(timeout=3.0)
            source.close()
        self.health.status = CameraStatus.OFFLINE.value

    def _analyse(
        self,
        frame: Frame,
        runner: PipelineRunner,
        camera: Camera,
        source: Any,
        session_factory: Any,
    ) -> None:
        """Analyse one frame. Runs on the worker thread, never the capture one."""
        # A failure while analysing ONE frame must never tear down the stream.
        # The pixels are still valid and the operator still needs to see the
        # camera, so isolate the error, keep counting frames and carry on.
        try:
            with _INFERENCE_GATE:
                result = runner.process(frame)
        except Exception as exc:  # pragma: no cover - defensive
            self.health.detail["process_error"] = f"{type(exc).__name__}: {exc}"
            camera_log.error(
                "Frame processing failed; keeping the stream alive",
                extra={"camera_id": self.camera_id, "error": str(exc)},
            )
            return

        self._latest_analysed_frame = frame
        annotated_frame = self._annotate_frame(frame)
        self._latest_annotated_frame = annotated_frame
        self.frame_buffer.put(annotated_frame)

        self.health.frames_processed += 1
        self.health.inference_fps = result.inference_fps
        # On frames where the scheduler gated detection the per-frame latency is
        # 0, which reads on the dashboard as "instant". Fall back to the
        # detector's measured mean so the number always means something.
        self.health.inference_latency_ms = round(
            result.latency_ms or runner.mean_latency_ms or 0.0, 2
        )
        self.health.queue_depth = self.frame_buffer.depth
        # "Dropped" must mean frames the SOURCE produced but never handed on
        # (i.e. we could not keep up). Frames the latest-frame buffer replaces
        # are a different thing: keeping only the freshest frame is the
        # intended freshness-over-completeness behaviour, so counting those as
        # drops reported a ~95% drop ratio on a perfectly healthy camera. They
        # are still reported, as an explicit diagnostic rather than a metric.
        self.health.frames_dropped = max(
            0, source.status.frames_read - source.status.frames_yielded
        )
        self.health.detail["frames_superseded"] = self.frame_buffer.dropped
        self.health.detail["analysis_lag_frames"] = self.frame_buffer.dropped

        # Buffer point-in-time analytics on every analysed frame (not only the
        # persisted one) so nothing observed in between is dropped.
        if result.customer is not None:
            self._pending_crossings.extend(result.customer.crossings)
            self._pending_tracks.extend(
                t for t in result.tracks if t.class_name == "person"
            )
        self._pending_queues.extend(result.queues)
        # Bound the buffers so a failing DB cannot grow them without limit.
        for buffer in (
            self._pending_crossings,
            self._pending_queues,
            self._pending_tracks,
        ):
            if len(buffer) > _MAX_PENDING_ANALYTICS:
                del buffer[: len(buffer) - _MAX_PENDING_ANALYTICS]

        now = time.time()
        if now - self._last_persist >= self.persist_interval:
            self._last_persist = now
            try:
                self._persist_result(result, camera, session_factory)
            except Exception as exc:  # pragma: no cover - DB hiccup
                camera_log.error("Failed to persist frame results", exc_info=exc)
                self.health.detail["persist_error"] = str(exc)

        if not self._offline_alerted:
            self._offline_alerted = True
            self._was_offline = False

    # ── health ──────────────────────────────────────────────────────────────
    def _update_health(self, source: Any, frame: Frame, runner: PipelineRunner) -> None:
        self.health.status = CameraStatus.ONLINE.value
        self.health.seconds_since_last_frame = 0.0
        self.health.capture_fps = source.status.delivered_fps or source.status.source_fps
        self.health.width = frame.width
        self.health.height = frame.height
        self.health.reconnect_count = source.status.reconnect_count
        self.health.last_error = None
        self.source_status = {
            "frames_read": source.status.frames_read,
            "frames_yielded": source.status.frames_yielded,
            "state": source.status.state,
            "source_type": source.source_type,
            # Whether a prerecorded source is playing at wall-clock speed, and
            # the measured speed ratio, so the dashboard can say "realtime"
            # honestly instead of implying it.
            "realtime": source.status.realtime,
            "realtime_ratio": source.status.realtime_ratio,
            "frame_skip": getattr(source, "frame_skip", 1),
        }
        # Frozen-frame detection
        signature = frame.signature
        if self._last_signature is None or abs(signature - self._last_signature) > 0.05:
            self._last_signature = signature
            self._last_signature_change = time.time()
        frozen_age = (
            time.time() - self._last_signature_change if self._last_signature_change else 0.0
        )
        self.health.detail["is_frozen"] = frozen_age > settings.frozen_frame_seconds
        self.health.detail["signature_age_seconds"] = round(frozen_age, 1)
        self.health.detail.update(self.source_status)

    def _handle_failure(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self.health.status = CameraStatus.ERROR.value
        self.health.last_error = message
        self.health.detail["error"] = message
        camera_log.error(
            "Camera worker error: %s",
            message,
            extra={"camera_id": self.camera_id, "error": message},
        )

    def _persist_status(
        self, camera: Camera, status: str, session_factory: Any
    ) -> None:
        def _write(session) -> None:
            row = session.get(Camera, self.camera_id)
            if row is None:
                return
            row.status = status
            row.last_connected_at = datetime.now(tz=timezone.utc)
            session.flush()

        try:
            run_in_session(_write, session_factory)
        except Exception:  # pragma: no cover
            pass

    # ── persistence ─────────────────────────────────────────────────────────
    def _persist_result(self, result: Any, camera: Camera, session_factory: Any) -> None:
        from app.services.inventory_service import inventory_service

        source_label = DetectionSource.DEMO.value if result.is_demo else DetectionSource.LIVE.value
        now = datetime.now(tz=timezone.utc)

        def _write(session) -> None:
            row = session.get(Camera, self.camera_id)
            if row is not None:
                row.last_frame_at = result.captured_at
                row.total_frames += len(result.detections)
                row.status = CameraStatus.ONLINE.value
                if row.width is None:
                    row.width, row.height = result.width, result.height
                session.flush()

            for detection in result.detections:
                session.add(
                    Detection(
                        camera_id=self.camera_id,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        task="person" if detection.class_name == "person" else "product",
                        product_id=detection.product_id,
                        sku=detection.sku,
                        similarity=detection.similarity,
                        x1=detection.bbox.x1,
                        y1=detection.bbox.y1,
                        x2=detection.bbox.x2,
                        y2=detection.bbox.y2,
                        confidence=detection.confidence,
                        track_id=detection.track_id,
                        frame_timestamp=result.captured_at,
                        source=source_label,
                        model_id=result.model_id,
                        frame_index=result.frame_index,
                        inference_latency_ms=round(result.latency_ms, 3),
                    )
                )

            for crossing in self._pending_crossings:
                session.add(
                    FootfallEvent(
                        camera_id=self.camera_id,
                        zone_id=crossing.zone_id,
                        track_uid=crossing.track_uid,
                        external_id=crossing.external_id,
                        event_type=crossing.event_type,
                        direction=crossing.direction,
                        crossing_x=crossing.x,
                        crossing_y=crossing.y,
                        confidence=crossing.confidence,
                        occurred_at=datetime.fromtimestamp(
                            crossing.occurred_at, tz=timezone.utc
                        )
                        if crossing.occurred_at > 1e9
                        else now,
                    )
                )
            # The pending buffer holds one entry per track per buffered frame, so
            # the same track id can appear many times. Sessions run with
            # autoflush=False, so every one of those repeats would miss the
            # existing-row lookup and emit a duplicate INSERT, violating the
            # (camera_id, track_uid, external_id) uniqueness constraint and
            # rolling back the whole frame's analytics. Collapse to the newest
            # sample per track and count how many buffered frames it survived.
            track_observations = collapse_tracks(self._pending_tracks)
            for track, seen_count in track_observations.values():
                _persist_track(session, self.camera_id, track, now, seen_count)

            for measurement in self._pending_queues:
                session.add(
                    QueueSnapshot(
                        camera_id=self.camera_id,
                        zone_id=measurement.zone_id,
                        queue_length=measurement.queue_length,
                        service_zone_occupancy=measurement.service_zone_occupancy,
                        service_stations=measurement.service_stations,
                        estimated_wait_seconds=measurement.estimated_wait_seconds,
                        status=measurement.status.value,
                        confidence=measurement.confidence,
                        measured_at=now,
                    )
                )
            return True

        run_in_session(_write, session_factory)
        self._pending_crossings.clear()
        self._pending_queues.clear()
        self._pending_tracks.clear()

        if result.stock_estimates:
            inventory_service.record_estimates(
                result.stock_estimates,
                store_id=camera.store_id,
                source="demo_camera_estimate" if result.is_demo else "camera_estimate",
                session_factory=session_factory,
            )

        self._apply_alerts(result, camera)

    def _apply_alerts(self, result: Any, camera: Camera) -> None:
        from app.services.event_service import EventType

        self._apply_anomaly_alerts(result, camera)

        for outcome in result.rule_outcomes:
            sku = outcome.payload.get("sku")
            product = self._product_for_sku(sku) if sku else None
            alert = alert_service.apply(
                outcome,
                camera_id=self.camera_id,
                shelf_id=outcome.payload.get("shelf_id"),
                zone_id=outcome.payload.get("zone_id"),
                sku=sku,
                product_id=product.id if product else None,
                store_id=camera.store_id,
            )
            if alert is None:
                continue
            self.manager.publish_alert_event(alert, outcome.alert_type or "", camera.name)

        _ = EventType

    def _apply_anomaly_alerts(self, result: Any, camera: Camera) -> None:
        """Turn shelf anomaly candidates into reviewable alerts with evidence.

        The evidence image is the frame the decision was made from, not a
        reconstruction: "this shelf looks odd" only becomes actionable if staff
        can see what it looked like.
        """
        from app.services.alert_service import RuleOutcome
        from app.vision.anomaly import save_evidence

        for candidate in getattr(result, "anomalies", []) or []:
            evidence_path = None
            if settings.evidence_enabled:
                evidence_path = save_evidence(
                    str(settings.evidence_dir), candidate, self._evidence_frame(result)
                )
            outcome = RuleOutcome(
                action="raise",
                alert_type=AlertType.ANOMALY_CANDIDATE.value,
                severity=candidate.severity.value,
                title=f"Shelf anomaly candidate: {candidate.kind.value.replace('_', ' ')}",
                message=candidate.message,
                payload={
                    "kind": candidate.kind.value,
                    "score": candidate.score,
                    # Read by AlertService.apply to populate the alert columns.
                    "confidence": candidate.score,
                    "observations": candidate.consecutive,
                    "consecutive_observations": candidate.consecutive,
                    "context": candidate.context,
                    # Stated plainly so no reader mistakes this for an incident:
                    # the system saw a visual change, nothing more.
                    "interpretation": (
                        "Visual change candidate for staff review. This is not "
                        "evidence of theft or of any person's intent."
                    ),
                },
                dedup_key=f"anomaly:{candidate.shelf_id}:{candidate.kind.value}",
                reason=candidate.kind.value,
            )
            alert = alert_service.apply(
                outcome,
                camera_id=self.camera_id,
                shelf_id=candidate.shelf_id,
                zone_id=candidate.zone_id,
                sku=None,
                product_id=None,
                store_id=camera.store_id,
                evidence_path=evidence_path,
            )
            if alert is None:
                continue
            self.manager.publish_alert_event(alert, outcome.alert_type or "", camera.name)

    def _evidence_frame(self, result: Any) -> Any:
        """The newest buffered frame, used as the anomaly's visual evidence."""
        return self.frame_buffer.latest

    def _product_for_sku(self, sku: str) -> Product | None:
        def _load(session) -> Product | None:
            return session.query(Product).filter(Product.sku == sku).first()

        try:
            return run_in_session(_load)
        except Exception:  # pragma: no cover
            return None

    # ── streaming ───────────────────────────────────────────────────────────
    def latest_frame(self, timeout: float = 2.0, annotated: bool = False) -> Frame | None:
        if annotated and self._latest_annotated_frame is not None:
            return self._latest_annotated_frame
        if not annotated and self._latest_analysed_frame is not None:
            return self._latest_analysed_frame
        frame = self.frame_buffer.get_latest(timeout=timeout)
        if frame is None or not annotated:
            return frame
        return self._annotate_frame(frame)

    def _annotate_frame(self, frame: Frame) -> Frame:
        """Render active detections, tracks, and labels directly onto a frame copy."""
        runner = self.runner
        if runner is None or not runner.live_detections:
            return frame

        annotated_img = frame.image.copy()
        h, w = annotated_img.shape[:2]

        for det in runner.live_detections:
            try:
                x1 = max(0, min(w - 1, int(round(float(det.get("x1", 0))))))
                y1 = max(0, min(h - 1, int(round(float(det.get("y1", 0))))))
                x2 = max(0, min(w - 1, int(round(float(det.get("x2", 0))))))
                y2 = max(0, min(h - 1, int(round(float(det.get("y2", 0))))))
                if x2 <= x1 or y2 <= y1:
                    continue

                conf = float(det.get("confidence", 0.0))
                label = str(det.get("sku") or det.get("class_name") or "object")
                track_id = det.get("track_id")
                is_person = bool(det.get("is_person", False))
                is_multipack = bool(det.get("is_multipack", False))

                if is_person:
                    box_color = (21, 204, 250)      # Yellow/Gold (#FACC15 BGR)
                    badge_bg = (16, 160, 200)
                elif is_multipack:
                    box_color = (8, 179, 234)       # Amber/Orange (#EAB308 BGR)
                    badge_bg = (6, 140, 185)
                else:
                    box_color = (153, 211, 52)      # Emerald green (#34D399 BGR)
                    badge_bg = (50, 140, 30)

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)

                conf_pct = int(round(conf * 100))
                text = f"{label} {conf_pct}%"
                if track_id is not None:
                    text += f" #{track_id}"
                if is_multipack:
                    text += " [PACK]"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.40
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                badge_y1 = max(0, y1 - text_h - 6)
                badge_y2 = y1
                badge_x2 = min(w, x1 + text_w + 8)
                cv2.rectangle(annotated_img, (x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
                cv2.rectangle(annotated_img, (x1, badge_y1), (badge_x2, badge_y2), box_color, 1)

                cv2.putText(
                    annotated_img,
                    text,
                    (x1 + 4, badge_y2 - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                if track_id is not None:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(annotated_img, (cx, cy), 3, box_color, -1)
            except Exception:
                continue

        return Frame(
            image=annotated_img,
            index=frame.index,
            timestamp=frame.timestamp,
            source_fps=frame.source_fps,
            camera_id=frame.camera_id,
            is_demo=frame.is_demo,
        )

    def latest_detections(self) -> dict[str, Any]:
        """Live detections for the newest ANALYSED frame.

        Deliberately not read from the detection history: that stacks boxes
        from many past frames onto the current image and makes the model look
        like it detects far more people than are on screen.
        """
        runner = self.runner
        if runner is None:
            return {"camera_id": self.camera_id, "detections": [], "frame": {}}
        return {
            "camera_id": self.camera_id,
            "detections": runner.live_detections,
            "frame": runner.live_frame,
        }


def collapse_tracks(tracks: Any) -> dict[Any, tuple[Any, int]]:
    """Collapse buffered per-frame track samples to one entry per track id.

    The worker buffers one track sample per tracked person per frame, so the
    same ``track_id`` appears many times between two database writes. Sessions
    run with ``autoflush=False``, so persisting those repeats makes every
    lookup miss the already-pending row and emit a duplicate INSERT, which
    violates the ``(camera_id, track_uid, external_id)`` uniqueness constraint
    and rolls back every other analytic from the same frame.

    Returns ``{track_id: (newest_sample, how_many_frames_it_survived)}`` so no
    information about observation counts is lost.
    """
    collapsed: dict[Any, tuple[Any, int]] = {}
    for track in tracks:
        previous = collapsed.get(track.track_id)
        collapsed[track.track_id] = (track, (previous[1] + 1) if previous else 1)
    return collapsed


def _persist_track(
    session: Any, camera_id: str, track: Any, now: datetime, frames_seen: int = 1
) -> None:
    """Insert or refresh one anonymous track row.

    Exactly one of insert/update is performed: calling ``session.add`` and
    ``session.merge`` on the same new row would emit two INSERTs and violate
    the (camera_id, track_uid, external_id) uniqueness constraint.
    """
    from app.models import Track

    track_uid = f"{camera_id}:{track.track_id}"
    row = (
        session.query(Track)
        .filter(
            Track.camera_id == camera_id,
            Track.track_uid == track_uid,
            Track.external_id == track.track_id,
        )
        .first()
    )
    if row is None:
        session.add(
            Track(
                camera_id=camera_id,
                track_uid=track_uid,
                external_id=track.track_id,
                class_name=track.class_name,
                confidence=track.confidence,
                first_seen=now,
                last_seen=now,
                frames_seen=frames_seen,
                is_active=True,
            )
        )
        return
    row.last_seen = now
    row.confidence = track.confidence
    row.frames_seen += frames_seen
    row.is_active = True


# ── Manager ──────────────────────────────────────────────────────────────────
class CameraManager:
    """Owns every camera worker."""

    def __init__(self) -> None:
        self.workers: dict[str, CameraWorker] = {}
        self._lock = threading.RLock()
        self._recognizer: Any = None
        self.total_starts = 0
        self.total_stops = 0
        self.total_reconnects = 0

    # ── recognizer ──────────────────────────────────────────────────────────
    @property
    def recognizer(self):
        if self._recognizer is None:
            try:
                from app.vision.embedder import ProductRecognizer

                embedder = registry.get("embedder")
                self._recognizer = ProductRecognizer(embedder)
                self.reload_recognizer_catalog()
            except Exception as exc:  # pragma: no cover - weights missing
                camera_log.warning("Recognizer unavailable", exc_info=exc)
        return self._recognizer

    def reload_recognizer_catalog(self) -> int:
        """Load product reference embeddings into the recognizer index."""
        recognizer = self._recognizer
        if recognizer is None or not settings.embedding_enabled:
            return 0
        from app.vision.embedder import CatalogEmbedding, build_entry
        from app.vision.detector import UltralyticsDetector  # noqa: F401  (crop helper)
        from app.models import ProductEmbedding

        def _load(session) -> list[tuple]:
            rows = (
                session.query(ProductEmbedding, Product)
                .join(Product, Product.id == ProductEmbedding.product_id)
                .filter(Product.is_active.is_(True))
                .all()
            )
            return [(embedding.product_id, product.sku, product.name, embedding.vector,
                     embedding.model_id, embedding.source_image) for embedding, product in rows]

        try:
            records = run_in_session(_load)
        except Exception as exc:  # pragma: no cover
            camera_log.warning("Could not load embedding catalog", exc_info=exc)
            return 0

        entries = [
            CatalogEmbedding(
                product_id=r[0], sku=r[1], name=r[2],
                vector=np.asarray(r[3], dtype=np.float32),
                model_key=self._recognizer.embedder.model_key,
            )
            for r in records
        ]
        recognizer.load_entries(entries)
        camera_log.info("Recognition catalog loaded", extra={"products": len(entries)})
        return len(entries)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def worker(self, camera_id: str) -> CameraWorker:
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker is None:
                worker = CameraWorker(camera_id, self)
                self.workers[camera_id] = worker
            return worker

    def is_running(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        return bool(worker and worker.is_running)

    def start(self, camera_id: str) -> bool:
        worker = self.worker(camera_id)
        started = worker.start()
        if started:
            self.total_starts += 1
            self.set_db_status(camera_id, CameraStatus.CONNECTING.value)
        return started

    def stop(self, camera_id: str) -> None:
        worker = self.workers.get(camera_id)
        if worker:
            worker.stop()
            self.total_stops += 1
            self.set_db_status(camera_id, CameraStatus.OFFLINE.value)

    def restart(self, camera_id: str) -> bool:
        """Reload a camera pipeline.

        Callers in request handlers must defer this past the request's commit
        (FastAPI ``BackgroundTasks``). Calling it inline deadlocks the single
        SQLite writer: the request session still holds its write transaction
        while the freshly started worker tries to persist a frame, so the
        worker blocks until the busy-timeout expires and the API call stalls
        for minutes.
        """
        self.stop(camera_id)
        return self.start(camera_id)

    def start_all(self, store_id: str | None = None) -> int:
        """Start every enabled camera. Failures are isolated per camera."""
        def _load(session) -> list[Camera]:
            query = session.query(Camera).filter(Camera.is_enabled.is_(True))
            if store_id:
                query = query.filter(Camera.store_id == store_id)
            return query.all()

        cameras = run_in_session(_load)
        started = 0
        for camera in cameras:
            try:
                if self.start(camera.id):
                    started += 1
            except Exception as exc:  # pragma: no cover - isolated
                camera_log.error(
                    "Failed to start camera", extra={"camera_id": camera.id, "error": str(exc)}
                )
        return started

    def stop_all(self) -> None:
        with self._lock:
            camera_ids = list(self.workers)
        for camera_id in camera_ids:
            self.stop(camera_id)

    def forget(self, camera_id: str) -> None:
        """Stop and drop a worker (used when a camera is deleted)."""
        with self._lock:
            worker = self.workers.pop(camera_id, None)
        if worker:
            worker.stop()

    # ── status ──────────────────────────────────────────────────────────────
    def set_db_status(self, camera_id: str, status: str) -> None:
        def _write(session) -> None:
            row = session.get(Camera, camera_id)
            if row is not None:
                row.status = status
                session.flush()

        try:
            run_in_session(_write)
        except Exception:  # pragma: no cover
            pass

    def health(self, camera_id: str) -> dict[str, Any]:
        worker = self.workers.get(camera_id)
        if worker is None:
            def _load(session) -> dict[str, Any]:
                camera = session.get(Camera, camera_id)
                if camera is None:
                    raise NotFoundError(f"Camera {camera_id!r} does not exist.")
                now = datetime.now(tz=timezone.utc)
                age = (
                    (now - camera.last_frame_at).total_seconds() if camera.last_frame_at else None
                )
                health = CameraHealth(
                    camera_id=camera.id,
                    name=camera.name,
                    status=camera.status,
                    is_enabled=camera.is_enabled,
                    seconds_since_last_frame=age,
                    capture_fps=camera.capture_fps,
                    frames_processed=camera.total_frames,
                    frames_dropped=camera.dropped_frames,
                    reconnect_count=camera.reconnect_count,
                    width=camera.width,
                    height=camera.height,
                    last_error=camera.last_error,
                )
                health.detail["running"] = False
                return health.as_dict()

            return run_in_session(_load)
        data = worker.health.as_dict()
        data["detail"]["running"] = worker.is_running
        if worker.runner is not None:
            data["detail"]["pipeline"] = worker.runner.snapshot()
        # A worker that has been running for a while and has NEVER produced a
        # frame is not online, whatever a stale database row says. This happens
        # with a webcam that has no device attached: the source retries with
        # backoff, never yields, and never regains control to correct itself.
        # Reporting it as `online` would be a false health claim.
        if (
            data["status"] == CameraStatus.ONLINE.value
            and worker.health.frames_processed == 0
            and worker.health.seconds_since_start > NO_FRAME_GRACE_SECONDS
        ):
            data["status"] = CameraStatus.ERROR.value
            data["last_error"] = (
                data["last_error"]
                or "Source connected but has not delivered a frame "
                f"({NO_FRAME_GRACE_SECONDS:.0f}s). Check the device or URL."
            )
        return data

    def summary(self) -> dict[str, Any]:
        """Counts and rates for the dashboard."""

        def _load(session) -> dict[str, Any]:
            rows = session.query(Camera).all()
            total = len(rows)
            online = sum(1 for c in rows if c.status == CameraStatus.ONLINE.value)
            offline = sum(
                1 for c in rows if c.status in (CameraStatus.OFFLINE.value, CameraStatus.ERROR.value)
            )
            disabled = sum(1 for c in rows if not c.is_enabled)
            return {
                "total": total,
                "online": online,
                "offline": offline,
                "degraded": sum(1 for c in rows if c.status == CameraStatus.DEGRADED.value),
                "disabled": disabled,
                "running_workers": sum(1 for w in self.workers.values() if w.is_running),
                "total_frames": sum(c.total_frames for c in rows),
                "dropped_frames": sum(c.dropped_frames for c in rows),
                "reconnect_count": sum(c.reconnect_count for c in rows),
            }

        summary = run_in_session(_load)
        summary["workers"] = {
            camera_id: worker.health.as_dict() for camera_id, worker in self.workers.items()
        }
        return summary

    def ai_summary(self) -> dict[str, Any]:
        hardware = detect_hardware().as_dict()
        statuses = registry.status_report()
        return {
            "device": hardware["device"],
            "provider": hardware["provider"],
            "cuda_available": hardware["cuda_available"],
            "onnx_providers": list(hardware["onnx_providers"]),
            "torch_version": hardware["torch_version"],
            "opencv_version": hardware["opencv_version"],
            "cpu_count": hardware["cpu_count"],
            "models": [
                {
                    "key": s["key"],
                    "name": s["name"],
                    "task": s["task"],
                    "version": s["version"],
                    "status": s["status"],
                    "weights_present": s["weights_present"],
                    "loaded": s["is_loaded"],
                    "device": s["device"],
                    "validation_note": s["validation_note"],
                    "measured": s["measured"],
                }
                for s in statuses
            ],
            "active_pipelines": [
                worker.runner.snapshot()
                for worker in self.workers.values()
                if worker.runner is not None
            ],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "workers": len(self.workers),
            "running": sum(1 for w in self.workers.values() if w.is_running),
            "total_starts": self.total_starts,
            "total_stops": self.total_stops,
            "recognition_catalog": (
                self._recognizer.catalog_size if self._recognizer is not None else 0
            ),
        }

    # ── event publishing ────────────────────────────────────────────────────
    def publish_alert_event(self, alert: Any, alert_type: str, camera_name: str | None = None) -> None:
        from app.services.event_service import EventType

        self._publish(
            EventType.ALERT_CREATED if alert.status == "active" else EventType.ALERT_UPDATED,
            severity=alert.severity,
            camera_id=alert.camera_id,
            camera_name=camera_name,
            shelf_id=alert.shelf_id,
            sku=alert.sku,
            message=alert.title,
            payload={
                "alert_id": alert.id,
                "alert_type": alert_type or alert.alert_type,
                "status": alert.status,
                "title": alert.title,
                "detail": alert.payload,
                "confidence": alert.confidence,
                "occurrence_count": alert.occurrence_count,
            },
            evidence={"path": alert.evidence_path} if alert.evidence_path else {},
            alert_id=alert.id,
        )

    def _publish(self, event_type: str, **kwargs: Any) -> None:
        """Fire-and-forget publish from a worker thread."""
        import asyncio

        from app.services.event_service import event_service

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():  # pragma: no cover - API thread
            asyncio.create_task(event_service.publish(event_type, **kwargs))
            return
        try:
            asyncio.run(event_service.publish(event_type, **kwargs))
        except Exception as exc:  # pragma: no cover - event bus must never break capture
            camera_log.debug("Deferred event publish failed", extra={"error": str(exc)})

    # ── connection testing ──────────────────────────────────────────────────
    def test_connection(self, camera_id: str | None = None, spec: dict[str, Any] | None = None) -> dict[str, Any]:
        """Probe a camera (existing or an unsaved spec) without touching workers."""
        started = time.perf_counter()
        if spec is not None:
            source = build_source(spec)
        else:
            if not camera_id:
                raise ValidationError("Either camera_id or a source spec is required.")
            session = SessionLocal()
            try:
                camera = session.get(Camera, camera_id)
                if camera is None:
                    raise NotFoundError(f"Camera {camera_id!r} does not exist.")
                source = build_source_for_camera(camera)
            finally:
                session.close()
        result = source.probe(timeout=settings.rtsp_connect_timeout)
        result["latency_ms"] = result.get("latency_ms") or round(
            (time.perf_counter() - started) * 1000, 2
        )
        result["camera_id"] = camera_id
        return result


camera_manager = CameraManager()


# ── MJPEG streaming ──────────────────────────────────────────────────────────
def encode_jpeg(frame: Frame, quality: int | None = None) -> bytes:
    quality = quality or settings.mjpeg_quality
    ok, buffer = cv2.imencode(".jpg", frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover - encoder failure is essentially impossible
        raise SourceUnavailableError("Could not encode the frame for streaming.")
    return buffer.tobytes()


def build_multipart_frame(jpeg: bytes) -> bytes:
    delimiter = f"--{MJPEG_BOUNDARY}".encode("ascii")
    return (
        delimiter
        + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
    )