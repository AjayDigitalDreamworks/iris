"""Video-upload processing jobs.

Flow (README §24)::

    Upload video -> validate -> create job -> background thread
      -> decode + sample -> detect -> track -> recognise
      -> persist detections / estimates / events -> report progress

Processing runs on a background thread so the browser never freezes and the API
stays responsive.  Uploads are validated (type, size, filename, extension,
path) before anything touches disk.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import NotFoundError, UploadValidationError
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models import Camera, Detection, DetectionSource, ProcessingJob, Store
from app.services.alert_service import alert_service
from app.services.db_service import paginate, run_in_session
from app.services.pipeline import CameraConfig, PipelineRunner
from app.vision.ingestion import VideoFileSource

logger = get_logger("app")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAGIC = {
    b"\x00\x00\x00\x18ftypmp4": "video/mp4",
    b"\x00\x00\x00\x1cftyp": "video/mp4",
    b"RIFF": "video/avi",
    b"\x1aE\xdf\xa3": "video/webm",
}


def validate_upload(filename: str, content: bytes, declared_type: str | None = None) -> str:
    """Validate an uploaded file. Raises :class:`UploadValidationError` on reject."""
    if not filename:
        raise UploadValidationError("A filename is required.")
    suffix = Path(filename).suffix.lower()
    allowed = settings.video_extensions
    if suffix not in allowed:
        raise UploadValidationError(
            f"Unsupported file type {suffix or '(none)'}. Allowed: {', '.join(allowed)}."
        )
    if not content:
        raise UploadValidationError("The uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise UploadValidationError(
            f"File is larger than the {settings.max_upload_mb} MB limit."
        )
    if _SAFE_NAME.search(Path(filename).stem):
        raise UploadValidationError("The filename contains unsupported characters.")
    return suffix


def sniff_content_type(content: bytes) -> str | None:
    for magic, mime in _MAGIC.items():
        if content.startswith(magic):
            return mime
    return None


def safe_filename(original: str) -> str:
    stem = _SAFE_NAME.sub("_", Path(original).stem)[:60] or "upload"
    return f"{stem}{Path(original).suffix.lower()}"


class JobService:
    """Creates and runs offline processing jobs."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}

    # ── creation ────────────────────────────────────────────────────────────
    def create(
        self,
        *,
        filename: str,
        content: bytes,
        uploaded_by: str | None = None,
        camera_id: str | None = None,
        content_type: str | None = None,
    ) -> ProcessingJob:
        validate_upload(filename, content, content_type)
        upload_dir = settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex[:12]}_{safe_filename(filename)}"
        stored_path = upload_dir / stored_name

        # Path traversal guard: the resolved destination must stay inside upload_dir.
        resolved = stored_path.resolve()
        if upload_dir.resolve() not in resolved.parents:
            logger.error(
                "Rejected path traversal upload attempt",
                extra={"source_filename": filename, "uploaded_by": uploaded_by},
            )
            raise UploadValidationError("The upload destination is invalid.")

        resolved.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()

        def _write(session) -> ProcessingJob:
            job = ProcessingJob(
                camera_id=camera_id,
                uploaded_by=uploaded_by,
                original_filename=filename,
                stored_path=str(resolved),
                content_type=content_type or sniff_content_type(content),
                size_bytes=len(content),
                sha256=digest,
                status="queued",
                stage="queued",
            )
            session.add(job)
            session.flush()
            return job

        job = run_in_session(_write)
        logger.info(
            "Processing job created",
            extra={"job_id": job.id, "source_filename": filename, "bytes": len(content)},
        )
        self.submit(job.id)
        return job

    def submit(self, job_id: str) -> None:
        thread = threading.Thread(
            target=self._run_wrapper, args=(job_id,), name=f"iris-job-{job_id[:8]}", daemon=True
        )
        self._threads[job_id] = thread
        thread.start()

    def cancel(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    # ── execution ───────────────────────────────────────────────────────────
    def _run_wrapper(self, job_id: str) -> None:
        acquired = self._semaphore.acquire(timeout=300)
        try:
            self.run(job_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Processing job crashed", extra={"job_id": job_id})
            self._update(job_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if acquired:
                self._semaphore.release()

    def run(self, job_id: str) -> ProcessingJob | None:
        """Process a stored upload end to end, updating progress as it goes."""
        from app.services.camera_manager import camera_manager

        cancel = self._cancel.setdefault(job_id, threading.Event())
        job = self.get(job_id)
        if job is None:
            return None

        path = Path(job.stored_path)
        if not path.exists():
            self._update(job_id, status="failed", stage="failed", error="Stored file is missing.")
            return None

        started = time.time()
        self._update(
            job_id, status="running", stage="decoding", started_at=datetime.now(tz=timezone.utc)
        )

        camera_id = job.camera_id
        config = self._build_config(job_id, camera_id)

        source = VideoFileSource(
            str(path), target_fps=settings.default_capture_fps, frame_skip=max(1, settings.frame_skip)
        )
        runner = PipelineRunner(
            config, recognizer=camera_manager.recognizer, alert_service=alert_service
        )

        total_frames = 0
        processed = 0
        detection_total = 0
        events_created = 0
        alerts_created = 0
        cancelled = False

        try:
            for frame in source.frames():
                if cancel.is_set():
                    cancelled = True
                    break
                result = runner.process(frame)
                processed += 1
                detection_total += len(result.detections)
                events_created += self._persist_frame(job, result, source_label=DetectionSource.UPLOAD.value)
                alerts_created += self._persist_alerts(job, result)
                if processed % 5 == 0:
                    progress = min(99.0, 100.0 * frame.index / max(1, total_frames or frame.index))
                    self._update(
                        job_id,
                        stage="analyzing",
                        progress_pct=round(progress, 2),
                        frames_processed=processed,
                        detection_count=detection_total,
                        events_created=events_created,
                        alerts_created=alerts_created,
                        frame_width=result.width,
                        frame_height=result.height,
                    )
        except Exception as exc:
            logger.exception("Job processing failed", extra={"job_id": job_id})
            self._update(
                job_id, status="failed", stage="failed",
                error=f"{type(exc).__name__}: {exc}",
                frames_processed=processed, detection_count=detection_total,
            )
            return self.get(job_id)
        finally:
            source.close()

        elapsed = time.time() - started
        summary = self._build_summary(job, processed, detection_total, runner)
        self._update(
            job_id,
            status="cancelled" if cancelled else "completed",
            stage="completed" if not cancelled else "cancelled",
            progress_pct=100.0 if not cancelled else 50.0,
            frames_processed=processed,
            detection_count=detection_total,
            events_created=events_created,
            alerts_created=alerts_created,
            elapsed_seconds=round(elapsed, 3),
            finished_at=datetime.now(tz=timezone.utc),
            result_summary=summary,
        )
        logger.info(
            "Processing job finished",
            extra={
                "job_id": job_id, "frames": processed,
                "detections": detection_total, "elapsed_s": round(elapsed, 2),
            },
        )
        return self.get(job_id)

    def _build_config(self, job_id: str, camera_id: str | None) -> CameraConfig:
        """Build a pipeline config for an offline job (no live camera needed)."""
        from app.services.pipeline import (
            line_config_from_zone,
            load_shelf_configs,
            queue_config_from_zone,
            zone_config_from_zone,
        )

        shelves = []
        store_id = None
        if camera_id:
            shelves, store_id = load_shelf_configs(camera_id)
            session = SessionLocal()
            try:
                camera = session.get(Camera, camera_id)
                zones = [z for z in camera.zones if z.is_active] if camera else []
            finally:
                session.close()
            lines = [line_config_from_zone(z) for z in zones if z.zone_type.endswith("_line")]
            queue_zones = [queue_config_from_zone(z) for z in zones if z.zone_type == "queue"]
            occupancy_zones = [zone_config_from_zone(z) for z in zones if z.zone_type == "zone"]
            name = camera.name if camera else f"job-{job_id[:8]}"
        else:
            lines, queue_zones, occupancy_zones = [], [], []
            name = f"job-{job_id[:8]}"
            session = SessionLocal()
            try:
                store = session.query(Store).first()
                store_id = store.id if store else None
            finally:
                session.close()

        return CameraConfig(
            camera_id=camera_id or f"job:{job_id}",
            camera_name=name,
            store_id=store_id,
            role="shelf",
            source_type="video_file",
            is_demo=False,
            shelves=shelves,
            lines=lines,
            queue_zones=queue_zones,
            occupancy_zones=occupancy_zones,
        )

    def _persist_frame(self, job: ProcessingJob, result: Any, *, source_label: str) -> int:
        from app.services.inventory_service import inventory_service

        events = 0

        def _write(session) -> None:
            for detection in result.detections:
                session.add(
                    Detection(
                        camera_id=job.camera_id,
                        job_id=job.id,
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
                        frame_timestamp=result.captured_at,
                        source=source_label,
                        model_id=result.model_id,
                        frame_index=result.frame_index,
                    )
                )
                events += 1
            for crossing in (result.customer.crossings if result.customer else []):
                events += 1
                from app.models import FootfallEvent

                session.add(
                    FootfallEvent(
                        camera_id=job.camera_id,
                        job_id=job.id,
                        track_uid=crossing.track_uid,
                        external_id=crossing.external_id,
                        event_type=crossing.event_type,
                        direction=crossing.direction,
                        crossing_x=crossing.x,
                        crossing_y=crossing.y,
                        confidence=crossing.confidence,
                        occurred_at=result.captured_at,
                    )
                )

        run_in_session(_write)

        if result.stock_estimates:
            inventory_service.record_estimates(
                result.stock_estimates, job_id=job.id, source="video_upload_estimate"
            )
        return events

    def _persist_alerts(self, job: ProcessingJob, result: Any) -> int:
        created = 0
        for outcome in result.rule_outcomes:
            sku = outcome.payload.get("sku")
            alert = alert_service.apply(
                outcome,
                camera_id=job.camera_id,
                shelf_id=outcome.payload.get("shelf_id"),
                zone_id=outcome.payload.get("zone_id"),
                sku=sku,
            )
            if alert is not None:
                created += 1
        return created

    def _build_summary(
        self, job: ProcessingJob, frames: int, detections: int, runner: PipelineRunner
    ) -> dict[str, Any]:
        from app.models import InventorySnapshot

        def _load(session) -> dict[str, Any]:
            snapshots = (
                session.query(InventorySnapshot)
                .filter(InventorySnapshot.job_id == job.id)
                .order_by(InventorySnapshot.observed_at.desc())
                .limit(500)
                .all()
            )
            latest_by_sku: dict[str, Any] = {}
            for snapshot in snapshots:
                latest_by_sku.setdefault(snapshot.sku, snapshot)
            return {
                "skus_observed": sorted(latest_by_sku),
                "final_estimates": [
                    {
                        "sku": snapshot.sku,
                        "estimated_quantity": round(snapshot.estimated_quantity, 3),
                        "stock_percentage": round(snapshot.stock_percentage, 2),
                        "confidence": round(snapshot.confidence, 4),
                    }
                    for snapshot in latest_by_sku.values()
                ],
                "low_stock_skus": [
                    s.sku
                    for s in latest_by_sku.values()
                    if 0 <= s.stock_percentage <= 20
                ],
                "note": (
                    "Estimates come from an offline replay of the uploaded file and are "
                    "camera estimates of VISIBLE stock only."
                ),
            }

        summary = run_in_session(_load)
        summary.update(
            {
                "frames_processed": frames,
                "detections": detections,
                "mean_latency_ms": runner.mean_latency_ms,
                "model_id": result_model(runner),
                "inference_fps": runner.inference_fps,
            }
        )
        return summary

    # ── accessors ───────────────────────────────────────────────────────────
    def _update(self, job_id: str, **fields: Any) -> None:
        def _write(session) -> None:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            for key, value in fields.items():
                if value is not None and hasattr(job, key):
                    setattr(job, key, value)
            session.flush()

        run_in_session(_write)

    def get(self, job_id: str) -> ProcessingJob | None:
        return run_in_session(lambda session: session.get(ProcessingJob, job_id))

    def require(self, job_id: str) -> ProcessingJob:
        job = self.get(job_id)
        if job is None:
            raise NotFoundError(f"Processing job {job_id!r} does not exist.")
        return job

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ProcessingJob], int]:
        def _load(session) -> tuple[list[ProcessingJob], int]:
            query = session.query(ProcessingJob)
            if status:
                query = query.filter(ProcessingJob.status == status)
            return paginate(query.order_by(ProcessingJob.created_at.desc()), limit, offset)

        return run_in_session(_load)

    def cleanup(self, older_than_days: int = 7) -> int:
        """Delete old job rows and their stored uploads (retention)."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=older_than_days)

        def _write(session) -> list[str]:
            rows = (
                session.query(ProcessingJob)
                .filter(ProcessingJob.created_at < cutoff)
                .all()
            )
            paths = [row.stored_path for row in rows]
            for row in rows:
                session.delete(row)
            session.flush()
            return paths

        paths = run_in_session(_write)
        removed = 0
        for path in paths:
            candidate = Path(path)
            if candidate.exists() and settings.upload_dir.resolve() in candidate.resolve().parents:
                try:
                    candidate.unlink()
                    removed += 1
                except OSError:  # pragma: no cover
                    pass
        return removed

    def stats(self) -> dict[str, Any]:
        def _load(session) -> dict[str, Any]:
            rows = session.query(ProcessingJob).all()
            counts: dict[str, int] = {}
            for row in rows:
                counts[row.status] = counts.get(row.status, 0) + 1
            return {"total": len(rows), "by_status": counts,
                    "running_threads": sum(1 for t in self._threads.values() if t.is_alive())}

        return run_in_session(_load)


def result_model(runner: PipelineRunner) -> str | None:
    from app.services.pipeline import result_model_id

    return result_model_id(runner.detector)


job_service = JobService()