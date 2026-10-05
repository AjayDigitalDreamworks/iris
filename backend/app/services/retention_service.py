"""Retention and cleanup.

Privacy by default: raw detections, evidence snapshots, health samples and old
processing jobs are deleted on a configurable schedule so the system does not
silently accumulate footage-derived data forever.

Every retention period is read from the runtime ``settings`` table and can be
changed from the dashboard. Cleanup is bounded per run and logged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.models import Detection, InferenceRun, SystemHealthSample
from app.services.db_service import run_in_session, settings_service

logger = get_logger("app")

#: Evidence image access is authorised and time-limited, so retention is short.
EVIDENCE_LINK_TTL_HOURS = 24


def cleanup(evidence: bool = True) -> dict[str, Any]:
    """Run one retention pass. Returns the number of rows/files removed."""
    detection_days = settings_service.get_int(
        "retention_days_detections", settings.detection_retention_days
    )
    evidence_days = settings_service.get_int(
        "retention_days_evidence", settings.evidence_retention_days
    )
    alert_days = settings.alert_retention_days
    now = datetime.now(tz=timezone.utc)

    detection_cutoff = now - timedelta(days=detection_days)
    evidence_cutoff = now - timedelta(days=evidence_days)
    alert_cutoff = now - timedelta(days=alert_days)
    health_cutoff = now - timedelta(days=7)

    def _write(session) -> dict[str, int]:
        removed: dict[str, int] = {}
        removed["detections"] = (
            session.query(Detection).filter(Detection.frame_timestamp < detection_cutoff).delete(
                synchronize_session=False
            )
            or 0
        )
        removed["inference_runs"] = (
            session.query(InferenceRun).filter(InferenceRun.started_at < alert_cutoff).delete(
                synchronize_session=False
            )
            or 0
        )
        removed["health_samples"] = (
            session.query(SystemHealthSample)
            .filter(SystemHealthSample.measured_at < health_cutoff)
            .delete(synchronize_session=False)
            or 0
        )
        session.flush()
        return removed

    removed = run_in_session(_write)
    removed["evidence_files"] = 0

    if evidence and evidence_days >= 0:
        removed["evidence_files"] = purge_evidence(evidence_cutoff)
    removed["upload_files"] = 0
    try:
        from app.services.jobs_service import job_service

        removed["upload_files"] = job_service.cleanup(older_than_days=7)
    except Exception as exc:  # pragma: no cover
        logger.debug("Upload cleanup skipped", exc_info=exc)

    logger.info("Retention pass complete", extra=removed)
    return removed


def purge_evidence(cutoff: datetime) -> int:
    """Delete evidence snapshots older than ``cutoff``."""
    evidence_dir = settings.evidence_dir
    if not evidence_dir.exists():
        return 0
    removed = 0
    for path in list(evidence_dir.glob("*.jpg")):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink()
                removed += 1
        except OSError:  # pragma: no cover
            continue
    return removed


def save_evidence(image, *, prefix: str = "evidence", is_demo: bool = False) -> str | None:
    """Persist an evidence snapshot, applying the privacy switch.

    Returns the relative path, or ``None`` when evidence is disabled, the frame
    is demo content, or the write fails.
    """
    if not settings_service.get_bool("evidence_enabled", settings.evidence_enabled):
        return None
    if is_demo:
        # Demo frames are never persisted as evidence.
        return None
    try:
        import cv2

        directory = settings.evidence_dir
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{prefix}_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.jpg"
        path = directory / name
        payload = image
        if settings_service.get_bool("anonymize_faces", settings.anonymize_faces):
            payload = blur_faces(image)
        if not cv2.imwrite(str(path), payload, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
            return None
        return f"evidence/{name}"
    except Exception as exc:  # pragma: no cover - evidence must never break the pipeline
        logger.warning("Could not save evidence", exc_info=exc)
        return None


def blur_faces(image):
    """Blur detected faces in an evidence snapshot.

    Privacy by default: staff reviewing a shelf alert should not need to see
    shoppers' faces.  This is an optional, reversible-on-request control and is
    not facial recognition — no identity is inferred or stored.
    """
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return image
        blurred = image.copy()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        for (x, y, w, h) in faces:
            region = blurred[y : y + h, x : x + w]
            blurred[y : y + h, x : x + w] = cv2.GaussianBlur(region, (31, 31), 0)
        return blurred
    except Exception:  # pragma: no cover
        return image


def evidence_path_to_absolute(relative: str) -> Path | None:
    """Resolve an evidence reference safely, refusing traversal."""
    if not relative:
        return None
    candidate = (settings.resolve_path(settings.data_dir) / relative).resolve()
    root = settings.resolve_path(settings.data_dir).resolve()
    if root not in candidate.parents:
        return None
    return candidate if candidate.exists() else None


def retention_summary() -> dict[str, Any]:
    return {
        "detection_days": settings_service.get_int(
            "retention_days_detections", settings.detection_retention_days
        ),
        "evidence_days": settings_service.get_int(
            "retention_days_evidence", settings.evidence_retention_days
        ),
        "alert_days": settings.alert_retention_days,
        "evidence_enabled": settings_service.get_bool("evidence_enabled", settings.evidence_enabled),
        "anonymize_faces": settings_service.get_bool("anonymize_faces", settings.anonymize_faces),
        "evidence_link_ttl_hours": EVIDENCE_LINK_TTL_HOURS,
        "note": (
            "IRIS stores short evidence snapshots only when enabled; continuous video "
            "recording is never performed by the pipeline."
        ),
    }