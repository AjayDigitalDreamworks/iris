"""System health and observability.

Reports each subsystem as HEALTHY / DEGRADED / OFFLINE together with the
metrics an operator actually needs: FPS, latency, CPU, memory, GPU, queue
depth, processing errors and camera reconnects.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.models import Camera, CameraStatus, ModelVersion, SystemHealthSample
from app.services.db_service import run_in_session, settings_service
from app.services.event_service import event_service

logger = get_logger("app")

_STARTED_AT = time.time()


def uptime_seconds() -> float:
    return round(time.time() - _STARTED_AT, 1)


def resource_metrics() -> dict[str, Any]:
    """CPU / memory / disk / GPU metrics. Uses psutil when available."""
    metrics: dict[str, Any] = {}
    try:
        import psutil

        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        virtual = psutil.virtual_memory()
        metrics["memory_percent"] = virtual.percent
        metrics["memory_used_mb"] = round(virtual.used / (1024**2), 1)
        metrics["memory_total_mb"] = round(virtual.total / (1024**2), 1)
        disk = psutil.disk_usage(str(settings.resolve_path(settings.data_dir).anchor or "C:/"))
        metrics["disk_percent"] = disk.percent
        metrics["disk_free_gb"] = round(disk.free / (1024**3), 2)
        try:
            temps = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
            cpu_temp = temps.get("cpu") or temps.get("coretemp") or []
            if cpu_temp:
                metrics["cpu_temp_c"] = round(cpu_temp[0].current, 1)
        except Exception:  # pragma: no cover - not available on all platforms
            pass
    except Exception as exc:  # pragma: no cover
        logger.debug("psutil unavailable", exc_info=exc)

    # GPU utilisation is only meaningful when a GPU exists.
    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - no GPU on this host
            metrics["gpu_percent"] = round(torch.cuda.utilization(), 1)
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
    except Exception:  # pragma: no cover
        pass
    return metrics


def gpu_info() -> dict[str, Any]:
    from app.vision.models_registry import detect_hardware

    hardware = detect_hardware()
    return {
        "available": hardware.cuda_available,
        "name": hardware.cuda_device_name,
        "device": hardware.device,
        "onnx_providers": list(hardware.onnx_providers),
    }


def storage_status() -> dict[str, Any]:
    data_dir = settings.resolve_path(settings.data_dir)
    try:
        usage = shutil.disk_usage(str(data_dir))
        evidence_dir = settings.evidence_dir
        evidence_files = sum(1 for _ in evidence_dir.glob("*.jpg")) if evidence_dir.exists() else 0
        uploads = settings.upload_dir
        upload_files = sum(1 for _ in uploads.glob("*")) if uploads.exists() else 0
        return {
            "status": "healthy",
            "data_dir": str(data_dir),
            "writable": os.access(data_dir, os.W_OK),
            "free_gb": round(usage.free / (1024**3), 2),
            "used_pct": round(100.0 * usage.used / usage.total, 2) if usage.total else 0.0,
            "evidence_files": evidence_files,
            "upload_files": upload_files,
        }
    except Exception as exc:  # pragma: no cover
        return {"status": "offline", "error": type(exc).__name__}


def database_status() -> dict[str, Any]:
    from app.db.session import check_database

    return check_database()


def camera_status() -> dict[str, Any]:
    from app.services.camera_manager import camera_manager

    def _load(session) -> dict[str, Any]:
        rows = session.query(Camera).all()
        online = sum(1 for c in rows if c.status == CameraStatus.ONLINE.value)
        errors = sum(1 for c in rows if c.status == CameraStatus.ERROR.value)
        stale_after = settings_service.get_float("camera_offline_alert_seconds", 60.0)
        now = datetime.now(tz=timezone.utc)
        stale = sum(
            1
            for c in rows
            if c.status == CameraStatus.ONLINE.value
            and c.last_frame_at is not None
            and (now - c.last_frame_at).total_seconds() > stale_after
        )
        return {
            "total": len(rows),
            "online": online,
            "error": errors,
            "stale": stale,
            "reconnects": sum(c.reconnect_count for c in rows),
            "dropped_frames": sum(c.dropped_frames for c in rows),
            "workers": camera_manager.stats(),
        }

    data = run_in_session(_load)
    if data["total"] == 0:
        return {"status": "degraded", **data, "detail": "No cameras are configured."}
    if data["error"] or data["stale"]:
        return {"status": "degraded", **data}
    return {"status": "healthy", **data}


def model_status() -> dict[str, Any]:
    from app.vision.models_registry import registry

    def _load(session) -> dict[str, Any]:
        rows = session.query(ModelVersion).all()
        return {
            "registered_in_db": len(rows),
            "missing_weights": [r.name for r in rows if r.weights_path and not r.weights_path],
        }

    statuses = registry.status_report()
    missing = [s["key"] for s in statuses if s["weights_present"] is False]
    degraded = bool(missing)
    return {
        "status": "degraded" if degraded else "healthy",
        "models": statuses,
        "missing_weights": missing,
        **run_in_session(_load),
    }


def worker_status() -> dict[str, Any]:
    from app.services.camera_manager import camera_manager
    from app.services.jobs_service import job_service

    stats = camera_manager.stats()
    pipelines = [
        worker.runner.snapshot()
        for worker in camera_manager.workers.values()
        if worker.runner is not None
    ]
    latencies = [p["mean_latency_ms"] for p in pipelines if p["mean_latency_ms"]]
    return {
        "status": "healthy" if stats["workers"] else "degraded",
        "camera_workers": stats,
        "processing_jobs": job_service.stats(),
        "active_pipelines": pipelines,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "queue_depth": sum(w.health.queue_depth for w in camera_manager.workers.values()),
        "processing_errors": sum(
            len(w.runner.errors) for w in camera_manager.workers.values() if w.runner
        ),
    }


def websocket_status() -> dict[str, Any]:
    stats = event_service.stats()
    return {
        "status": "healthy",
        "clients": stats["websocket_clients"],
        "total_connections": stats["total_connections"],
        "dropped_connections": stats["dropped_connections"],
        "events_published": stats["published"],
        "events_coalesced": stats["coalesced"],
        "pending_coalesced": stats["pending_coalesced"],
    }


def collect() -> dict[str, Any]:
    """Full health snapshot used by /api/v1/system/health and the System page."""
    from app.db.session import engine  # noqa: F401  (import check)

    components: dict[str, Any] = {
        "backend": {
            "status": "healthy",
            "detail": f"{settings.app_name} {settings.app_version} ({settings.env})",
        },
        "database": database_status(),
        "camera": camera_status(),
        "ai_model": model_status(),
        "worker": worker_status(),
        "websocket": websocket_status(),
        "storage": storage_status(),
    }
    statuses = [component.get("status", "unknown") for component in components.values()]
    if "offline" in statuses:
        overall = "offline"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    from app.services.camera_manager import camera_manager

    metrics = {
        **resource_metrics(),
        "uptime_seconds": uptime_seconds(),
        "gpu": gpu_info(),
        "active_workers": camera_manager.stats()["running"],
    }
    return {
        "status": overall,
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.env,
        "timestamp": datetime.now(tz=timezone.utc),
        "components": components,
        "metrics": metrics,
    }


def record_sample() -> SystemHealthSample | None:
    """Persist one health sample (called by the background monitor)."""
    snapshot = collect()
    metrics = snapshot["metrics"]

    def _write(session) -> SystemHealthSample:
        sample = SystemHealthSample(
            cpu_percent=metrics.get("cpu_percent"),
            memory_percent=metrics.get("memory_percent"),
            memory_used_mb=metrics.get("memory_used_mb"),
            gpu_percent=metrics.get("gpu_percent"),
            gpu_name=metrics.get("gpu_name"),
            disk_percent=metrics.get("disk_percent"),
            disk_free_gb=metrics.get("disk_free_gb"),
            queue_depth=snapshot["components"]["worker"].get("queue_depth", 0),
            active_workers=snapshot["components"]["worker"]["camera_workers"]["running"],
            processing_errors=snapshot["components"]["worker"].get("processing_errors", 0),
            camera_reconnects=snapshot["components"]["camera"].get("reconnects", 0),
            websocket_clients=snapshot["components"]["websocket"]["clients"],
            status=snapshot["status"],
            detail={"storage": snapshot["components"]["storage"]},
        )
        session.add(sample)
        session.flush()
        return sample

    try:
        return run_in_session(_write)
    except Exception as exc:  # pragma: no cover
        logger.debug("Health sample not recorded", exc_info=exc)
        return None


def history(limit: int = 100) -> list[SystemHealthSample]:
    def _load(session) -> list[SystemHealthSample]:
        return list(
            session.query(SystemHealthSample)
            .order_by(SystemHealthSample.measured_at.desc())
            .limit(limit)
            .all()
        )

    return run_in_session(_load)