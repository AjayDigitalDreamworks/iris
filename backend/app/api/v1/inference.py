"""Video/image upload and offline processing jobs (README §24)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.deps import CurrentUser, DbSession, ManagerUser, PageParams, write_audit
from app.core.config import settings
from app.core.errors import UploadValidationError
from app.core.logging import get_logger
from app.schemas import ActionResponse, JobCreateResponse, JobRead, MessageResponse, Page
from app.services.db_service import paginate
from app.services.jobs_service import job_service

router = APIRouter(prefix="/inference", tags=["inference"])
logger = get_logger("app")

CHUNK = 1024 * 1024


@router.post(
    "/jobs", response_model=JobCreateResponse, status_code=202,
    summary="Upload a video or image for offline analysis",
)
async def create_job(
    request: Request,
    principal: ManagerUser,
    file: UploadFile = File(...),
    camera_id: str | None = Form(default=None),
) -> JobCreateResponse:
    """Accept an upload, validate it, and start a background analysis job.

    The browser is never blocked: processing happens on a worker thread and
    progress is polled through ``GET /inference/jobs/{id}``.
    """
    filename = file.filename or "upload.bin"
    content = bytearray()
    total = 0
    while chunk := await file.read(CHUNK):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise UploadValidationError(
                f"Upload exceeds the {settings.max_upload_mb} MB limit."
            )
        content.extend(chunk)
    await file.close()

    job = await asyncio.to_thread(
        job_service.create,
        filename=filename,
        content=bytes(content),
        uploaded_by=principal.username,
        camera_id=camera_id,
        content_type=file.content_type,
    )
    return JobCreateResponse(
        job=JobRead.model_validate(job),
        message=(
            f"Upload accepted ({total} bytes). Processing runs in the background — "
            "poll this job for progress."
        ),
    )


@router.get("/jobs", response_model=Page[JobRead], summary="List processing jobs")
def list_jobs(
    session: DbSession, page: PageParams, _: CurrentUser, status: str | None = None
) -> Page[JobRead]:
    rows, total = job_service.list(status=status, limit=page.limit, offset=page.offset)
    _ = session
    return Page.build([JobRead.model_validate(r) for r in rows], total, page.limit, page.offset)


@router.get("/jobs/{job_id}", response_model=JobRead, summary="Job status and results")
def get_job(job_id: str, _: CurrentUser) -> JobRead:
    return JobRead.model_validate(job_service.require(job_id))


@router.get(
    "/jobs/{job_id}/summary", summary="Analytics produced by a completed job"
)
def job_summary(job_id: str, _: CurrentUser) -> dict:
    job = job_service.require(job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress_pct": job.progress_pct,
        "frames_processed": job.frames_processed,
        "detection_count": job.detection_count,
        "events_created": job.events_created,
        "alerts_created": job.alerts_created,
        "elapsed_seconds": job.elapsed_seconds,
        "result_summary": job.result_summary,
        "error": job.error,
    }


@router.post("/jobs/{job_id}/cancel", response_model=ActionResponse, summary="Cancel a job")
def cancel_job(job_id: str, request: Request, principal: ManagerUser, session: DbSession) -> ActionResponse:
    job_service.require(job_id)
    cancelled = job_service.cancel(job_id)
    write_audit(session, principal, "job.cancel_requested", entity_type="processing_job",
                entity_id=job_id, detail={"accepted": cancelled}, request=request)
    return ActionResponse(
        action="cancel",
        message="Cancellation requested." if cancelled else "The job already finished.",
        detail={"accepted": cancelled},
    )


@router.delete("/jobs/{job_id}", response_model=MessageResponse, summary="Delete a job (admin)")
def delete_job(
    job_id: str, request: Request, principal: ManagerUser, session: DbSession
) -> MessageResponse:
    from app.models import ProcessingJob

    from app.core.errors import NotFoundError

    job = session.get(ProcessingJob, job_id)
    if job is None:
        raise NotFoundError(f"Processing job {job_id!r} does not exist.")
    job_service.cancel(job_id)
    session.delete(job)
    write_audit(session, principal, "job.deleted", entity_type="processing_job",
                entity_id=job_id, request=request)
    return MessageResponse(message=f"Job {job_id} deleted.")


@router.get("/capabilities", summary="Detection capability summary for the frontend")
def capabilities(_: CurrentUser) -> dict:
    from app.services.camera_manager import camera_manager

    summary = camera_manager.ai_summary()
    return {
        "device": summary["device"],
        "models": [
            {
                "key": m["key"],
                "name": m["name"],
                "task": m["task"],
                "status": m["status"],
                "weights_present": m["weights_present"],
                "validation_note": m["validation_note"],
            }
            for m in summary["models"]
        ],
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "allowed_extensions": list(settings.video_extensions),
        },
        "note": (
            "Uploaded media is analysed by the same pipeline as live streams. "
            "Results are camera estimates of visible stock, not authoritative inventory."
        ),
    }


@router.get(
    "/telemetry/{camera_id}",
    summary="Live capture/inference telemetry for one camera",
)
def stream_telemetry(camera_id: str, session: DbSession, _: CurrentUser) -> dict:
    """Live telemetry for the Live dashboard.

    Reads the camera worker's in-memory health (never the possibly-stale row)
    so the numbers reflect what the stream is doing right now.
    """
    from app.core.errors import NotFoundError
    from app.models import Camera
    from app.services.camera_manager import camera_manager
    from app.vision.models_registry import detect_hardware

    camera = session.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera {camera_id!r} does not exist.")
    health = camera_manager.health(camera_id)
    hardware = detect_hardware()
    return {
        "camera_id": camera_id,
        "name": health.get("name") or camera.name,
        "status": health.get("status"),
        "capture_fps": health.get("capture_fps") or 0.0,
        "inference_fps": health.get("inference_fps") or 0.0,
        "inference_latency_ms": health.get("inference_latency_ms"),
        "frames_processed": health.get("frames_processed") or 0,
        "frames_dropped": health.get("frames_dropped") or 0,
        "drop_ratio": health.get("drop_ratio") or 0.0,
        "hardware": {
            "device": hardware.device,
            "provider": hardware.provider,
            "cuda_available": hardware.cuda_available,
            "platform": hardware.platform,
        },
    }


_ = paginate  # imported for symmetry with other list endpoints