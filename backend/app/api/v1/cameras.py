"""Camera management, zones and connection testing.

Security note: a camera's RTSP URL is encrypted at rest and is **never**
returned.  The API only exposes ``stream_url_preview`` (``rtsp://user:***@host/path``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import (
    AdminUser,
    CurrentUser,
    DbSession,
    ManagerUser,
    PageParams,
    flush_or_conflict,
    commit_before_camera_work,
    rate_limit,
    write_audit,
)
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.security import encrypt_secret, redact_url
from app.models import Camera, CameraHealthSample, CameraStatus, CameraZone, Store
from app.schemas import (
    ActionResponse,
    CameraCreate,
    CameraHealthRead,
    CameraRead,
    CameraUpdate,
    CameraZoneCreate,
    CameraZoneRead,
    CameraZoneUpdate,
    ConnectionTestResult,
    MessageResponse,
    Page,
)
from app.schemas.common import ComponentHealth
from app.services.camera_manager import camera_manager
from app.services.db_service import paginate

router = APIRouter(prefix="/cameras", tags=["cameras"])
logger = get_logger("app")


def _to_read(camera: Camera) -> CameraRead:
    data = CameraRead.model_validate(camera)
    object.__setattr__(data, "has_credentials", bool(camera.stream_url_encrypted))
    return data


def _get_camera(session, camera_id: str) -> Camera:
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera {camera_id!r} does not exist.")
    return camera


# ── CRUD ─────────────────────────────────────────────────────────────────────
@router.get("", response_model=Page[CameraRead], summary="List cameras")
def list_cameras(
    session: DbSession,
    page: PageParams,
    _: CurrentUser,
    store_id: str | None = None,
    status: str | None = None,
    source_type: str | None = None,
) -> Page[CameraRead]:
    query = session.query(Camera)
    if store_id:
        query = query.filter(Camera.store_id == store_id)
    if status:
        query = query.filter(Camera.status == status)
    if source_type:
        query = query.filter(Camera.source_type == source_type)
    rows, total = paginate(query.order_by(Camera.created_at.desc()), page.limit, page.offset)
    return Page.build([_to_read(c) for c in rows], total, page.limit, page.offset)


@router.post("", response_model=CameraRead, status_code=201, summary="Register a camera (admin)")
def create_camera(
    payload: CameraCreate, request: Request, session: DbSession, principal: AdminUser
) -> CameraRead:
    store = session.get(Store, payload.store_id)
    if store is None:
        raise NotFoundError(f"Store {payload.store_id!r} does not exist.")
    if payload.source_type == "rtsp" and not payload.stream_url:
        raise ValidationError("An RTSP camera requires a stream_url.")
    if payload.source_type == "url" and not payload.stream_url:
        raise ValidationError(
            "A web-video camera requires a stream_url (a YouTube, Vimeo or "
            "direct media link)."
        )
    if payload.source_type == "ip_webcam" and not payload.stream_url:
        raise ValidationError(
            "An IP Webcam camera requires a stream_url (the phone's address, "
            "e.g. 192.168.1.42:8080)."
        )
    if payload.source_type == "video_file" and not payload.file_path:
        raise ValidationError("A video-file camera requires a file_path.")

    data = payload.model_dump(exclude={"stream_url"})
    camera = Camera(**data)
    if payload.stream_url:
        camera.stream_url_encrypted = encrypt_secret(payload.stream_url)
        camera.stream_url_preview = redact_url(payload.stream_url)
    session.add(camera)
    flush_or_conflict(session, "camera")
    # Give the camera the zones its role needs, so footfall/queue/stock analytics
    # have a region to measure from the moment it starts. Idempotent and clearly
    # labelled; an operator's own geometry always wins.
    from app.services.provisioning import provision_camera_defaults

    provision_camera_defaults(session, camera)
    session.flush()
    write_audit(
        session, principal, "camera.created", entity_type="camera", entity_id=camera.id,
        detail={"name": camera.name, "source_type": camera.source_type}, request=request,
    )
    # 'name' is a reserved LogRecord attribute; using it as an extra field
    # raises inside logging and takes the whole request down.
    logger.info(
        "Camera registered",
        extra={"camera_id": camera.id, "camera_name": camera.name},
    )
    return _to_read(camera)


@router.get("/{camera_id}", response_model=CameraRead, summary="Camera details")
def get_camera(camera_id: str, session: DbSession, _: CurrentUser) -> CameraRead:
    return _to_read(_get_camera(session, camera_id))


@router.patch("/{camera_id}", response_model=CameraRead, summary="Update a camera (admin)")
def update_camera(
    camera_id: str, payload: CameraUpdate, request: Request,
    session: DbSession, principal: AdminUser,
) -> CameraRead:
    camera = _get_camera(session, camera_id)
    data = payload.model_dump(exclude_unset=True, exclude={"stream_url"})
    for key, value in data.items():
        if value is not None:
            setattr(camera, key, value)
    if payload.stream_url:
        camera.stream_url_encrypted = encrypt_secret(payload.stream_url)
        camera.stream_url_preview = redact_url(payload.stream_url)
    session.flush()
    write_audit(session, principal, "camera.updated", entity_type="camera", entity_id=camera.id,
                detail={"fields": sorted(data.keys())}, request=request)
    return _to_read(camera)


@router.delete("/{camera_id}", response_model=MessageResponse, summary="Delete a camera (admin)")
def delete_camera(
    camera_id: str, request: Request, session: DbSession, principal: AdminUser
) -> MessageResponse:
    camera = _get_camera(session, camera_id)
    session.delete(camera)
    camera_manager.forget(camera_id)
    write_audit(session, principal, "camera.deleted", entity_type="camera", entity_id=camera_id,
                request=request)
    return MessageResponse(message=f"Camera {camera_id} deleted.")


# ── Operations ───────────────────────────────────────────────────────────────
@router.post(
    "/{camera_id}/start", response_model=ActionResponse, summary="Start the camera pipeline (manager)"
)
def start_camera(camera_id: str, session: DbSession, _: ManagerUser) -> ActionResponse:
    camera = _get_camera(session, camera_id)
    if not camera.is_enabled:
        raise ValidationError("This camera is disabled. Enable it before starting.")
    commit_before_camera_work(session)
    started = camera_manager.start(camera_id)
    return ActionResponse(
        action="start",
        message=f"Camera {camera.name} {'started' if started else 'was already running'}.",
        detail={"camera_id": camera_id, "started": started},
    )


@router.post(
    "/{camera_id}/stop", response_model=ActionResponse, summary="Stop the camera pipeline (manager)"
)
def stop_camera(camera_id: str, session: DbSession, _: ManagerUser) -> ActionResponse:
    camera = _get_camera(session, camera_id)
    commit_before_camera_work(session)
    camera_manager.stop(camera_id)
    return ActionResponse(action="stop", message=f"Camera {camera.name} stopped.",
                          detail={"camera_id": camera_id})


@router.post(
    "/{camera_id}/restart", response_model=ActionResponse, summary="Restart the camera (manager)"
)
def restart_camera(camera_id: str, session: DbSession, _: ManagerUser) -> ActionResponse:
    camera = _get_camera(session, camera_id)
    commit_before_camera_work(session)
    camera_manager.restart(camera_id)
    return ActionResponse(action="restart", message=f"Camera {camera.name} restarted.",
                          detail={"camera_id": camera_id})


@router.post(
    "/test", response_model=ConnectionTestResult, summary="Test a source without saving (admin)"
)
def test_unsaved_source(
    payload: dict[str, Any], _: AdminUser, _limit=Depends(rate_limit("camera_test"))
) -> ConnectionTestResult:
    if not isinstance(payload, dict):
        raise ValidationError("A source specification object is required.")
    try:
        result = camera_manager.test_connection(spec=payload)
    except Exception as exc:
        return ConnectionTestResult(
            ok=False, message="Could not open the source.", error=f"{type(exc).__name__}: {exc}"
        )
    return ConnectionTestResult(**result)


@router.post(
    "/{camera_id}/test", response_model=ConnectionTestResult, summary="Test a saved camera (admin)"
)
def test_camera(
    camera_id: str, session: DbSession, _: AdminUser, _limit=Depends(rate_limit("camera_test"))
) -> ConnectionTestResult:
    camera = _get_camera(session, camera_id)
    result = camera_manager.test_connection(camera_id=camera_id)
    return ConnectionTestResult(**result)


@router.get("/{camera_id}/health", response_model=CameraHealthRead, summary="Live camera health")
def camera_health(camera_id: str, session: DbSession, _: CurrentUser) -> CameraHealthRead:
    _get_camera(session, camera_id)
    return CameraHealthRead(**camera_manager.health(camera_id))


@router.get("/health/summary", response_model=list[ComponentHealth], summary="Health for every camera")
def camera_health_summary(session: DbSession, _: CurrentUser) -> list[ComponentHealth]:
    rows: list[ComponentHealth] = []
    for camera in session.query(Camera).all():
        data = camera_manager.health(camera.id)
        status = "healthy" if data["status"] == CameraStatus.ONLINE.value else "degraded"
        if data["status"] in (CameraStatus.ERROR.value,) or data["is_stale"]:
            status = "offline"
        rows.append(
            ComponentHealth(
                name=f"camera:{camera.name}",
                status=status,
                detail=data.get("last_error") or data["status"],
                latency_ms=data.get("inference_latency_ms"),
                metrics={
                    "camera_id": camera.id,
                    "inference_fps": data["inference_fps"],
                    "capture_fps": data["capture_fps"],
                    "drop_ratio": data["drop_ratio"],
                    "reconnect_count": data["reconnect_count"],
                    "stale": data["is_stale"],
                },
            )
        )
    return rows


@router.get(
    "/{camera_id}/health/history", response_model=list[CameraHealthRead], summary="Health samples"
)
def camera_health_history(
    camera_id: str, session: DbSession, _: CurrentUser, limit: int = 100
) -> list[CameraHealthRead]:
    samples = (
        session.query(CameraHealthSample)
        .filter(CameraHealthSample.camera_id == camera_id)
        .order_by(CameraHealthSample.measured_at.desc())
        .limit(min(limit, 1000))
        .all()
    )
    return [
        CameraHealthRead(
            camera_id=sample.camera_id,
            name=_get_camera(session, camera_id).name,
            status=sample.status,
            is_enabled=True,
            is_stale=sample.status != CameraStatus.ONLINE.value,
            is_frozen=sample.is_frozen,
            capture_fps=sample.capture_fps or 0.0,
            inference_fps=sample.inference_fps or 0.0,
            inference_latency_ms=sample.inference_latency_ms,
            queue_depth=sample.queue_depth,
            frames_processed=sample.frames_processed,
            frames_dropped=sample.frames_dropped,
            reconnect_count=0,
            error=sample.error,
        )
        for sample in samples
    ]


# ── Zones ────────────────────────────────────────────────────────────────────
@router.get("/{camera_id}/zones", response_model=list[CameraZoneRead], summary="List camera zones")
def list_zones(camera_id: str, session: DbSession, _: CurrentUser) -> list[CameraZoneRead]:
    _get_camera(session, camera_id)
    return [
        CameraZoneRead.model_validate(z)
        for z in session.query(CameraZone).filter(CameraZone.camera_id == camera_id).all()
    ]


@router.post(
    "/{camera_id}/zones", response_model=CameraZoneRead, status_code=201,
    summary="Create a zone (admin)",
)
def create_zone(
    camera_id: str, payload: CameraZoneCreate, request: Request,
    session: DbSession, principal: AdminUser,
) -> CameraZoneRead:
    _get_camera(session, camera_id)
    data = payload.model_dump()
    data.pop("camera_id", None)
    zone = CameraZone(camera_id=camera_id, **data)
    session.add(zone)
    session.flush()
    write_audit(session, principal, "camera_zone.created", entity_type="camera_zone",
                entity_id=zone.id, detail={"name": zone.name, "type": zone.zone_type}, request=request)
    # Zone changes require a pipeline reload.
    commit_before_camera_work(session)
    camera_manager.restart(camera_id)
    return CameraZoneRead.model_validate(zone)


@router.patch("/zones/{zone_id}", response_model=CameraZoneRead, summary="Update a zone (admin)")
def update_zone(
    zone_id: str, payload: CameraZoneUpdate, request: Request,
    session: DbSession, principal: AdminUser,
) -> CameraZoneRead:
    zone = session.get(CameraZone, zone_id)
    if zone is None:
        raise NotFoundError(f"Zone {zone_id!r} does not exist.")
    for key, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(zone, key, value)
    session.flush()
    write_audit(session, principal, "camera_zone.updated", entity_type="camera_zone",
                entity_id=zone.id, request=request)
    commit_before_camera_work(session)
    camera_manager.restart(zone.camera_id)
    return CameraZoneRead.model_validate(zone)


@router.delete("/zones/{zone_id}", response_model=MessageResponse, summary="Delete a zone (admin)")
def delete_zone(
    zone_id: str, request: Request, session: DbSession, principal: AdminUser,
) -> MessageResponse:
    zone = session.get(CameraZone, zone_id)
    if zone is None:
        raise NotFoundError(f"Zone {zone_id!r} does not exist.")
    camera_id = zone.camera_id
    session.delete(zone)
    write_audit(session, principal, "camera_zone.deleted", entity_type="camera_zone",
                entity_id=zone_id, request=request)
    commit_before_camera_work(session)
    camera_manager.restart(camera_id)
    return MessageResponse(message=f"Zone {zone_id} deleted.")


@router.post(
    "/{camera_id}/zones/provision",
    summary="Create default zones (and a shelf) for a camera (admin)",
)
def provision_zones(
    camera_id: str, request: Request, session: DbSession, principal: AdminUser,
) -> dict:
    """Give a camera the zones its role needs so analytics produce output.

    A camera with no zones yields NO footfall, queue or stock numbers. This
    creates clearly-labelled default regions derived from the camera's role. It
    is idempotent: it never overwrites a zone that already exists, so an
    operator's own geometry is preserved.
    """
    from app.services.provisioning import provision_camera_defaults

    camera = _get_camera(session, camera_id)
    report = provision_camera_defaults(session, camera)
    write_audit(session, principal, "camera_zones.provisioned", entity_type="camera",
                entity_id=camera.id, detail=report, request=request)
    commit_before_camera_work(session)
    camera_manager.restart(camera_id)
    return {"camera_id": camera_id, **report}


@router.post(
    "/provision-zones",
    summary="Create default zones for every camera (admin)",
)
def provision_all_zones(
    request: Request, session: DbSession, principal: AdminUser,
) -> dict:
    """Provision defaults for every camera, then restart them so the newly
    created zones take effect in their pipelines."""
    from app.services.provisioning import provision_all

    report = provision_all(session)
    write_audit(session, principal, "camera_zones.provisioned_all", entity_type="system",
                detail={"zones": report["zones"], "shelves": report["shelves"]}, request=request)
    commit_before_camera_work(session)
    for row in session.query(Camera).all():
        camera_manager.restart(row.id)
    return report