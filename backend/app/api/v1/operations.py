"""Inventory, customer analytics, queue, alert, event, analytics, model and system routers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy import func

from app.api.deps import (
    AdminUser,
    CurrentUser,
    DbSession,
    ManagerUser,
    PageParams,
    StaffUser,
    write_audit,
)
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models import ACTIVE_STATUSES, Alert, Detection, RetailEvent
from app.schemas import (
    AcknowledgeRequest,
    AlertRead,
    AlertSummary,
    DashboardResponse,
    DetectionRead,
    EventEnvelope,
    EventRead,
    FootfallResponse,
    InferenceRunRead,
    InventoryAdjustment,
    InventoryEstimate,
    InventoryItem,
    InventorySummary,
    InventoryTransactionRead,
    MessageResponse,
    ModelMetricsUpdate,
    ModelVersionRead,
    Page,
    QueueResponse,
    ReportRequest,
    ReportResult,
    ResolveRequest,
    SettingRead,
    SettingUpdate,
    ShelfAnalyticsResponse,
    ShelfAnalyticsRow,
)
from app.services.alert_service import alert_service
from app.services.analytics_service import RangeQuery, analytics_service
from app.services.db_service import paginate, run_in_session, settings_service
from app.services.inventory_service import inventory_service

logger = get_logger("app")

inventory_router = APIRouter(prefix="/inventory", tags=["inventory"])
customers_router = APIRouter(prefix="/customers", tags=["customers"])
queues_router = APIRouter(prefix="/queues", tags=["queues"])
alerts_router = APIRouter(prefix="/alerts", tags=["alerts"])
events_router = APIRouter(prefix="/events", tags=["events"])
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])
reports_router = APIRouter(prefix="/reports", tags=["reports"])
models_router = APIRouter(prefix="/models", tags=["models"])
detections_router = APIRouter(prefix="/detections", tags=["detections"])
system_router = APIRouter(prefix="/system", tags=["system"])


def _range(hours: int) -> RangeQuery:
    now = datetime.now(tz=timezone.utc)
    return RangeQuery(range_start=now - timedelta(hours=hours), range_end=now)


# ── Inventory ────────────────────────────────────────────────────────────────
def _to_estimate(view: Any) -> InventoryEstimate:
    return InventoryEstimate(
        sku=view.sku,
        product_id=view.product_id,
        product_name=view.product_name,
        shelf_id=view.shelf_id,
        shelf_code=view.shelf_code,
        camera_id=view.camera_id,
        observed_quantity=view.observed_quantity,
        estimated_quantity=view.estimated_quantity,
        capacity_units=view.capacity_units,
        stock_percentage=view.stock_percentage,
        occupancy_pct=view.occupancy_pct,
        confidence=view.confidence,
        observation_count=view.observation_count,
        method=view.method,
        observed_at=view.observed_at or datetime.now(tz=timezone.utc),
        observation_age_seconds=view.observation_age_seconds,
        registered_quantity=view.registered_quantity,
        registered_source=view.registered_source,
        discrepancy=view.discrepancy,
        state=view.state.value,
        low_stock_threshold_pct=view.low_stock_threshold_pct,
        out_of_stock_threshold_pct=view.out_of_stock_threshold_pct,
        is_stale=view.is_stale,
    )


@inventory_router.get("", response_model=list[InventoryEstimate], summary="Live inventory estimates")
def list_inventory(
    _: CurrentUser,
    store_id: str | None = None,
    camera_id: str | None = None,
    shelf_id: str | None = None,
    within_minutes: int = Query(default=15, ge=1, le=1440),
    only_problems: bool = False,
) -> list[InventoryEstimate]:
    """Current camera-derived estimates, explicitly separated from registered stock."""
    views = inventory_service.latest_estimates(
        store_id=store_id, camera_id=camera_id, shelf_id=shelf_id, within_minutes=within_minutes
    )
    estimates = [_to_estimate(v) for v in views]
    if only_problems:
        estimates = [e for e in estimates if e.state in {"low_stock", "out_of_stock"}]
    return estimates


@inventory_router.get("/summary", response_model=InventorySummary, summary="Inventory health summary")
def inventory_summary(_: CurrentUser, store_id: str | None = None) -> InventorySummary:
    data = inventory_service.summary(store_id=store_id)
    return InventorySummary(**data, generated_at=datetime.now(tz=timezone.utc))


@inventory_router.get(
    "/snapshots", response_model=Page[InventoryItem], summary="Raw estimate history"
)
def inventory_snapshots(
    session: DbSession, page: PageParams, _: CurrentUser,
    sku: str | None = None, camera_id: str | None = None, hours: int = 24,
) -> Page[InventoryItem]:
    from app.models import InventorySnapshot

    query = session.query(InventorySnapshot).filter(
        InventorySnapshot.observed_at >= datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    )
    if sku:
        query = query.filter(InventorySnapshot.sku == sku)
    if camera_id:
        query = query.filter(InventorySnapshot.camera_id == camera_id)
    rows, total = paginate(query.order_by(InventorySnapshot.observed_at.desc()), page.limit, page.offset)
    return Page.build([InventoryItem.model_validate(r) for r in rows], total, page.limit, page.offset)


@inventory_router.get("/transactions", response_model=list[InventoryTransactionRead],
                       summary="Stock movement audit trail")
def inventory_transactions(
    _: CurrentUser, sku: str | None = None, limit: int = 100
) -> list[InventoryTransactionRead]:
    rows, _ = inventory_service.transactions(sku=sku, limit=limit)
    return [InventoryTransactionRead.model_validate(r) for r in rows]


@inventory_router.get("/{sku}", summary="SKU detail with history and comparison")
def sku_detail(sku: str, _: CurrentUser, hours: int = 24) -> dict[str, Any]:
    data = inventory_service.sku_detail(sku)
    product = data["product"]
    shelf = data["shelf"]
    snapshots = data["snapshots"]
    latest = snapshots[-1] if snapshots else None
    return {
        "sku": sku,
        "product": {
            "id": product.id,
            "name": product.name,
            "brand": product.brand,
            "barcode": product.barcode,
            "expected_quantity": product.expected_quantity,
            "quantity_source": product.quantity_source,
            "min_threshold": product.min_threshold,
            "last_counted_at": product.last_counted_at,
            "last_counted_by": product.last_counted_by,
        },
        "shelf": {"id": shelf.id, "code": shelf.code, "capacity_units": shelf.capacity_units}
        if shelf else None,
        "latest_estimate": InventoryItem.model_validate(latest) if latest else None,
        "comparison": data["comparison"],
        "history": [InventoryItem.model_validate(s) for s in snapshots],
        "transactions": [InventoryTransactionRead.model_validate(t) for t in data["transactions"]],
        "note": (
            "The estimate is a camera observation of VISIBLE shelf stock. It never "
            "overwrites the registered quantity; human or POS confirmation is required "
            "for consequential inventory changes."
        ),
    }


@inventory_router.post(
    "/{sku}/adjust", summary="Manually correct registered stock (manager)"
)
def adjust_inventory(
    sku: str, payload: InventoryAdjustment, request: Request,
    principal: ManagerUser, session: DbSession,
) -> dict[str, Any]:
    result = inventory_service.adjust(
        sku,
        payload.new_quantity,
        reason=payload.reason,
        username=principal.username,
        reference=payload.reference,
        create_transaction=payload.create_transaction,
        session_factory=session,
    )
    write_audit(
        session, principal, "inventory.adjusted", entity_type="product",
        entity_id=result.get("transaction_id"),
        detail={"sku": sku, "change": result["change"],
                "previous": result["previous_quantity"], "new": result["new_quantity"]},
        request=request,
    )
    return result


# ── Customers ────────────────────────────────────────────────────────────────
@customers_router.get("/footfall", response_model=FootfallResponse, summary="Footfall analytics")
def footfall(
    _: CurrentUser,
    camera_id: str | None = None,
    hours: int = Query(default=24, ge=1, le=720),
    zone_id: str | None = None,
) -> FootfallResponse:
    query = _range(hours)
    query.camera_id = camera_id
    query.zone_id = zone_id
    return FootfallResponse(
        **{**analytics_service.footfall(query),
           "range_start": query.range_start, "range_end": query.range_end}
    )


@customers_router.get("/occupancy", summary="Current occupancy and active tracks")
def occupancy(_: CurrentUser, camera_id: str | None = None) -> dict[str, Any]:
    from app.models import Track
    from app.services.camera_manager import camera_manager

    def _load(session) -> dict[str, Any]:
        query = session.query(Track).filter(Track.is_active.is_(True))
        if camera_id:
            query = query.filter(Track.camera_id == camera_id)
        active = query.count()
        by_camera = dict(
            session.query(Track.camera_id, func.count(Track.id))
            .filter(Track.is_active.is_(True))
            .group_by(Track.camera_id)
            .all()
        )
        return {"active_tracks": active, "by_camera": by_camera}

    data = _load_session(_load)
    data["workers"] = {
        camera_id_: worker.health.as_dict() for camera_id_, worker in camera_manager.workers.items()
    }
    data["note"] = (
        "Track IDs are temporary and anonymous. IRIS performs no facial recognition and "
        "stores no identity information."
    )
    return data


def _load_session(work):
    return run_in_session(work)


# ── Queues ───────────────────────────────────────────────────────────────────
@queues_router.get("", response_model=list[QueueResponse], summary="Queue analytics per zone")
def queue_analytics(
    _: CurrentUser,
    camera_id: str | None = None,
    zone_id: str | None = None,
    hours: int = Query(default=24, ge=1, le=720),
) -> list[QueueResponse]:
    query = _range(hours)
    query.camera_id = camera_id
    query.zone_id = zone_id
    return [
        QueueResponse(**{**row, "range_start": query.range_start, "range_end": query.range_end})
        for row in analytics_service.queues(query)
    ]


@queues_router.get("/thresholds", summary="Configured queue thresholds")
def queue_thresholds(_: CurrentUser) -> dict[str, Any]:
    service = settings_service
    return {
        "busy_threshold": service.get_int("queue_busy_threshold", 4),
        "critical_threshold": service.get_int("queue_critical_threshold", 7),
        "service_time_seconds": service.get_float("queue_service_time_seconds", 150.0),
        "note": (
            "Waiting time is an M/M/c estimate under the stated service-time assumption. "
            "Thresholds are configurable from the Settings page."
        ),
    }


# ── Alerts ───────────────────────────────────────────────────────────────────
@alerts_router.get("", response_model=Page[AlertRead], summary="List alerts")
def list_alerts(
    session: DbSession,
    page: PageParams,
    _: CurrentUser,
    status: str | None = None,
    severity: str | None = None,
    alert_type: str | None = None,
    camera_id: str | None = None,
    sku: str | None = None,
    hours: int | None = None,
) -> Page[AlertRead]:
    query = session.query(Alert)
    if status:
        query = query.filter(Alert.status == status)
    if severity:
        query = query.filter(Alert.severity == severity)
    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)
    if camera_id:
        query = query.filter(Alert.camera_id == camera_id)
    if sku:
        query = query.filter(Alert.sku == sku)
    if hours:
        query = query.filter(
            Alert.created_at >= datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        )
    rows, total = paginate(query.order_by(Alert.created_at.desc()), page.limit, page.offset)
    return Page.build([AlertRead.model_validate(a) for a in rows], total, page.limit, page.offset)


@alerts_router.get("/active", response_model=list[AlertRead], summary="Active and acknowledged alerts")
def active_alerts(session: DbSession, _: CurrentUser) -> list[AlertRead]:
    rows = (
        session.query(Alert)
        .filter(Alert.status.in_(ACTIVE_STATUSES))
        .order_by(Alert.created_at.desc())
        .all()
    )
    return [AlertRead.model_validate(a) for a in rows]


@alerts_router.get("/summary", response_model=AlertSummary, summary="Alert counts")
def alerts_summary(_: CurrentUser) -> AlertSummary:
    data = analytics_service.alert_summary(_range(24))
    return AlertSummary(**data, generated_at=datetime.now(tz=timezone.utc))


@alerts_router.get("/{alert_id}", response_model=AlertRead, summary="Alert detail")
def get_alert(alert_id: str, session: DbSession, _: CurrentUser) -> AlertRead:
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise NotFoundError(f"Alert {alert_id!r} does not exist.")
    return AlertRead.model_validate(alert)


@alerts_router.post("/{alert_id}/acknowledge", response_model=AlertRead, summary="Acknowledge (staff)")
def acknowledge_alert(
    alert_id: str, payload: AcknowledgeRequest, request: Request,
    principal: StaffUser, session: DbSession,
) -> AlertRead:
    alert = alert_service.acknowledge(alert_id, principal.username, payload.note)
    write_audit(session, principal, "alert.acknowledged", entity_type="alert",
                entity_id=alert_id, request=request)
    return AlertRead.model_validate(alert)


@alerts_router.post("/{alert_id}/resolve", response_model=AlertRead, summary="Resolve (staff)")
def resolve_alert(
    alert_id: str, payload: ResolveRequest, request: Request,
    principal: StaffUser, session: DbSession,
) -> AlertRead:
    alert = alert_service.resolve(
        alert_id, principal.username, payload.note,
        resolution_source=payload.resolution_source,
    )
    write_audit(
        session, principal, "alert.resolved", entity_type="alert", entity_id=alert_id,
        detail={"source": payload.resolution_source, "note": payload.note[:200]}, request=request,
    )
    return AlertRead.model_validate(alert)


# ── Events ───────────────────────────────────────────────────────────────────
@events_router.get("", response_model=Page[EventRead], summary="Event history")
def list_events(
    session: DbSession, page: PageParams, _: CurrentUser,
    event_type: str | None = None, severity: str | None = None, camera_id: str | None = None,
    hours: int = 24,
) -> Page[EventRead]:
    query = session.query(RetailEvent).filter(
        RetailEvent.occurred_at >= datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    )
    if event_type:
        query = query.filter(RetailEvent.event_type == event_type)
    if severity:
        query = query.filter(RetailEvent.severity == severity)
    if camera_id:
        query = query.filter(RetailEvent.camera_id == camera_id)
    rows, total = paginate(query.order_by(RetailEvent.occurred_at.desc()), page.limit, page.offset)
    return Page.build([EventRead.model_validate(r) for r in rows], total, page.limit, page.offset)


@events_router.get("/recent", response_model=list[EventEnvelope], summary="Recent live events")
def recent_events(_: CurrentUser, limit: int = 25) -> list[EventEnvelope]:
    rows = analytics_service.recent_events(limit=limit)
    return [
        EventEnvelope(
            event_id=r.event_id,
            event_type=r.event_type,
            timestamp=r.occurred_at,
            severity=r.severity,
            store_id=r.store_id,
            camera_id=r.camera_id,
            shelf_id=r.shelf_id,
            message=(r.payload or {}).get("message") or r.event_type,
            payload=r.payload or {},
            evidence={"path": r.evidence_path} if r.evidence_path else {},
        )
        for r in rows
    ]


@events_router.get("/types", summary="Event type catalogue")
def event_types(_: CurrentUser) -> dict[str, Any]:
    from app.services.event_service import EventType, iter_event_types

    return {
        "event_types": list(iter_event_types()),
        "coalescing": sorted(
            value
            for name, value in vars(EventType).items()
            if not name.startswith("_") and value in {"detection.summary", "queue.updated",
                                                    "customers.occupancy", "inventory.updated",
                                                    "camera.status"}
        ),
    }


# ── Analytics ────────────────────────────────────────────────────────────────
@analytics_router.get("/dashboard", response_model=DashboardResponse, summary="Overview dashboard")
def dashboard(_: CurrentUser, store_id: str | None = None) -> DashboardResponse:
    data = analytics_service.dashboard(store_id=store_id)
    from app.core.config import settings

    data["demo_mode"] = settings.demo_mode
    return DashboardResponse(
        inventory=InventorySummary(
            **data["inventory"], generated_at=datetime.now(tz=timezone.utc)
        ),
        alerts=AlertSummary(**data["alerts"], generated_at=datetime.now(tz=timezone.utc)),
        **{k: v for k, v in data.items()
           if k not in {"inventory", "alerts", "recent_events", "skus_configured"}},
        recent_events=[
            EventEnvelope(
                event_id=e["event_id"],
                event_type=e["event_type"],
                timestamp=e["timestamp"],
                severity=e["severity"],
                camera_id=e.get("camera_id"),
                message=e.get("message"),
                payload=e.get("payload", {}),
            )
            for e in data["recent_events"]
        ],
    )


@analytics_router.get("/shelves", response_model=ShelfAnalyticsResponse, summary="Shelf analytics")
def shelf_analytics(
    _: CurrentUser, store_id: str | None = None, camera_id: str | None = None,
    shelf_id: str | None = None, hours: int = 24,
) -> ShelfAnalyticsResponse:
    query = _range(hours)
    query.store_id = store_id
    query.camera_id = camera_id
    query.shelf_id = shelf_id
    rows = analytics_service.shelves(query)
    return ShelfAnalyticsResponse(
        range_start=query.range_start,
        range_end=query.range_end,
        rows=[ShelfAnalyticsRow(**row) for row in rows],
        note="Occupancy is an area ratio from the shelf baseline analyser, not an item count.",
    )


@analytics_router.get("/detections", summary="Detection statistics")
def detection_analytics(
    _: CurrentUser, camera_id: str | None = None, hours: int = 24
) -> dict[str, Any]:
    query = _range(hours)
    query.camera_id = camera_id
    return analytics_service.detection_summary(query)


# ── Reports ──────────────────────────────────────────────────────────────────
@reports_router.post("/export", response_model=ReportResult, summary="Generate a report export")
def export_report(payload: ReportRequest, _: ManagerUser) -> ReportResult:
    from app.services import report_service

    result = report_service.generate(
        payload.report_type,
        range_start=payload.range_start,
        range_end=payload.range_end,
        output_format=payload.format,
        camera_id=payload.camera_id,
        shelf_id=payload.shelf_id,
    )
    return ReportResult(**{k: v for k, v in result.items() if not k.startswith("_")})


@reports_router.get("/download", summary="Download a generated report")
def download_report(
    _: CurrentUser, path: str, request: Request
) -> Any:
    from fastapi.responses import FileResponse

    from app.core.errors import NotFoundError as NF
    from app.services import report_service

    resolved = report_service.resolve_export(path)
    if resolved is None:
        raise NF("Report file not found.")
    _ = request
    return FileResponse(resolved, filename=resolved.name)


# ── Detections ───────────────────────────────────────────────────────────────
@detections_router.get("", response_model=Page[DetectionRead], summary="Detection history")
def list_detections(
    session: DbSession, page: PageParams, _: CurrentUser,
    camera_id: str | None = None, sku: str | None = None,
    class_name: str | None = None, hours: int = 1,
) -> Page[DetectionRead]:
    query = session.query(Detection).filter(
        Detection.frame_timestamp >= datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    )
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    if sku:
        query = query.filter(Detection.sku == sku)
    if class_name:
        query = query.filter(Detection.class_name == class_name)
    rows, total = paginate(query.order_by(Detection.frame_timestamp.desc()), page.limit, page.offset)
    return Page.build([DetectionRead.model_validate(r) for r in rows], total, page.limit, page.offset)


# ── Models ───────────────────────────────────────────────────────────────────
@models_router.get("", response_model=list[ModelVersionRead], summary="Registered models")
def list_models(session: DbSession, _: CurrentUser) -> list[ModelVersionRead]:
    from app.models import ModelVersion

    rows = session.query(ModelVersion).order_by(ModelVersion.task, ModelVersion.name).all()
    return [ModelVersionRead.model_validate(r) for r in rows]


@models_router.get("/registry", summary="Live registry status including missing weights")
def model_registry(_: CurrentUser) -> dict[str, Any]:
    from app.services.camera_manager import camera_manager

    return camera_manager.ai_summary()


@models_router.get("/{model_id}", response_model=ModelVersionRead, summary="Model detail")
def get_model(model_id: str, session: DbSession, _: CurrentUser) -> ModelVersionRead:
    from app.models import ModelVersion

    model = session.get(ModelVersion, model_id)
    if model is None:
        raise NotFoundError(f"Model {model_id!r} does not exist.")
    return ModelVersionRead.model_validate(model)


@models_router.patch(
    "/{model_id}/metrics", response_model=ModelVersionRead,
    summary="Record MEASURED evaluation metrics (admin)",
)
def update_model_metrics(
    model_id: str, payload: ModelMetricsUpdate, request: Request,
    session: DbSession, principal: AdminUser,
) -> ModelVersionRead:
    from app.models import ModelVersion

    model = session.get(ModelVersion, model_id)
    if model is None:
        raise NotFoundError(f"Model {model_id!r} does not exist.")
    model.metrics = {**(model.metrics or {}), **payload.metrics}
    model.metrics_measured_at = datetime.now(tz=timezone.utc)
    if payload.notes:
        model.notes = payload.notes
    session.flush()
    write_audit(session, principal, "model.metrics_updated", entity_type="model_version",
                entity_id=model_id, detail={"metrics": payload.metrics}, request=request)
    return ModelVersionRead.model_validate(model)


@models_router.get("/runs/recent", response_model=list[InferenceRunRead], summary="Inference runs")
def recent_runs(_: CurrentUser, limit: int = 50) -> list[InferenceRunRead]:
    from app.models import InferenceRun

    def _load(session) -> list[InferenceRun]:
        return list(
            session.query(InferenceRun)
            .order_by(InferenceRun.started_at.desc())
            .limit(min(limit, 500))
            .all()
        )

    return [InferenceRunRead.model_validate(r) for r in run_in_session(_load)]


# ── System ───────────────────────────────────────────────────────────────────
@system_router.get("/health", summary="Application and service health")
def system_health() -> dict[str, Any]:
    from app.services.health_service import collect

    snapshot = collect()
    return {
        **snapshot,
        "components": [
            {"name": name, **{k: v for k, v in data.items() if k != "status"},
             "status": data.get("status", "unknown")}
            for name, data in snapshot["components"].items()
        ],
    }


@system_router.get("/health/history", summary="Recorded health samples")
def system_health_history(_: CurrentUser, limit: int = 100) -> list[dict[str, Any]]:
    from app.services.health_service import history

    return [
        {
            "measured_at": s.measured_at,
            "status": s.status,
            "cpu_percent": s.cpu_percent,
            "memory_percent": s.memory_percent,
            "gpu_percent": s.gpu_percent,
            "disk_percent": s.disk_percent,
            "queue_depth": s.queue_depth,
            "processing_errors": s.processing_errors,
            "websocket_clients": s.websocket_clients,
        }
        for s in history(limit=limit)
    ]


@system_router.get("/settings", response_model=list[SettingRead], summary="Runtime business rules")
def list_settings(session: DbSession, _: CurrentUser) -> list[SettingRead]:
    from app.models import Setting

    rows = session.query(Setting).order_by(Setting.category, Setting.key).all()
    return [
        SettingRead(
            key=r.key,
            value=r.typed_value,
            value_type=r.value_type,
            category=r.category,
            description=r.description,
            updated_by=r.updated_by,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@system_router.put("/settings/{key}", response_model=SettingRead, summary="Update a rule (admin)")
def update_setting(
    key: str, payload: SettingUpdate, request: Request,
    session: DbSession, principal: AdminUser,
) -> SettingRead:
    settings_service.upsert(
        key,
        payload.value,
        category=payload.category or "general",
        description=payload.description,
        updated_by=principal.username,
        session=session,
    )
    settings_service.invalidate()
    write_audit(session, principal, "setting.updated", entity_type="setting", entity_id=key,
                detail={"key": key}, request=request)
    from app.models import Setting

    row = session.query(Setting).filter(Setting.key == key).first()
    return SettingRead(
        key=row.key, value=row.typed_value, value_type=row.value_type, category=row.category,
        description=row.description, updated_by=row.updated_by, updated_at=row.updated_at,
    )


@system_router.get("/config", summary="Effective configuration (secrets redacted)")
def effective_config(_: CurrentUser) -> dict[str, Any]:
    from app.core.config import settings as cfg

    safe = cfg.model_dump(exclude={"secret_key", "jwt_secret", "default_admin_password"})
    return {
        "settings": {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                     for k, v in safe.items()},
        "warnings": cfg.warn_if_insecure(),
    }


@system_router.get("/privacy", summary="Privacy and retention status")
def privacy_status(_: CurrentUser) -> dict[str, Any]:
    from app.services.retention_service import retention_summary

    return retention_summary()


@system_router.post("/retention/run", summary="Run a retention pass (admin)")
def run_retention(_: AdminUser) -> dict[str, Any]:
    from app.services.retention_service import cleanup

    return cleanup()


@system_router.post("/maintenance/sample", summary="Record a health sample (admin)")
def record_sample(_: AdminUser) -> dict[str, Any]:
    from app.services.health_service import record_sample

    sample = record_sample()
    return {"recorded": sample is not None, "measured_at": sample.measured_at if sample else None}


@system_router.get("/audit", summary="Audit log (admin)")
def audit_log(
    session: DbSession, _: AdminUser, page: PageParams, action: str | None = None,
    entity_type: str | None = None,
) -> Page[dict[str, Any]]:
    from app.models import AuditLog

    query = session.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action)
    if entity_type:
        query = query.filter(AuditLog.entity_type == entity_type)
    rows, total = paginate(query.order_by(AuditLog.created_at.desc()), page.limit, page.offset)
    return Page.build(
        [
            {
                "id": r.id,
                "username": r.username,
                "action": r.action,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "detail": r.detail,
                "ip_address": r.ip_address,
                "request_id": r.request_id,
                "created_at": r.created_at,
            }
            for r in rows
        ],
        total, page.limit, page.offset,
    )


@system_router.get("/stats", summary="Service counters")
def service_stats(_: CurrentUser) -> dict[str, Any]:
    from app.services.camera_manager import camera_manager
    from app.services.event_service import event_service
    from app.services.jobs_service import job_service

    return {
        "cameras": camera_manager.stats(),
        "events": event_service.stats(),
        "jobs": job_service.stats(),
        "alerts": alert_service.stats(),
    }