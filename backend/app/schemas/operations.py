"""Detection, inventory, alert, event, analytics, planogram, report and model schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import BBox, ORMModel

# ── Detections / streams ─────────────────────────────────────────────────────
class DetectionRead(ORMModel):
    id: str
    camera_id: str | None
    job_id: str | None
    shelf_id: str | None
    zone_id: str | None
    class_id: int
    class_name: str
    task: str
    product_id: str | None
    sku: str | None
    similarity: float | None
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int | None
    frame_timestamp: datetime
    source: str
    model_id: str | None
    inference_latency_ms: float | None


class DetectionCreate(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=80)
    task: str = "product"
    product_id: str | None = None
    sku: str | None = None
    similarity: float | None = Field(default=None, ge=0, le=1)
    bbox: BBox
    confidence: float = Field(ge=0, le=1)
    track_id: int | None = None


class StreamFrame(BaseModel):
    """Compact overlay payload pushed over the WebSocket (never raw frames)."""

    camera_id: str
    camera_name: str
    timestamp: datetime
    frame_index: int
    width: int
    height: int
    capture_fps: float
    inference_fps: float
    inference_latency_ms: float | None
    status: str
    stale: bool = False
    detections: list[DetectionCreate]
    person_count: int = 0
    model_id: str | None = None


# ── Inventory ────────────────────────────────────────────────────────────────
class InventoryEstimate(BaseModel):
    """A camera-derived estimate, explicitly separated from registered stock."""

    sku: str
    product_id: str | None = None
    product_name: str | None = None
    shelf_id: str | None = None
    shelf_code: str | None = None
    camera_id: str | None = None
    observed_quantity: int = 0
    estimated_quantity: float = 0.0
    capacity_units: int = 0
    stock_percentage: float = 0.0
    occupancy_pct: float | None = None
    confidence: float = 0.0
    observation_count: int = 0
    method: str = "detection_count"
    source: Literal["camera_estimate"] = "camera_estimate"
    observed_at: datetime
    observation_age_seconds: float | None = None
    # Registered / authoritative figures for comparison (may be None).
    registered_quantity: int | None = None
    registered_source: str | None = None
    discrepancy: int | None = None
    # Derived business state (rule-driven, thresholds configurable)
    state: Literal["ok", "low_stock", "out_of_stock", "unknown"] = "unknown"
    low_stock_threshold_pct: float | None = None
    out_of_stock_threshold_pct: float | None = None
    is_stale: bool = False


class InventoryItem(ORMModel):
    id: str
    sku: str
    product_id: str | None
    shelf_id: str | None
    camera_id: str | None
    observed_quantity: int
    estimated_quantity: float
    capacity_units: int
    stock_percentage: float
    occupancy_pct: float | None
    confidence: float
    observation_count: int
    source: str
    method: str
    observation_age_seconds: float | None
    notes: str | None
    observed_at: datetime


class InventoryAdjustment(BaseModel):
    """Manual correction. Always recorded with a reason and an audit trail."""

    new_quantity: int = Field(ge=0)
    reason: str = Field(min_length=3, max_length=500)
    reference: str | None = Field(default=None, max_length=80)
    create_transaction: bool = True


class InventoryTransactionRead(ORMModel):
    id: str
    sku: str
    product_id: str | None
    change: int
    resulting_quantity: int | None
    source: str
    reason: str | None
    reference: str | None
    performed_by: str | None
    created_at: datetime


class InventorySummary(BaseModel):
    total_skus: int = 0
    observed_skus: int = 0
    low_stock_count: int = 0
    out_of_stock_count: int = 0
    healthy_count: int = 0
    unknown_count: int = 0
    mean_confidence: float = 0.0
    mean_stock_percentage: float = 0.0
    generated_at: datetime
    stale_skus: int = 0


# ── Alerts ───────────────────────────────────────────────────────────────────
class AlertRead(ORMModel):
    id: str
    store_id: str | None
    camera_id: str | None
    shelf_id: str | None
    zone_id: str | None
    product_id: str | None
    sku: str | None
    alert_type: str
    severity: str
    status: str
    title: str
    message: str
    payload: dict[str, Any]
    evidence_path: str | None
    evidence_expires_at: datetime | None
    confidence: float
    observation_count: int
    dedup_key: str
    occurrence_count: int
    created_at: datetime
    last_observed_at: datetime
    acknowledged_at: datetime | None
    acknowledged_by: str | None
    resolved_at: datetime | None
    resolved_by: str | None
    resolution_note: str | None
    resolution_source: str | None


class AlertFilter(BaseModel):
    status: str | None = None
    severity: str | None = None
    alert_type: str | None = None
    camera_id: str | None = None
    shelf_id: str | None = None
    sku: str | None = None
    since: datetime | None = None
    until: datetime | None = None


class AcknowledgeRequest(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class ResolveRequest(BaseModel):
    note: str = Field(min_length=3, max_length=1000)
    resolution_source: Literal["condition_resolved", "manual", "false_positive"] = "manual"


class AlertSummary(BaseModel):
    active: int = 0
    acknowledged: int = 0
    resolved_24h: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_type: dict[str, int] = Field(default_factory=dict)
    oldest_active_at: datetime | None = None
    generated_at: datetime


# ── Events ───────────────────────────────────────────────────────────────────
class EventRead(ORMModel):
    id: str
    event_id: str
    event_type: str
    store_id: str | None
    camera_id: str | None
    shelf_id: str | None
    product_id: str | None
    alert_id: str | None
    severity: str
    payload: dict[str, Any]
    source: str
    model_id: str | None
    occurred_at: datetime


class EventEnvelope(BaseModel):
    """The wire format pushed over /ws/events (README §6.3/§6.4)."""

    event_id: str
    event_type: str
    timestamp: datetime
    severity: str
    store_id: str | None = None
    camera_id: str | None = None
    camera_name: str | None = None
    shelf_id: str | None = None
    sku: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


# ── Analytics ────────────────────────────────────────────────────────────────
class FootfallPoint(BaseModel):
    bucket: datetime
    entries: int = 0
    exits: int = 0
    net: int = 0
    current_occupancy: int | None = None


class FootfallResponse(BaseModel):
    camera_id: str | None
    camera_name: str | None
    granularity: str
    range_start: datetime
    range_end: datetime
    total_entries: int = 0
    total_exits: int = 0
    current_occupancy: int = 0
    peak_occupancy: int = 0
    mean_dwell_seconds: float | None = None
    max_dwell_seconds: float | None = None
    active_tracks: int = 0
    points: list[FootfallPoint] = Field(default_factory=list)
    note: str | None = None


class QueuePoint(BaseModel):
    bucket: datetime
    queue_length: float = 0.0
    estimated_wait_seconds: float | None = None
    status: str = "normal"
    service_zone_occupancy: int = 0


class QueueResponse(BaseModel):
    zone_id: str | None
    zone_name: str | None
    camera_id: str | None
    granularity: str
    range_start: datetime
    range_end: datetime
    current_queue_length: int = 0
    service_zone_occupancy: int = 0
    service_stations: int = 1
    estimated_wait_seconds: float | None = None
    status: str = "normal"
    busy_threshold: int
    critical_threshold: int
    mean_queue_length: float | None = None
    peak_queue_length: int | None = None
    points: list[QueuePoint] = Field(default_factory=list)
    note: str | None = None


class ShelfAnalyticsRow(BaseModel):
    shelf_id: str
    shelf_code: str
    shelf_name: str
    camera_id: str | None
    capacity_units: int
    observed_units: int = 0
    estimated_quantity: float = 0.0
    stock_percentage: float = 0.0
    occupancy_pct: float | None = None
    confidence: float = 0.0
    detected_skus: int = 0
    empty_positions: int | None = None
    planogram_compliance_pct: float | None = None
    last_observed_at: datetime | None = None
    state: str = "unknown"
    #: True only when the shelf was actually observed to have nothing on it.
    #: An unobserved shelf is False, never True.
    is_out_of_stock: bool = False
    out_of_stock_reason: str | None = None


class ShelfAnalyticsResponse(BaseModel):
    range_start: datetime
    range_end: datetime
    rows: list[ShelfAnalyticsRow] = Field(default_factory=list)
    note: str | None = None


class PlanogramComplianceRow(BaseModel):
    product_id: str | None
    sku: str | None
    product_name: str | None
    section_code: str | None
    expected_facings: int
    observed_facings: int
    facing_difference: int
    status: Literal["compliant", "missing", "misplaced", "over_faced", "low_confidence"]
    confidence: float = 0.0


class PlanogramCompliance(BaseModel):
    planogram_id: str | None
    shelf_id: str | None
    shelf_code: str | None
    version: int | None = None
    compliance_pct: float = 0.0
    checked_at: datetime
    rows: list[PlanogramComplianceRow] = Field(default_factory=list)
    note: str = (
        "Compliance compares the configured planogram against camera observations. "
        "It is only valid where a planogram is defined and detections are confident."
    )


class DashboardResponse(BaseModel):
    generated_at: datetime
    store_name: str | None = None
    cameras: dict[str, Any] = Field(default_factory=dict)
    inventory: InventorySummary
    alerts: AlertSummary
    customers: dict[str, Any] = Field(default_factory=dict)
    queues: list[dict[str, Any]] = Field(default_factory=list)
    ai: dict[str, Any] = Field(default_factory=dict)
    recent_events: list[EventEnvelope] = Field(default_factory=list)
    demo_mode: bool = False


# ── Planogram configuration ──────────────────────────────────────────────────
class PlanogramItemCreate(BaseModel):
    product_id: str | None = None
    section_id: str | None = None
    position_index: int = Field(default=0, ge=0)
    expected_facings: int = Field(default=1, ge=1, le=200)
    bbox: list[float] | None = None
    notes: str | None = None


class PlanogramItemRead(ORMModel):
    id: str
    product_id: str | None
    section_id: str | None
    position_index: int
    expected_facings: int
    bbox: list[float] | None
    notes: str | None


class PlanogramCreate(BaseModel):
    shelf_id: str
    name: str = Field(min_length=1, max_length=160)
    version: int | None = Field(default=None, ge=1)
    reference_image_path: str | None = None
    items: list[PlanogramItemCreate] = Field(default_factory=list)
    activate: bool = True


class PlanogramRead(ORMModel):
    id: str
    shelf_id: str
    version: int
    name: str
    status: str
    reference_image_path: str | None
    created_by: str | None
    activated_at: datetime | None
    created_at: datetime
    items: list[PlanogramItemRead] = Field(default_factory=list)


class PlanogramComparisonRequest(BaseModel):
    camera_id: str | None = None
    shelf_id: str | None = None
    lookback_minutes: int = Field(default=15, ge=1, le=1440)


# ── Reports ──────────────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    report_type: Literal[
        "inventory", "footfall", "queue", "alerts", "shelves", "detections", "anomalies"
    ]
    range_start: datetime | None = None
    range_end: datetime | None = None
    format: Literal["csv", "json", "pdf"] = "csv"
    camera_id: str | None = None
    shelf_id: str | None = None


class ReportResult(BaseModel):
    report_type: str
    format: str
    filename: str
    row_count: int
    generated_at: datetime
    download_path: str


# ── Models / monitoring ──────────────────────────────────────────────────────
class ModelVersionRead(ORMModel):
    id: str
    name: str
    task: str
    version: str
    status: str
    weights_path: str | None
    weights_sha256: str | None
    source: str | None
    provider: str
    input_size: int | None
    labels: list[str] | None
    confidence_threshold: float | None
    device: str | None
    metrics: dict[str, Any]
    metrics_measured_at: datetime | None
    notes: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ModelMetricsUpdate(BaseModel):
    metrics: dict[str, Any]
    notes: str | None = None


class InferenceRunRead(ORMModel):
    id: str
    model_id: str | None
    camera_id: str | None
    job_id: str | None
    started_at: datetime
    finished_at: datetime | None
    frames: int
    latency_ms_avg: float
    latency_ms_p95: float | None
    throughput_fps: float | None
    detections_count: int
    device: str | None
    status: str
    error: str | None


# ── Processing jobs ──────────────────────────────────────────────────────────
class JobRead(ORMModel):
    id: str
    camera_id: str | None
    uploaded_by: str | None
    original_filename: str
    content_type: str | None
    size_bytes: int
    status: str
    stage: str
    progress_pct: float
    error: str | None
    frame_width: int | None
    frame_height: int | None
    total_frames: int | None
    frames_processed: int
    duration_seconds: float | None
    source_fps: float | None
    detection_count: int
    events_created: int
    alerts_created: int
    result_summary: dict[str, Any]
    elapsed_seconds: float | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class JobCreateResponse(BaseModel):
    job: JobRead
    message: str


# ── Settings ─────────────────────────────────────────────────────────────────
class SettingRead(ORMModel):
    key: str
    value: Any
    value_type: str
    category: str
    description: str | None
    updated_by: str | None
    updated_at: datetime


class SettingUpdate(BaseModel):
    value: Any
    category: str | None = None
    description: str | None = None