"""Operational models: alerts, retail events, model registry, runs, health, sync."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONEncoded, Base, UTCDateTime, pk_column, utcnow


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Lifecycle from README §5.2."""

    CANDIDATE = "candidate"
    SUPPRESSED = "suppressed"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertType(str, Enum):
    LOW_STOCK = "inventory.low_stock"
    OUT_OF_STOCK = "inventory.out_of_stock"
    RESTOCK_DETECTED = "inventory.restock_detected"
    DISCREPANCY = "inventory.discrepancy"
    PLANOGRAM_MISMATCH = "planogram.mismatch"
    QUEUE_CRITICAL = "queue.critical"
    QUEUE_BUSY = "queue.busy"
    CAMERA_OFFLINE = "camera.offline"
    CAMERA_FROZEN = "camera.frozen"
    CAMERA_RECOVERED = "camera.recovered"
    MODEL_ERROR = "model.error"
    MODEL_DEGRADED = "model.degraded"
    ANOMALY_CANDIDATE = "anomaly.candidate"
    OCCUPANCY_HIGH = "customers.occupancy_high"
    DWELL_HIGH = "customers.dwell_high"
    SOURCE_ERROR = "system.source_error"


ACTIVE_STATUSES = (AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value)


class Alert(Base):
    """A business-rule alert with a full lifecycle and dedup/cooldown metadata."""

    __tablename__ = "alerts"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="SET NULL"), index=True)
    shelf_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelves.id", ondelete="SET NULL"), index=True)
    zone_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("camera_zones.id", ondelete="SET NULL"), index=True)
    product_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("products.id", ondelete="SET NULL"), index=True)
    sku: Mapped[str | None] = mapped_column(String(64), index=True)

    alert_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), default=AlertSeverity.WARNING.value, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default=AlertStatus.CANDIDATE.value, nullable=False, index=True)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)
    evidence_path: Mapped[str | None] = mapped_column(Text)
    evidence_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Dedup / cooldown bookkeeping
    dedup_key: Mapped[str] = mapped_column(String(191), nullable=False, index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    last_observed_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    acknowledged_by: Mapped[str | None] = mapped_column(String(80))
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    resolved_by: Mapped[str | None] = mapped_column(String(80))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    resolution_source: Mapped[str | None] = mapped_column(String(32))

    __table_args__ = (
        Index("ix_alerts_status_severity", "status", "severity"),
        Index("ix_alerts_camera_time", "camera_id", "created_at"),
        Index("ix_alerts_type_time", "alert_type", "created_at"),
        Index("ix_alerts_sku_status", "sku", "status"),
    )


class RetailEvent(Base):
    """The canonical, append-only business event log.

    Every significant thing that happens in a store produces exactly one row
    here with a globally unique ``event_id`` so downstream consumers (the
    WebSocket gateway, the optional cloud outbox) can process idempotently.
    """

    __tablename__ = "retail_events"

    id: Mapped[str] = pk_column()
    event_id: Mapped[str] = mapped_column(String(48), unique=True, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    store_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="SET NULL"), index=True)
    shelf_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelves.id", ondelete="SET NULL"), index=True)
    product_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("products.id", ondelete="SET NULL"), index=True)
    alert_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("alerts.id", ondelete="SET NULL"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True)

    severity: Mapped[str] = mapped_column(String(16), default=AlertSeverity.INFO.value, nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)
    evidence_path: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(24), default="pipeline", nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(64), index=True)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_events_type_time", "event_type", "occurred_at"),
        Index("ix_events_camera_time", "camera_id", "occurred_at"),
    )


class ModelVersion(Base):
    """The model registry: one row per deployed model artefact."""

    __tablename__ = "model_versions"

    id: Mapped[str] = pk_column()
    name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    task: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="baseline", nullable=False, index=True)

    weights_path: Mapped[str] = mapped_column(Text)
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str | None] = mapped_column(String(160))
    provider: Mapped[str] = mapped_column(String(40), default="pytorch")
    input_size: Mapped[int | None] = mapped_column(Integer)
    labels: Mapped[list[str] | None] = mapped_column(JSONEncoded)
    confidence_threshold: Mapped[float | None] = mapped_column(Float)
    device: Mapped[str | None] = mapped_column(String(40))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)
    metrics_measured_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    notes: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    runs: Mapped[list["InferenceRun"]] = relationship(back_populates="model", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
        Index("ix_model_task_active", "task", "is_active"),
    )


class InferenceRun(Base):
    """A measured inference execution — the evidence behind latency figures."""

    __tablename__ = "inference_runs"

    id: Mapped[str] = pk_column()
    model_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("model_versions.id", ondelete="SET NULL"), index=True)
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="SET NULL"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True)

    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    frames: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    latency_ms_avg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms_p95: Mapped[float | None] = mapped_column(Float)
    throughput_fps: Mapped[float | None] = mapped_column(Float)
    detections_count: Mapped[int] = mapped_column(Integer, default=0)
    device: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16), default="ok", nullable=False, index=True)
    error: Mapped[str | None] = mapped_column(Text)

    model: Mapped[ModelVersion | None] = relationship(back_populates="runs")

    __table_args__ = (Index("ix_runs_model_time", "model_id", "started_at"),)


class SystemHealthSample(Base):
    """Periodic snapshot of backend, worker and resource health."""

    __tablename__ = "system_health_samples"

    id: Mapped[str] = pk_column()
    measured_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    cpu_percent: Mapped[float | None] = mapped_column(Float)
    memory_percent: Mapped[float | None] = mapped_column(Float)
    memory_used_mb: Mapped[float | None] = mapped_column(Float)
    gpu_percent: Mapped[float | None] = mapped_column(Float)
    gpu_name: Mapped[str | None] = mapped_column(String(80))
    disk_percent: Mapped[float | None] = mapped_column(Float)
    disk_free_gb: Mapped[float | None] = mapped_column(Float)
    queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    active_workers: Mapped[int] = mapped_column(Integer, default=0)
    processing_errors: Mapped[int] = mapped_column(Integer, default=0)
    camera_reconnects: Mapped[int] = mapped_column(Integer, default=0)
    websocket_clients: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="healthy", nullable=False, index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)

    __table_args__ = (Index("ix_health_time", "measured_at"),)


class SyncOutbox(Base):
    """Optional cloud/ERP outbox (Layer 8). Unused in the local prototype.

    The table exists so the local-first architecture has a retry-safe hand-off
    point without changing any local behaviour.
    """

    __tablename__ = "sync_outbox"

    id: Mapped[str] = pk_column()
    event_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    target: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    next_attempt_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)

    __table_args__ = (UniqueConstraint("event_id", "target", name="uq_outbox_event_target"),)