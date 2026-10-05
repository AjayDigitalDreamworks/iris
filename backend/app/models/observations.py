"""Observation models: detections, tracks, footfall, queue and inventory history."""

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


class DetectionSource(str, Enum):
    LIVE = "live"
    UPLOAD = "upload"
    IMAGE = "image"
    DEMO = "demo"


class Detection(Base):
    """One object detection produced by a model on one frame."""

    __tablename__ = "detections"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("processing_jobs.id", ondelete="CASCADE"), index=True)
    shelf_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelves.id", ondelete="SET NULL"), index=True)
    zone_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("camera_zones.id", ondelete="SET NULL"), index=True)

    # What was seen
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    task: Mapped[str] = mapped_column(String(40), default="product", nullable=False, index=True)

    # Recognition result (may be 'unknown')
    product_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("products.id", ondelete="SET NULL"), index=True)
    sku: Mapped[str | None] = mapped_column(String(64), index=True)
    similarity: Mapped[float | None] = mapped_column(Float)

    # Geometry (pixel coordinates of the ORIGINAL frame)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    track_id: Mapped[int | None] = mapped_column(Integer, index=True)

    frame_timestamp: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(16), default=DetectionSource.LIVE.value, nullable=False, index=True)
    model_id: Mapped[str | None] = mapped_column(String(64), index=True)
    frame_index: Mapped[int | None] = mapped_column(Integer)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)

    __table_args__ = (
        Index("ix_detections_camera_time", "camera_id", "frame_timestamp"),
        Index("ix_detections_sku_time", "sku", "frame_timestamp"),
        Index("ix_detections_job", "job_id", "frame_index"),
    )


class Track(Base):
    """An anonymous, temporary multi-object track.

    Track IDs are ephemeral by design: they identify a blob of pixels over a
    short time window and are NOT a person identity.  They expire after
    ``settings.track_max_ttl_seconds``.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    track_uid: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    external_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    first_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    frames_seen: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    zone_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("camera_zones.id", ondelete="SET NULL"))
    entered_line_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    exited_line_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    dwell_seconds: Mapped[float | None] = mapped_column(Float)
    centroid_path: Mapped[list[list[float]] | None] = mapped_column(JSONEncoded)

    __table_args__ = (
        UniqueConstraint("camera_id", "track_uid", "external_id", name="uq_track_camera_uid_external"),
        Index("ix_tracks_camera_active", "camera_id", "is_active"),
    )


class FootfallEvent(Base):
    """A direction-aware line crossing (entry / exit)."""

    __tablename__ = "footfall_events"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    zone_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("camera_zones.id", ondelete="SET NULL"), index=True)
    track_uid: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    external_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # in | out
    direction: Mapped[str | None] = mapped_column(String(8))
    crossing_x: Mapped[float | None] = mapped_column(Float)
    crossing_y: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True)

    __table_args__ = (
        Index("ix_footfall_camera_time", "camera_id", "occurred_at"),
        Index("ix_footfall_zone_type_time", "zone_id", "event_type", "occurred_at"),
    )


class QueueSnapshot(Base):
    """Queue metrics sampled over time for one queue zone."""

    __tablename__ = "queue_snapshots"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    zone_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("camera_zones.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True)
    queue_length: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    service_zone_occupancy: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    service_stations: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    estimated_wait_seconds: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="normal", nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    measured_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)

    __table_args__ = (Index("ix_queue_zone_time", "zone_id", "measured_at"),)


class InventorySnapshot(Base):
    """A camera-derived *estimate* of visible shelf stock.

    This is deliberately a separate table from the authoritative inventory
    figure on :class:`Product`.  A camera estimate never overwrites registered
    stock.
    """

    __tablename__ = "inventory_snapshots"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    shelf_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelves.id", ondelete="CASCADE"), index=True)
    product_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("products.id", ondelete="SET NULL"), index=True)
    sku: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True)

    observed_quantity: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_quantity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    capacity_units: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stock_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    occupancy_pct: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    observation_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    source: Mapped[str] = mapped_column(String(24), default="camera_estimate", nullable=False)
    method: Mapped[str] = mapped_column(String(48), default="detection_count", nullable=False)
    observation_age_seconds: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_inventory_sku_time", "sku", "observed_at"),
        Index("ix_inventory_shelf_time", "shelf_id", "observed_at"),
    )


class InventoryTransaction(Base):
    """An append-only stock movement (sales, restock, manual correction)."""

    __tablename__ = "inventory_transactions"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True)
    product_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("products.id", ondelete="SET NULL"), index=True)
    sku: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    shelf_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelves.id", ondelete="SET NULL"), index=True)

    change: Mapped[int] = mapped_column(Integer, nullable=False)
    resulting_quantity: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text)
    reference: Mapped[str | None] = mapped_column(String(80))
    performed_by: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)

    __table_args__ = (Index("ix_inventory_tx_sku_time", "sku", "created_at"),)