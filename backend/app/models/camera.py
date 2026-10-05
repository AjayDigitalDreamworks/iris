"""Camera, zone, health and video-processing-job ORM models."""

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
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONEncoded, Base, UTCDateTime, pk_column, utcnow


class SourceType(str, Enum):
    """Where frames come from."""

    RTSP = "rtsp"
    WEBCAM = "webcam"
    IP_WEBCAM = "ip_webcam"  # an Android phone streaming MJPEG via the IP Webcam app
    VIDEO_FILE = "video_file"
    IMAGE = "image"
    SYNTHETIC = "synthetic"  # DEMO MODE fixture generator - clearly labelled
    URL = "url"  # web video link (YouTube / Vimeo / direct media URL)


class CameraStatus(str, Enum):
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    DEGRADED = "degraded"      # producing frames but unhealthy (frozen/high drop)
    ERROR = "error"
    DISABLED = "disabled"


class CameraRole(str, Enum):
    SHELF = "shelf"
    ENTRANCE = "entrance"
    CHECKOUT = "checkout"
    QUEUE = "queue"  # a camera pointed at a queue/counter for queue-length alerts
    OTHER = "other"


class ZoneType(str, Enum):
    SHELF = "shelf"
    QUEUE = "queue"
    ENTRANCE_LINE = "entrance_line"
    EXIT_LINE = "exit_line"
    ZONE = "zone"


class Camera(Base):
    """A configured video source.

    ``stream_url_encrypted`` holds the credential-bearing URL encrypted at rest
    with Fernet.  The plaintext URL is never serialised into an API response.
    """

    __tablename__ = "cameras"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("stores.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    location: Mapped[str | None] = mapped_column(String(160))
    role: Mapped[str] = mapped_column(String(24), default=CameraRole.SHELF.value, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(24), nullable=False, index=True)

    # Credential-bearing location, encrypted at rest. ``stream_url_preview`` is
    # the redacted form that is safe to return to the browser.
    stream_url_encrypted: Mapped[str | None] = mapped_column(Text)
    stream_url_preview: Mapped[str | None] = mapped_column(String(255))
    file_path: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(String(24), default=CameraStatus.OFFLINE.value, nullable=False, index=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    capture_fps: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    frame_skip: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    reconnect_delay: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    reconnect_max_delay: Mapped[float] = mapped_column(Float, default=120.0, nullable=False)
    run_inference: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    privacy_mask: Mapped[list[float] | None] = mapped_column(JSONEncoded)
    notes: Mapped[str | None] = mapped_column(Text)

    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    codec: Mapped[str | None] = mapped_column(String(60))
    native_fps: Mapped[float | None] = mapped_column(Float)

    last_frame_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_connected_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_disconnected_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    reconnect_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_frames: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dropped_frames: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    store: Mapped["Store"] = relationship(back_populates="cameras")  # noqa: F821
    zones: Mapped[list["CameraZone"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_cameras_store_status", "store_id", "status"),
        Index("ix_cameras_store_enabled", "store_id", "is_enabled"),
        # Two cameras with the same name in one store are indistinguishable to
        # an operator and to the seed/upsert paths, which key on the name.
        # A unique index (not a table constraint) keeps this portable: SQLite
        # enforces it without the table rebuild a batch_alter would need.
        Index("uq_cameras_store_name", "store_id", "name", unique=True),
    )


class CameraZone(Base):
    """A polygon (shelf/queue/zone) or a crossing line (entrance/exit)."""

    __tablename__ = "camera_zones"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    zone_type: Mapped[str] = mapped_column(String(24), default=ZoneType.SHELF.value, nullable=False, index=True)
    # Normalised coordinates in [0,1] so a zone survives a resolution change.
    polygon: Mapped[list[list[float]]] = mapped_column(JSONEncoded, default=list, nullable=False)
    line: Mapped[list[list[float]] | None] = mapped_column(JSONEncoded)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Queue analytics
    service_stations: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    camera: Mapped[Camera] = relationship(back_populates="zones")

    __table_args__ = (Index("ix_camera_zones_type_active", "camera_id", "zone_type", "is_active"),)


class CameraHealthSample(Base):
    """Time-series of per-camera health for uptime/recovery reporting."""

    __tablename__ = "camera_health_samples"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    measured_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    capture_fps: Mapped[float | None] = mapped_column(Float)
    inference_fps: Mapped[float | None] = mapped_column(Float)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    frames_dropped: Mapped[int] = mapped_column(Integer, default=0)
    queue_depth: Mapped[int] = mapped_column(Integer, default=0)
    is_frozen: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_camera_health_camera_time", "camera_id", "measured_at"),)


class ProcessingJob(Base):
    """An offline analysis run over an uploaded video or image.

    Processing happens in a background thread so the browser never blocks and
    the API stays responsive while a job runs.
    """

    __tablename__ = "processing_jobs"

    id: Mapped[str] = pk_column()
    camera_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("cameras.id", ondelete="SET NULL"), index=True
    )
    uploaded_by: Mapped[str | None] = mapped_column(String(80))
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), index=True)

    status: Mapped[str] = mapped_column(String(24), default="queued", nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(48), default="queued", nullable=False)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    frame_width: Mapped[int | None] = mapped_column(Integer)
    frame_height: Mapped[int | None] = mapped_column(Integer)
    total_frames: Mapped[int | None] = mapped_column(Integer)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    source_fps: Mapped[float | None] = mapped_column(Float)

    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    events_created: Mapped[int] = mapped_column(Integer, default=0)
    alerts_created: Mapped[int] = mapped_column(Integer, default=0)
    result_summary: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float)

    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)