"""Core ORM models: stores, users, audit logs and runtime settings."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.security import Role
from app.db.base import JSONEncoded, Base, UTCDateTime, new_uuid, pk_column, utcnow


class Store(Base):
    """A physical retail store that owns cameras, shelves and products."""

    __tablename__ = "stores"

    id: Mapped[str] = pk_column()
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    location: Mapped[str | None] = mapped_column(String(255))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="USD", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    cameras: Mapped[list["Camera"]] = relationship(back_populates="store")  # noqa: F821

    __table_args__ = (Index("ix_stores_active", "is_active"),)


class User(Base):
    """An operator account. Passwords are bcrypt hashes only."""

    __tablename__ = "users"

    id: Mapped[str] = pk_column()
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default=Role.STAFF.value, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (Index("ix_users_role_active", "role", "is_active"),)

    def is_admin(self) -> bool:
        return self.role == Role.ADMIN.value


class AuditLog(Base):
    """Append-only record of administrative actions (never stores secrets)."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = pk_column()
    user_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    username: Mapped[str | None] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(60), index=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONEncoded, default=dict)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    request_id: Mapped[str | None] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False, index=True)

    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)


class Setting(Base):
    """Runtime-configurable business rule.

    Business thresholds live here rather than in code so an operator can tune
    them from the dashboard.  ``value`` holds a JSON-encoded scalar/object.
    """

    __tablename__ = "settings"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    value: Mapped[Any] = mapped_column(JSONEncoded, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), default="float", nullable=False)
    category: Mapped[str] = mapped_column(String(40), default="general", nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("store_id", "key", name="uq_settings_store_key"),)

    @property
    def typed_value(self) -> Any:
        return _coerce(self.value, self.value_type)


class SystemSettingMeta(Base):
    """Tiny helper table recording schema/bootstrap version."""

    __tablename__ = "system_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)


def _coerce(value: Any, value_type: str) -> Any:
    """Coerce a JSON-decoded setting value into its declared type."""
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "bool":
        return bool(value)
    if value_type == "str":
        return str(value)
    return value


def default_settings_payload() -> list[dict[str, Any]]:
    """Canonical default business rules written by the seed/migration."""
    from app.core.config import settings as cfg

    return [
        {
            "key": "low_stock_threshold_pct",
            "value": cfg.default_low_stock_threshold,
            "value_type": "float",
            "category": "inventory",
            "description": "Estimated visible stock at or below this percentage of shelf capacity raises a LOW_STOCK condition.",
        },
        {
            "key": "out_of_stock_threshold_pct",
            "value": cfg.default_out_of_stock_threshold,
            "value_type": "float",
            "category": "inventory",
            "description": "Estimated visible stock at or below this percentage raises a POSSIBLE_OUT_OF_STOCK condition.",
        },
        {
            "key": "alert_min_confidence",
            "value": cfg.alert_min_confidence,
            "value_type": "float",
            "category": "alerts",
            "description": "Minimum estimate confidence required before an alert is raised.",
        },
        {
            "key": "alert_required_observations",
            "value": cfg.alert_required_observations,
            "value_type": "int",
            "category": "alerts",
            "description": "Number of consecutive stable observations required before an alert fires.",
        },
        {
            "key": "alert_cooldown_seconds",
            "value": cfg.alert_cooldown_seconds,
            "value_type": "int",
            "category": "alerts",
            "description": "Minimum seconds between two alerts sharing the same deduplication key.",
        },
        {
            "key": "queue_busy_threshold",
            "value": cfg.default_queue_busy,
            "value_type": "int",
            "category": "queue",
            "description": "People counted in a queue zone at or above this value = BUSY.",
        },
        {
            "key": "queue_critical_threshold",
            "value": cfg.default_queue_critical,
            "value_type": "int",
            "category": "queue",
            "description": "People counted in a queue zone at or above this value = CRITICAL.",
        },
        {
            "key": "queue_alert_cooldown_seconds",
            "value": 180,
            "value_type": "int",
            "category": "queue",
            "description": "Cooldown between repeated queue alerts for the same zone.",
        },
        {
            "key": "queue_service_time_seconds",
            "value": 150.0,
            "value_type": "float",
            "category": "queue",
            "description": "Assumed mean service time per checkout used for the waiting-time ESTIMATE.",
        },
        {
            "key": "footfall_dwell_alert_seconds",
            "value": 900.0,
            "value_type": "float",
            "category": "customers",
            "description": "Dwell time above this many seconds raises a dwell condition.",
        },
        {
            "key": "planogram_min_compliance_pct",
            "value": 85.0,
            "value_type": "float",
            "category": "planogram",
            "description": "Below this compliance percentage a planogram alert is raised.",
        },
        {
            "key": "camera_offline_alert_seconds",
            "value": 60.0,
            "value_type": "float",
            "category": "camera",
            "description": "Seconds without a frame before a CAMERA_OFFLINE alert is raised.",
        },
        {
            "key": "retention_days_detections",
            "value": cfg.detection_retention_days,
            "value_type": "int",
            "category": "privacy",
            "description": "Days of raw detection history retained before automatic cleanup.",
        },
        {
            "key": "retention_days_evidence",
            "value": cfg.evidence_retention_days,
            "value_type": "int",
            "category": "privacy",
            "description": "Days evidence snapshots are retained before automatic deletion.",
        },
        {
            "key": "evidence_enabled",
            "value": cfg.evidence_enabled,
            "value_type": "bool",
            "category": "privacy",
            "description": "Master switch for saving evidence snapshots (off = no imagery retained).",
        },
        {
            "key": "anonymize_faces",
            "value": cfg.anonymize_faces,
            "value_type": "bool",
            "category": "privacy",
            "description": "Blur detected faces in stored evidence images (privacy-by-default).",
        },
    ]


__all__ = ["Store", "User", "AuditLog", "Setting", "SystemSettingMeta", "default_settings_payload", "new_uuid"]