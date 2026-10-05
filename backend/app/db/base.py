"""Declarative base, shared column types and common mixins."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, MetaData, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetime that also works on SQLite."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: type) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: datetime | None, dialect: type) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class JSONEncoded(TypeDecorator):
    """Portable JSON column (SQLite stores TEXT, PostgreSQL uses JSONB)."""

    impl = JSON
    cache_ok = True


class Base(DeclarativeBase):
    """Base class for every IRIS ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def to_dict(self) -> dict:
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }


def utcnow() -> datetime:
    """Timezone-aware current time (single source of truth for timestamps)."""
    return datetime.now(tz=timezone.utc)


def new_uuid() -> str:
    return uuid.uuid4().hex


class TimestampMixin:
    """Adds created_at / updated_at to a model."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


class IdMixin:
    """Adds a short, URL-safe, non-enumerable primary key."""

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=new_uuid, index=True
    )


class BaseModel(IdMixin, TimestampMixin, Base):
    """Convenience base combining id + timestamps."""

    __abstract__ = True


def pk_column() -> Mapped[str]:
    return mapped_column(String(32), primary_key=True, default=new_uuid)