"""Shared schema primitives: envelopes, pagination, geometry and health types."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas that mirror an ORM row (read-only by default)."""

    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    """Uniform pagination envelope used by every list endpoint."""

    items: list[T]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    has_more: bool = False

    @classmethod
    def build(cls, items: list[T], total: int, limit: int, offset: int) -> "Page[T]":
        return cls(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(items) < total,
        )


class MessageResponse(BaseModel):
    message: str
    detail: dict[str, Any] | None = None


class ActionResponse(BaseModel):
    ok: bool = True
    action: str
    message: str
    detail: dict[str, Any] | None = None


class BBox(BaseModel):
    """Axis-aligned box in normalised [0,1] coordinates."""

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: list[list[float]] = Field(min_length=3, max_length=64)


class ComponentHealth(BaseModel):
    """Health of one subsystem: HEALTHY / DEGRADED / OFFLINE."""

    name: str
    status: str = Field(pattern="^(healthy|degraded|offline)$")
    detail: str | None = None
    latency_ms: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class SystemHealthResponse(BaseModel):
    status: str = Field(pattern="^(healthy|degraded|offline)$")
    service: str
    version: str
    environment: str
    timestamp: datetime
    uptime_seconds: float
    components: list[ComponentHealth]
    metrics: dict[str, Any] = Field(default_factory=dict)