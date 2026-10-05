"""Shared vision data types.

These are plain dataclasses (not ORM rows, not Pydantic models) because they
travel on hot paths between the capture thread, the inference worker and the
analytics stage.  Conversion to API/DB schemas happens at the boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(slots=True)
class BBox:
    """Pixel-space axis-aligned box."""

    x1: float
    y1: float
    x2: float
    y2: float

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

    def normalized(self, frame_width: int, frame_height: int) -> "BBox":
        if frame_width <= 0 or frame_height <= 0:
            return BBox(0.0, 0.0, 1.0, 1.0)
        return BBox(
            min(max(self.x1 / frame_width, 0.0), 1.0),
            min(max(self.y1 / frame_height, 0.0), 1.0),
            min(max(self.x2 / frame_width, 0.0), 1.0),
            min(max(self.y2 / frame_height, 0.0), 1.0),
        )

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(slots=True)
class Frame:
    """One decoded video frame plus everything needed to interpret it."""

    image: np.ndarray  # BGR uint8, shape (H, W, 3)
    index: int
    timestamp: float  # monotonic capture time
    captured_at: datetime = field(default_factory=utcnow)
    camera_id: str | None = None
    source_fps: float = 0.0
    is_demo: bool = False

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def signature(self) -> float:
        """Cheap frozen-frame signature (mean absolute luma + shape)."""
        if self.image.size == 0:
            return 0.0
        luma = self.image[..., :3].mean()
        return float(luma) + self.image.shape[0] * 1e-3 + self.image.shape[1] * 1e-6

    def validate(self) -> None:
        if self.image is None or self.image.size == 0:
            raise ValueError("Frame image is empty or corrupt.")
        if self.image.ndim != 3 or self.image.shape[2] not in (3, 4):
            raise ValueError(f"Frame must be HxWx3 or HxWx4, got {self.image.shape}")


@dataclass(slots=True)
class Detection:
    """A single detected object on a frame."""

    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    task: str = "product"
    track_id: int | None = None
    product_id: str | None = None
    sku: str | None = None
    similarity: float | None = None
    embedding: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_recognised(self) -> bool:
        return self.sku is not None and self.similarity is not None


@dataclass(slots=True)
class TrackedObject:
    """One row of a tracker's output for the current frame."""

    track_id: int
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class InferenceResult:
    """Everything one inference pass produced, with measured timings."""

    frame_index: int
    detections: list[Detection]
    tracks: list[TrackedObject] = field(default_factory=list)
    latency_ms: float = 0.0
    preprocessed_ms: float = 0.0
    postprocess_ms: float = 0.0
    model_id: str | None = None
    model_version: str | None = None
    device: str = "cpu"
    frame_width: int = 0
    frame_height: int = 0
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


@dataclass(slots=True)
class SourceStatus:
    """Live state of a frame source, used for the camera health endpoint."""

    source_type: str
    state: str = "idle"          # idle | connecting | running | error | stopped
    message: str | None = None
    frames_read: int = 0
    frames_yielded: int = 0
    frames_dropped: int = 0
    started_at: float | None = None
    last_frame_at: float | None = None
    #: Set by network sources when a connection is (re-)established. Declared
    #: here because ``SourceStatus`` uses ``slots``: without it, assigning this
    #: raises AttributeError and every RTSP/URL camera fails to open.
    last_connected_at: float | None = None
    last_error: str | None = None
    reconnect_count: int = 0
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    delivered_fps: float = 0.0
    is_frozen: bool = False
    #: True when this source paces itself so video time tracks wall-clock time.
    realtime: bool = False
    #: Delivered video-time ÷ wall-clock time. 1.0 means true realtime playback.
    realtime_ratio: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, time.time() - self.started_at)

    @property
    def seconds_since_last_frame(self) -> float | None:
        if self.last_frame_at is None:
            return None
        return max(0.0, time.time() - self.last_frame_at)

    @property
    def drop_ratio(self) -> float:
        total = self.frames_read + self.frames_dropped
        return self.frames_dropped / total if total else 0.0