"""Auth, camera and catalog schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.common import ORMModel


# ── Auth ─────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserRead"


class UserBase(BaseModel):
    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    full_name: str = Field(min_length=1, max_length=160)
    email: EmailStr | None = None
    role: Literal["admin", "manager", "staff", "service"] = "staff"


class UserCreate(UserBase):
    password: str = Field(min_length=10, max_length=256)


class UserRead(ORMModel):
    id: str
    username: str
    full_name: str
    email: str | None = None
    role: str
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=160)
    email: EmailStr | None = None
    role: Literal["admin", "manager", "staff", "service"] | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=10, max_length=256)


# ── Camera ───────────────────────────────────────────────────────────────────
class CameraBase(BaseModel):
    store_id: str
    name: str = Field(min_length=1, max_length=120)
    location: str | None = Field(default=None, max_length=160)
    role: Literal["shelf", "entrance", "checkout", "queue", "other"] = "shelf"
    source_type: Literal["rtsp", "webcam", "ip_webcam", "video_file", "image", "synthetic", "url"]
    capture_fps: float = Field(default=5.0, gt=0, le=60)
    frame_skip: int = Field(default=1, ge=1, le=60)
    reconnect_delay: float = Field(default=5.0, gt=0, le=600)
    reconnect_max_delay: float = Field(default=120.0, gt=0, le=3600)
    run_inference: bool = True
    is_enabled: bool = True
    notes: str | None = None


class CameraCreate(CameraBase):
    """Credentials are write-only: they are encrypted and never returned."""

    stream_url: str | None = Field(default=None, max_length=1024)
    file_path: str | None = Field(default=None, max_length=1024)
    image_path: str | None = Field(default=None, max_length=1024)
    privacy_mask: list[float] | None = None

    @field_validator("stream_url")
    @classmethod
    def _stream_url_scheme(cls, value: str | None, info) -> str | None:
        """A scheme is required — except for a phone's IP Webcam address.

        IP Webcam addresses are commonly typed bare (``192.168.1.42:8080``);
        :func:`app.vision.ingestion.sources.normalise_ip_webcam_url` adds the
        scheme when the source is opened.
        """
        if not value:
            return value
        if info.data.get("source_type") == "ip_webcam":
            return value
        if not value.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise ValueError("stream_url must start with rtsp://, rtsps://, http:// or https://")
        return value

    @model_validator(mode="after")
    def _source_needs_its_reference(self) -> "CameraCreate":
        """A source that needs a URL/path must actually be given one.

        Catching it here returns a 422 naming the missing field, instead of
        letting the camera be created and then fail every time it starts.
        """
        if self.source_type == "rtsp" and not self.stream_url:
            raise ValueError("An RTSP camera requires a stream_url.")
        if self.source_type == "ip_webcam" and not self.stream_url:
            raise ValueError(
                "An IP Webcam camera requires a stream_url (the phone's address, "
                "e.g. 192.168.1.42:8080)."
            )
        if self.source_type == "url" and not self.stream_url:
            raise ValueError(
                "A web-video camera requires a stream_url (a YouTube, Vimeo or "
                "direct media link)."
            )
        if self.source_type == "video_file" and not self.file_path:
            raise ValueError("A video-file camera requires a file_path.")
        if self.source_type == "image" and not (self.image_path or self.file_path):
            raise ValueError("An image camera requires an image_path.")
        if self.source_type == "webcam" and self.file_path:
            # The webcam device index reuses the file_path field ("0", "1",
            # "webcam:2"). Reject garbage here instead of at camera start.
            raw = self.file_path.strip()
            if raw.startswith("webcam:"):
                raw = raw.split(":", 1)[1]
            if not raw.isdigit():
                raise ValueError(
                    "A webcam camera's device index must be a number, e.g. 0, 1, 2."
                )
        return self


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    location: str | None = Field(default=None, max_length=160)
    role: Literal["shelf", "entrance", "checkout", "queue", "other"] | None = None
    capture_fps: float | None = Field(default=None, gt=0, le=60)
    frame_skip: int | None = Field(default=None, ge=1, le=60)
    reconnect_delay: float | None = Field(default=None, gt=0, le=600)
    reconnect_max_delay: float | None = Field(default=None, gt=0, le=3600)
    run_inference: bool | None = None
    is_enabled: bool | None = None
    notes: str | None = None
    stream_url: str | None = Field(default=None, max_length=1024)
    privacy_mask: list[float] | None = None


class CameraRead(ORMModel):
    id: str
    store_id: str
    name: str
    location: str | None
    role: str
    source_type: str
    # Redacted: NEVER contains the password.
    stream_url_preview: str | None = None
    has_credentials: bool = False
    status: str
    is_enabled: bool
    capture_fps: float
    frame_skip: int
    run_inference: bool
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    native_fps: float | None = None
    last_frame_at: datetime | None = None
    last_connected_at: datetime | None = None
    last_disconnected_at: datetime | None = None
    last_error: str | None = None
    reconnect_count: int = 0
    total_frames: int = 0
    dropped_frames: int = 0
    privacy_mask: list[float] | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class CameraZoneBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    zone_type: Literal["shelf", "queue", "entrance_line", "exit_line", "zone"] = "shelf"
    polygon: list[list[float]] = Field(default_factory=list)
    line: list[list[float]] | None = None
    is_active: bool = True
    service_stations: int = Field(default=1, ge=1, le=32)
    description: str | None = None

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: list[list[float]]) -> list[list[float]]:
        for point in value:
            if len(point) != 2:
                raise ValueError("Each polygon point must be [x, y]")
            for coord in point:
                if not 0.0 <= coord <= 1.0:
                    raise ValueError("Polygon coordinates must be normalised to [0,1]")
        return value


class CameraZoneCreate(CameraZoneBase):
    camera_id: str


class CameraZoneUpdate(BaseModel):
    name: str | None = None
    zone_type: Literal["shelf", "queue", "entrance_line", "exit_line", "zone"] | None = None
    polygon: list[list[float]] | None = None
    line: list[list[float]] | None = None
    is_active: bool | None = None
    service_stations: int | None = Field(default=None, ge=1, le=32)
    description: str | None = None


class CameraZoneRead(ORMModel):
    id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list[list[float]]
    line: list[list[float]] | None = None
    is_active: bool
    service_stations: int
    description: str | None = None
    created_at: datetime


class CameraHealthRead(BaseModel):
    camera_id: str
    name: str
    status: str
    is_enabled: bool
    is_stale: bool
    is_frozen: bool
    seconds_since_last_frame: float | None = None
    capture_fps: float
    inference_fps: float = 0.0
    inference_latency_ms: float | None = None
    queue_depth: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    drop_ratio: float = 0.0
    reconnect_count: int = 0
    uptime_ratio_24h: float | None = None
    width: int | None = None
    height: int | None = None
    last_error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ConnectionTestResult(BaseModel):
    ok: bool
    camera_id: str | None = None
    message: str
    latency_ms: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frames_read: int = 0
    error: str | None = None


# ── Catalog ──────────────────────────────────────────────────────────────────
class ProductBase(BaseModel):
    sku: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=120)
    barcode: str | None = Field(default=None, max_length=64)
    category_id: str | None = None
    shelf_id: str | None = None
    unit_size: str | None = Field(default=None, max_length=64)
    price: float | None = Field(default=None, ge=0)
    expected_quantity: int | None = Field(default=None, ge=0)
    min_threshold: int | None = Field(default=None, ge=0)
    facing_capacity: int | None = Field(default=None, ge=0)
    detector_class_name: str | None = Field(default=None, max_length=80)
    quantity_source: Literal["camera_estimate", "manual", "pos", "erp", "import"] = "manual"
    reference_image_path: str | None = None
    is_active: bool = True
    notes: str | None = None


class ProductCreate(ProductBase):
    store_id: str | None = None


class ProductUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    brand: str | None = None
    barcode: str | None = None
    category_id: str | None = None
    shelf_id: str | None = None
    unit_size: str | None = None
    price: float | None = Field(default=None, ge=0)
    expected_quantity: int | None = Field(default=None, ge=0)
    min_threshold: int | None = Field(default=None, ge=0)
    facing_capacity: int | None = Field(default=None, ge=0)
    detector_class_name: str | None = None
    quantity_source: Literal["camera_estimate", "manual", "pos", "erp", "import"] | None = None
    reference_image_path: str | None = None
    is_active: bool | None = None
    notes: str | None = None


class ProductRead(ORMModel):
    id: str
    store_id: str | None
    category_id: str | None
    shelf_id: str | None
    sku: str
    name: str
    brand: str | None
    barcode: str | None
    unit_size: str | None
    price: float | None
    expected_quantity: int | None
    min_threshold: int | None
    quantity_source: str
    last_counted_at: datetime | None
    reference_image_path: str | None
    facing_capacity: int | None
    detector_class_name: str | None
    is_active: bool
    notes: str | None
    created_at: datetime
    updated_at: datetime


class ProductCategoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    code: str | None = Field(default=None, max_length=64)
    description: str | None = None
    parent_id: str | None = None


class ProductCategoryRead(ORMModel):
    id: str
    name: str
    code: str | None
    description: str | None
    parent_id: str | None
    created_at: datetime


class ShelfBase(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)
    aisle: str | None = None
    level: int | None = None
    capacity_units: int = Field(default=0, ge=0)
    camera_id: str | None = None
    zone_id: str | None = None
    polygon: list[list[float]] | None = None
    is_active: bool = True
    description: str | None = None


class ShelfCreate(ShelfBase):
    store_id: str | None = None


class ShelfUpdate(BaseModel):
    name: str | None = None
    aisle: str | None = None
    level: int | None = None
    capacity_units: int | None = Field(default=None, ge=0)
    camera_id: str | None = None
    zone_id: str | None = None
    polygon: list[list[float]] | None = None
    is_active: bool | None = None
    description: str | None = None


class ShelfRead(ORMModel):
    id: str
    store_id: str | None
    camera_id: str | None
    zone_id: str | None
    code: str
    name: str
    aisle: str | None
    level: int | None
    capacity_units: int
    polygon: list[list[float]] | None
    is_active: bool
    description: str | None
    created_at: datetime
    updated_at: datetime


class ShelfSectionCreate(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    position_index: int = Field(default=0, ge=0)
    capacity_units: int = Field(default=0, ge=0)
    bbox: list[float] | None = None


class ShelfSectionRead(ORMModel):
    id: str
    shelf_id: str
    code: str
    name: str
    position_index: int
    capacity_units: int
    bbox: list[float] | None
    is_active: bool


TokenResponse.model_rebuild()