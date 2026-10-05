"""Application configuration.

All runtime settings come from environment variables prefixed with ``IRIS_``
or from an ``.env`` file.  Business thresholds defined here are *defaults*
only — the live values live in the ``settings`` table and can be changed from
the dashboard without touching code or restarting the service.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="IRIS_",
        env_file=(".env", str(REPO_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    # ── Runtime ─────────────────────────────────────────────────────────────
    env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    app_name: str = "IRIS"
    app_version: str = "1.0.0"

    # ── API server ──────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # ── Database ────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/iris.db"
    db_echo: bool = False
    db_pool_size: int = 5

    # ── Paths ───────────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")
    log_dir: Path = Path("./logs")
    config_dir: Path = Path("./configs")

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5

    # ── Security ────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_TO_A_32_BYTE_RANDOM_SECRET_STRING__"
    jwt_secret: str = "CHANGE_ME_TO_A_LONG_RANDOM_SECRET__"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720
    password_min_length: int = 10
    rate_limit_per_minute: int = 240
    auth_enabled: bool = False
    default_admin_username: str = "admin"
    default_admin_password: str = "CHANGE_ME_BEFORE_ANY_REAL_USE"

    # ── AI / inference ──────────────────────────────────────────────────────
    device: str = "auto"
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.45
    #: Weights for the general detector that does ALL detection (persons and
    #: products) in one pass. yolo11n is ~2.8x faster than yolo11s on CPU, which
    #: is what keeps the live boxes updating on this hardware; set
    #: IRIS_DETECTOR_WEIGHTS=yolo11s.pt for the most accurate available model at
    #: the cost of roughly a third of the frame rate.
    detector_weights: str = "yolo11n.pt"
    person_weights: str = "yolo11s.pt"
    #: How many cameras may run model inference at the same time. Measured on
    #: this 8-thread box: 3 cameras x 2 intra-op threads beat every other
    #: combination, so 3 concurrent cameras each using 2 threads is the default.
    inference_max_concurrency: int = 3
    torch_num_threads: int = 4
    #: Optional second detector for multipacks. The Retail-product-detection
    #: repository that defines the combination rule keeps its trained weights
    #: on an external drive, so this path is usually unset: the detector then
    #: runs general-only and says so, rather than pretending the stage ran.
    retail_multipack_weights: str = ""
    detector_imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    track_max_ttl_seconds: float = 30.0
    embedding_enabled: bool = True
    embedding_min_similarity: float = 0.35
    min_detection_confidence: float = 0.20

    # ── Camera pipeline ─────────────────────────────────────────────────────
    frame_queue_size: int = 4
    frame_skip: int = 1
    default_capture_fps: float = 25.0
    rtsp_connect_timeout: float = 10.0
    rtsp_reconnect_delay: float = 5.0
    rtsp_reconnect_max_delay: float = 120.0
    #: Port the Android "IP Webcam" app serves MJPEG on when the operator types
    #: a bare address like ``192.168.1.42`` without an explicit port.
    ip_webcam_default_port: int = 8080
    frozen_frame_seconds: float = 15.0
    camera_stale_seconds: float = 60.0
    mjpeg_quality: int = 70
    mjpeg_max_clients: int = 16
    #: Frames wider than this are downscaled at capture. A 2730x1440 frame
    #: cannot be JPEG-encoded fast enough on CPU to feel live, and the detector
    #: resizes to 640 anyway. Set to 0 to disable downscaling.
    frame_max_width: int = 960
    #: Confidence floor for drawing/among live detections, applied on top of the
    #: per-camera threshold. Raising it suppresses weak, spurious person boxes
    #: that make a scene look busier than it is.
    live_detection_confidence: float = 0.35
    max_upload_mb: int = 512
    allowed_video_extensions: str = ".mp4,.avi,.mov,.mkv,.webm,.m4v"

    # ── Business rule DEFAULTS (overridable in the settings table) ──────────
    default_low_stock_threshold: float = 20.0
    default_out_of_stock_threshold: float = 5.0
    default_queue_busy: int = 4
    default_queue_critical: int = 7
    alert_cooldown_seconds: int = 300
    alert_required_observations: int = 3
    alert_min_confidence: float = 0.80

    # ── Privacy / retention ─────────────────────────────────────────────────
    evidence_enabled: bool = True
    evidence_retention_days: int = 7
    detection_retention_days: int = 30
    alert_retention_days: int = 365
    anonymize_faces: bool = True

    # ── Optional external services (unused in the local prototype) ──────────
    redis_url: str = ""
    mqtt_url: str = ""

    # ── Demo mode ───────────────────────────────────────────────────────────
    demo_mode: bool = False

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("allowed_video_extensions")
    @classmethod
    def _normalise_extensions(cls, value: str) -> str:
        items = [v.strip().lower() if v.strip().startswith(".") else f".{v.strip().lower()}"
                 for v in value.split(",") if v.strip()]
        return ",".join(items)

    # ── Derived helpers ─────────────────────────────────────────────────────
    @property
    def upload_dir(self) -> Path:
        return self.resolve_path(self.data_dir) / "uploads"

    @property
    def evidence_dir(self) -> Path:
        return self.resolve_path(self.data_dir) / "evidence"

    @property
    def export_dir(self) -> Path:
        return self.resolve_path(self.data_dir) / "exports"

    @property
    def resolved_model_dir(self) -> Path:
        return self.resolve_path(self.model_dir)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def video_extensions(self) -> tuple[str, ...]:
        return tuple(e for e in self.allowed_video_extensions.split(",") if e)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolve_path(self, path: Path | str) -> Path:
        """Resolve a possibly-relative configured path against the repo root."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (REPO_ROOT / candidate).resolve()

    def ensure_directories(self) -> None:
        """Create the writable directories IRIS needs at runtime."""
        for directory in (
            self.resolve_path(self.data_dir),
            self.upload_dir,
            self.evidence_dir,
            self.export_dir,
            self.resolved_model_dir,
            self.resolve_path(self.log_dir),
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def warn_if_insecure(self) -> list[str]:
        """Return a list of insecure-configuration warnings (used at startup)."""
        warnings: list[str] = []
        if self.is_production:
            if self.auth_enabled is False:
                warnings.append("IRIS_AUTH_ENABLED is false in production.")
            if "CHANGE_ME" in self.jwt_secret:
                warnings.append("IRIS_JWT_SECRET is still the placeholder value.")
            if "CHANGE_ME" in self.secret_key:
                warnings.append("IRIS_SECRET_KEY is still the placeholder value.")
            if self.debug:
                warnings.append("IRIS_DEBUG is true in production.")
            if self.database_url.startswith("sqlite"):
                warnings.append("SQLite is not recommended for production deployments.")
        return warnings


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


settings = get_settings()