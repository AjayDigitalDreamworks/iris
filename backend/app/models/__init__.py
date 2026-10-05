"""ORM model registry.

Importing this package registers every mapped class on ``Base.metadata`` which
is what Alembic autogenerate and ``create_all`` rely on.
"""

from app.db.base import Base
from app.models.camera import (
    Camera,
    CameraHealthSample,
    CameraStatus,
    CameraZone,
    CameraRole,
    ProcessingJob,
    SourceType,
    ZoneType,
)
from app.models.catalog import (
    Planogram,
    PlanogramItem,
    Product,
    ProductCategory,
    ProductEmbedding,
    ProductSource,
    Shelf,
    ShelfSection,
)
from app.models.core import (
    AuditLog,
    Setting,
    Store,
    SystemSettingMeta,
    User,
    default_settings_payload,
)
from app.models.observations import (
    Detection,
    DetectionSource,
    FootfallEvent,
    InventorySnapshot,
    InventoryTransaction,
    QueueSnapshot,
    Track,
)
from app.models.ops import (
    ACTIVE_STATUSES,
    Alert,
    AlertSeverity,
    AlertStatus,
    AlertType,
    InferenceRun,
    ModelVersion,
    RetailEvent,
    SyncOutbox,
    SystemHealthSample,
)

__all__ = [
    "Base",
    # core
    "Store",
    "User",
    "AuditLog",
    "Setting",
    "SystemSettingMeta",
    "default_settings_payload",
    # camera
    "Camera",
    "CameraZone",
    "CameraHealthSample",
    "ProcessingJob",
    "SourceType",
    "CameraStatus",
    "CameraRole",
    "ZoneType",
    # catalog
    "ProductCategory",
    "Product",
    "ProductEmbedding",
    "ProductSource",
    "Shelf",
    "ShelfSection",
    "Planogram",
    "PlanogramItem",
    # observations
    "Detection",
    "DetectionSource",
    "Track",
    "FootfallEvent",
    "QueueSnapshot",
    "InventorySnapshot",
    "InventoryTransaction",
    # ops
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "AlertType",
    "ACTIVE_STATUSES",
    "RetailEvent",
    "ModelVersion",
    "InferenceRun",
    "SystemHealthSample",
    "SyncOutbox",
]