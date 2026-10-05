"""API v1 router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, cameras, inference, products, shelves, streams, stores
from app.api.v1 import operations as ops

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(stores.router)
api_router.include_router(cameras.router)
api_router.include_router(shelves.router)
api_router.include_router(shelves.planograms_router)
api_router.include_router(products.router)
api_router.include_router(products.categories_router)
api_router.include_router(ops.inventory_router)
api_router.include_router(ops.customers_router)
api_router.include_router(ops.queues_router)
api_router.include_router(ops.alerts_router)
api_router.include_router(ops.events_router)
api_router.include_router(ops.analytics_router)
api_router.include_router(ops.reports_router)
api_router.include_router(ops.models_router)
api_router.include_router(ops.detections_router)
api_router.include_router(ops.system_router)
api_router.include_router(inference.router)
# Video endpoints. These must be mounted or the Live page has nothing to
# display: /streams/{camera_id}/mjpeg and /snapshot.jpg are the only paths that
# ever put camera pixels on the wire.
api_router.include_router(streams.router)

__all__ = ["api_router"]


#: Documentation-friendly map of the resource prefixes the API exposes.
RESOURCE_PREFIXES = (
    "/auth",
    "/stores",
    "/cameras",
    "/cameras/{camera_id}/zones",
    "/cameras/{camera_id}/health",
    "/streams",
    "/shelves",
    "/planograms",
    "/products",
    "/product-categories",
    "/inventory",
    "/customers",
    "/queues",
    "/alerts",
    "/events",
    "/analytics",
    "/reports",
    "/models",
    "/detections",
    "/inference",
    "/system",
)