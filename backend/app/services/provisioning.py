"""Default zone / shelf provisioning.

A camera with no zones produces no analytics: no footfall, no queue length, no
stock. That is correct but useless — an operator who has just added a camera
sees empty pages everywhere. This module gives every camera a *sensible default*
geometry derived from its role so the analytics light up immediately.

The geometry is explicitly a STARTING POINT, not a measurement:

* it is written with a description that says so;
* it is never overwritten by a later run (idempotent), so once an operator
  adjusts a zone by hand, provisioning leaves it alone;
* nothing here invents stock counts — it only decides WHERE to look.
"""

from __future__ import annotations

import threading
from typing import Any

from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.models import Camera, CameraZone, Product, Shelf, Store, ZoneType

logger = get_logger("camera")

#: The label every provisioned zone carries, so the dashboard can say these are
#: defaults rather than surveyed geometry.
DEFAULT_NOTE = "Auto-configured default region — adjust to match the scene."

#: Normalised geometry, chosen to cover the middle of a typical store frame.
SHELF_POLYGON = [[0.05, 0.15], [0.95, 0.15], [0.95, 0.85], [0.05, 0.85]]
QUEUE_POLYGON = [[0.15, 0.45], [0.85, 0.45], [0.85, 0.95], [0.15, 0.95]]
OCCUPANCY_POLYGON = [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]
ENTRANCE_LINE = [[0.05, 0.6], [0.95, 0.6]]

DEFAULT_SHELF_CAPACITY = 48


def _role_zone(camera: Camera) -> tuple[str, str, list[list[float]], list[list[float]], int]:
    """Return (name, zone_type, polygon, line, service_stations) for a role."""
    role = (camera.role or "other").lower()
    if role == ZoneType.SHELF.value:
        return "Shelf region (auto)", ZoneType.SHELF.value, SHELF_POLYGON, [], 1
    if role == ZoneType.QUEUE.value or role == "checkout":
        return "Checkout queue (auto)", ZoneType.QUEUE.value, QUEUE_POLYGON, [], 2
    if role == ZoneType.ENTRANCE_LINE.value or role == "entrance":
        return "Entrance line (auto)", ZoneType.ENTRANCE_LINE.value, [], ENTRANCE_LINE, 1
    return "Occupancy region (auto)", ZoneType.ZONE.value, OCCUPANCY_POLYGON, [], 1


def provision_camera_defaults(session: Any, camera: Camera) -> dict[str, Any]:
    """Give a camera the zones (and shelf) its role needs, idempotently.

    Returns a small report: what was created vs already present. Commits
    nothing — the caller owns the transaction.
    """
    store_id = camera.store_id
    if not store_id:
        store = session.query(Store).first()
        if store is None:
            return {"zones": 0, "shelves": 0, "reason": "no store configured"}
        store_id = store.id
        camera.store_id = store.id

    existing_types = {
        z.zone_type for z in session.query(CameraZone).filter(CameraZone.camera_id == camera.id)
    }
    name, zone_type, polygon, line, stations = _role_zone(camera)

    zones_created = 0
    zone_id: str | None = None
    found = (
        session.query(CameraZone)
        .filter(CameraZone.camera_id == camera.id, CameraZone.zone_type == zone_type)
        .first()
    )
    if found is not None:
        zone_id = found.id
    else:
        zone = CameraZone(
            camera_id=camera.id,
            name=name,
            zone_type=zone_type,
            polygon=polygon,
            line=line,
            service_stations=stations,
            is_active=True,
            description=DEFAULT_NOTE,
        )
        session.add(zone)
        session.flush()
        zone_id = zone.id
        zones_created = 1
        logger.info(
            "Provisioned default zone",
            extra={"camera_id": camera.id, "zone_type": zone_type},
        )

    # Non-shelf cameras also get a queue region, so the Queues page reports
    # congestion rather than sitting empty. (A queue region simply counts the
    # tracked people standing in it — it is a measurement, not a claim that a
    # checkout line exists there.)
    if zone_type != ZoneType.SHELF.value:
        has_queue = (
            session.query(CameraZone)
            .filter(
                CameraZone.camera_id == camera.id,
                CameraZone.zone_type == ZoneType.QUEUE.value,
            )
            .first()
        )
        if has_queue is None:
            session.add(
                CameraZone(
                    camera_id=camera.id,
                    name="Queue region (auto)",
                    zone_type=ZoneType.QUEUE.value,
                    polygon=QUEUE_POLYGON,
                    line=[],
                    service_stations=2,
                    is_active=True,
                    description=DEFAULT_NOTE,
                )
            )
            session.flush()
            zones_created += 1

    # A shelf-role camera also needs a Shelf row: that is what the stock
    # estimator keys on, and it is what links catalog SKUs to a fixture.
    shelves_created = 0
    if zone_type == ZoneType.SHELF.value:
        shelf = session.query(Shelf).filter(Shelf.camera_id == camera.id).first()
        if shelf is None:
            # Code selection, insert and flush must happen under one lock:
            # two cameras registered at the same moment both read the same
            # shelf count and would otherwise insert the same code, which the
            # (store_id, code) unique index rejects — surfacing to the
            # operator as a 500 on "add camera".
            with _SHELF_CODE_LOCK:
                code = _next_shelf_code(session, store_id, camera.id)
                shelf = Shelf(
                    store_id=store_id,
                    camera_id=camera.id,
                    zone_id=zone_id,
                    code=code,
                    name=f"{camera.name} shelf (auto)",
                    capacity_units=DEFAULT_SHELF_CAPACITY,
                    polygon=polygon,
                    is_active=True,
                    description=DEFAULT_NOTE,
                )
                session.add(shelf)
                session.flush()
            shelves_created = 1
            logger.info("Provisioned default shelf", extra={"camera_id": camera.id, "code": code})

        # Attach any store SKU that is not yet placed on a shelf, so stock has
        # something to estimate against.
        unplaced = (
            session.query(Product)
            .filter(Product.store_id == store_id, Product.shelf_id.is_(None))
            .all()
        )
        for product in unplaced:
            product.shelf_id = shelf.id

    return {
        "zones": zones_created,
        "shelves": shelves_created,
        "zone_type": zone_type,
        "existing_zone_types": sorted(existing_types),
        "note": DEFAULT_NOTE,
    }


_SHELF_CODE_LOCK = threading.Lock()


def _next_shelf_code(session: Any, store_id: str, camera_id: str | None = None) -> str:
    """Return an unused shelf code for the store.

    The original ``count + 1`` scheme collided the moment any shelf was
    deleted (AUTO-01..03 present, AUTO-01 removed -> count+1 = AUTO-03 hit the
    unique index and "add camera" returned a 500). Scanning existing codes
    fixes that, but two cameras registered at the same instant can still both
    pick the same code: each database session cannot see the other's
    uncommitted insert. So when the camera id is known the code is derived
    from it — camera ids are unique per store and each camera gets at most one
    auto shelf, which makes the code race-free by construction. Sequential
    codes remain as a fallback for callers without a camera id.
    """
    existing = {
        row.code
        for row in session.query(Shelf).filter(Shelf.store_id == store_id).all()
    }
    candidates: list[str] = []
    if camera_id:
        candidates.append(f"AUTO-{camera_id[:8].upper()}")
    candidates.extend(f"AUTO-{number:02d}" for number in range(1, 100))
    for candidate in candidates:
        if candidate not in existing:
            return candidate
    raise ValidationError(
        "No free auto shelf code available for this store; create the shelf manually."
    )


def provision_all(session: Any) -> dict[str, Any]:
    """Provision every camera. Returns totals plus a per-camera breakdown."""
    created_zones = 0
    created_shelves = 0
    details: list[dict[str, Any]] = []
    for camera in session.query(Camera).all():
        report = provision_camera_defaults(session, camera)
        created_zones += report.get("zones", 0)
        created_shelves += report.get("shelves", 0)
        details.append({"camera_id": camera.id, "name": camera.name, **report})
    return {"zones": created_zones, "shelves": created_shelves, "cameras": details}
