"""Default zone / shelf provisioning.

A camera with no zones produces no analytics at all. These tests pin the
behaviour that makes a freshly-added camera useful without inventing data:
default, clearly-labelled geometry, created once and never overwritten.
"""

from __future__ import annotations

from app.models import Camera, CameraRole, CameraStatus, CameraZone, Shelf, ZoneType
from app.services.provisioning import _next_shelf_code, provision_camera_defaults


def _camera(db_session, store, role: str, name: str) -> Camera:
    camera = Camera(
        store_id=store.id,
        name=name,
        role=role,
        source_type="video_file",
        capture_fps=5.0,
        is_enabled=False,
        status=CameraStatus.OFFLINE.value,
        notes="provisioning test",
    )
    db_session.add(camera)
    db_session.flush()
    return camera


def test_shelf_camera_gets_a_zone_and_a_shelf(db_session, store):
    camera = _camera(db_session, store, CameraRole.SHELF.value, "Provision Shelf Cam")
    report = provision_camera_defaults(db_session, camera)
    assert report["zones"] == 1
    assert report["shelves"] == 1

    zone = (
        db_session.query(CameraZone)
        .filter(CameraZone.camera_id == camera.id, CameraZone.zone_type == ZoneType.SHELF.value)
        .one()
    )
    assert zone.polygon, "a shelf zone must carry a polygon or nothing is measured"
    assert "default" in (zone.description or "").lower()

    shelf = db_session.query(Shelf).filter(Shelf.camera_id == camera.id).one()
    assert shelf.zone_id == zone.id
    assert shelf.capacity_units > 0


def test_entrance_camera_gets_a_line_and_a_queue_region(db_session, store):
    camera = _camera(db_session, store, CameraRole.ENTRANCE.value, "Provision Entrance Cam")
    provision_camera_defaults(db_session, camera)
    types = {
        z.zone_type
        for z in db_session.query(CameraZone).filter(CameraZone.camera_id == camera.id)
    }
    assert ZoneType.ENTRANCE_LINE.value in types
    assert ZoneType.QUEUE.value in types, "the Queues page needs a region to count"


def test_provisioning_is_idempotent_and_never_overwrites(db_session, store):
    camera = _camera(db_session, store, CameraRole.SHELF.value, "Provision Idempotent Cam")
    provision_camera_defaults(db_session, camera)

    # An operator adjusts the geometry by hand…
    zone = (
        db_session.query(CameraZone)
        .filter(CameraZone.camera_id == camera.id)
        .first()
    )
    zone.polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    zone.description = "hand drawn"
    db_session.flush()

    # …and a second provisioning run leaves it alone.
    second = provision_camera_defaults(db_session, camera)
    assert second["zones"] == 0
    assert second["shelves"] == 0
    assert zone.description == "hand drawn"
    assert len(zone.polygon) == 4


def test_shelf_code_never_collides_after_a_deletion(db_session, store):
    """Regression: 'count + 1' codes collided once any shelf was deleted.

    With AUTO-01..AUTO-03 present and AUTO-01 removed, count+1 = AUTO-03 hit
    the (store_id, code) unique index and registering a camera returned a 500.
    """
    import pytest

    from app.core.errors import ValidationError

    db_session.add(Shelf(store_id=store.id, code="AUTO-02", name="s2"))
    db_session.flush()
    assert _next_shelf_code(db_session, store.id) == "AUTO-01"

    db_session.add(Shelf(store_id=store.id, code="AUTO-01", name="s1"))
    db_session.flush()
    assert _next_shelf_code(db_session, store.id) == "AUTO-03"

    # All sequential codes taken -> camera-derived unique fallback.
    # AUTO-01 and AUTO-02 already exist from the steps above.
    for number in range(3, 100):
        db_session.add(Shelf(store_id=store.id, code=f"AUTO-{number:02d}", name=f"s{number}"))
    db_session.flush()
    assert _next_shelf_code(db_session, store.id, "abcdef12xyz") == "AUTO-ABCDEF12"

    # Even the camera-derived code taken -> nothing safe left to hand out.
    db_session.add(Shelf(store_id=store.id, code="AUTO-ABCDEF12", name="s-cam"))
    db_session.flush()
    with pytest.raises(ValidationError):
        _next_shelf_code(db_session, store.id, "abcdef12xyz")


def test_two_shelf_cameras_get_distinct_shelf_codes(db_session, store):
    first = _camera(db_session, store, CameraRole.SHELF.value, "Provision Cam A")
    second = _camera(db_session, store, CameraRole.SHELF.value, "Provision Cam B")
    provision_camera_defaults(db_session, first)
    provision_camera_defaults(db_session, second)
    codes = [s.code for s in db_session.query(Shelf).filter(Shelf.store_id == store.id)]
    assert len(codes) == len(set(codes)) == 2
