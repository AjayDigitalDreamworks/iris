"""API and service integration tests.

These run against the real FastAPI application, the real ORM and the real
services over an isolated SQLite database. Nothing here is mocked; assertions
are made on the same JSON the dashboard consumes.
"""

from __future__ import annotations

import pytest

from app.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    AlertType,
    Camera,
    CameraRole,
    CameraStatus,
    CameraZone,
    Product,
    ProductCategory,
    Store,
    ZoneType,
)


# ── helpers ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def seeded_store(db_session):
    """A committed store + camera + zone + product, removed afterwards.

    Committed (not rolled back) because the API under test opens its own
    sessions, so the fixture must delete what it created to stay isolated.
    """
    store = Store(code="TEST-STORE", name="Test store", timezone="UTC")
    db_session.add(store)
    category = ProductCategory(name="Grocery")
    db_session.add(category)
    db_session.flush()
    camera = Camera(
        store_id=store.id,
        name="API Test Camera",
        role=CameraRole.SHELF.value,
        source_type="synthetic",
        capture_fps=2.0,
        is_enabled=False,
        status=CameraStatus.OFFLINE.value,
        notes="test fixture",
    )
    db_session.add(camera)
    db_session.flush()
    zone = CameraZone(
        camera_id=camera.id,
        name="Test shelf zone",
        zone_type=ZoneType.SHELF.value,
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        is_active=True,
    )
    db_session.add(zone)
    product = Product(
        store_id=store.id,
        category_id=category.id,
        sku="TEST-SKU-1",
        name="Test product",
        expected_quantity=10,
        facing_capacity=24,
    )
    db_session.add(product)
    db_session.commit()
    ids = {
        "store": store.id,
        "camera": camera.id,
        "zone": zone.id,
        "product": product.id,
    }
    try:
        yield ids
    finally:
        db_session.query(Product).filter(Product.id == product.id).delete()
        db_session.query(CameraZone).filter(CameraZone.id == zone.id).delete()
        db_session.query(Camera).filter(Camera.id == camera.id).delete()
        db_session.query(ProductCategory).filter(
            ProductCategory.id == category.id
        ).delete()
        db_session.query(Store).filter(Store.id == store.id).delete()
        db_session.commit()


# ── meta / health ────────────────────────────────────────────────────────────
class TestMetaEndpoints:
    def test_banner_lists_api_base(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert "IRIS" in body["service"]
        assert body["api"].startswith("/api/v1")

    def test_health_reports_components(self, client):
        body = client.get("/health").json()
        assert body["status"] in {"healthy", "degraded", "offline"}
        names = {c["name"] for c in body["components"]}
        assert {"backend", "database", "camera", "worker"} <= names

    def test_readiness_probe(self, client):
        body = client.get("/api/v1/system/ready").json()
        assert body["ready"] is True
        assert body["database"]["status"] == "healthy"

    def test_openapi_documents_every_router(self, client):
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        for prefix in (
            "/api/v1/cameras",
            "/api/v1/products",
            "/api/v1/alerts",
            "/api/v1/analytics",
            "/api/v1/auth",
            # Video endpoints are documented in RESOURCE_PREFIXES, so they
            # must actually be mounted: when the streams router was left out,
            # every documented video URL 404'd and the Live page showed nothing.
            "/api/v1/streams",
        ):
            assert any(p.startswith(prefix) for p in paths), prefix

    def test_stream_endpoints_are_reachable(self, client, seeded_store):
        """The three video paths must not 404 for a camera that exists."""
        camera_id = seeded_store["camera"]
        status = client.get(f"/api/v1/streams/{camera_id}/status")
        assert status.status_code == 200, status.text
        body = status.json()
        assert body["camera_id"] == camera_id
        # A camera that was never started has no frames: that is a 503 from the
        # snapshot endpoint, NOT a 404 (which would mean the route is missing).
        snapshot = client.get(f"/api/v1/streams/{camera_id}/snapshot.jpg")
        assert snapshot.status_code in (200, 503), snapshot.status_code
        # The MJPEG route is asserted through the spec rather than called: an
        # unbounded stream cannot be read to completion in a test, and its
        # absence is exactly what we are guarding against.
        spec = client.get("/openapi.json").json()
        mjpeg = spec["paths"]["/api/v1/streams/{camera_id}/mjpeg"]["get"]
        assert "multipart/x-mixed-replace" in mjpeg["responses"]["200"]["content"]


# ── cameras ──────────────────────────────────────────────────────────────────
class TestCamerasAPI:
    def test_list_and_create(self, client, store):
        assert client.get("/api/v1/cameras").status_code == 200
        response = client.post(
            "/api/v1/cameras",
            json={
                "store_id": store.id,
                "name": "Created via API",
                "role": CameraRole.ENTRANCE.value,
                "source_type": "synthetic",
                "capture_fps": 3.0,
                "is_enabled": False,
            },
        )
        assert response.status_code in (200, 201), response.text
        created = response.json()
        assert created["name"] == "Created via API"
        assert created["is_enabled"] is False

    def test_duplicate_name_is_rejected(self, client, store):
        payload = {
            "store_id": store.id,
            "name": "Duplicate Camera",
            "role": CameraRole.SHELF.value,
            "source_type": "synthetic",
            "capture_fps": 1.0,
        }
        first = client.post("/api/v1/cameras", json=payload)
        assert first.status_code in (200, 201), first.text
        second = client.post("/api/v1/cameras", json=payload)
        assert second.status_code >= 400
        assert "code" in second.json().get("error", {})

    def test_web_video_camera_accepts_a_youtube_link(self, client, store):
        """A YouTube link is a first-class camera source.

        The link is a credential-bearing string in the general case, so it is
        stored encrypted and only ever returned redacted.
        """
        link = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        response = client.post(
            "/api/v1/cameras",
            json={
                "store_id": store.id,
                "name": "YouTube Aisle Camera",
                "role": CameraRole.OTHER.value,
                "source_type": "url",
                "stream_url": link,
                "capture_fps": 5.0,
                "is_enabled": False,
            },
        )
        assert response.status_code in (200, 201), response.text
        created = response.json()
        assert created["source_type"] == "url"
        # Never echo the raw URL back: query values are masked, because signed
        # media links carry their credentials there.
        assert "stream_url" not in created or created["stream_url"] is None
        assert "dQw4w9WgXcQ" not in response.text
        assert created["has_credentials"] is True
        # The host stays visible so an operator can still tell cameras apart.
        assert "youtube.com" in created["stream_url_preview"]
        client.delete(f"/api/v1/cameras/{created['id']}")

    def test_web_video_camera_without_a_link_is_rejected(self, client, store):
        response = client.post(
            "/api/v1/cameras",
            json={
                "store_id": store.id,
                "name": "Missing Link Camera",
                "role": CameraRole.OTHER.value,
                "source_type": "url",
            },
        )
        assert response.status_code == 422
        assert "stream_url" in response.text

    def test_video_file_camera_without_a_path_is_rejected(self, client, store):
        response = client.post(
            "/api/v1/cameras",
            json={
                "store_id": store.id,
                "name": "Missing Path Camera",
                "role": CameraRole.OTHER.value,
                "source_type": "video_file",
            },
        )
        assert response.status_code == 422
        assert "file_path" in response.text

    def test_unknown_camera_returns_404_with_error_envelope(self, client):
        response = client.get("/api/v1/cameras/does-not-exist")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert "request_id" in error

    def test_validation_error_lists_field_problems(self, client):
        response = client.post("/api/v1/cameras", json={"name": ""})
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] in {"validation_error", "unprocessable_entity"}
        assert body["error"]["details"]["errors"]

    def test_camera_test_endpoint_reports_reachability(self, client):
        response = client.post(
            "/api/v1/cameras/test",
            json={"source_type": "synthetic"},
        )
        assert response.status_code in (200, 400, 422)


# ── zones ────────────────────────────────────────────────────────────────────
class TestZonesAPI:
    def test_zones_are_listed_and_patchable(self, client, seeded_store):
        camera_id = seeded_store["camera"]
        listed = client.get(f"/api/v1/cameras/{camera_id}/zones")
        assert listed.status_code == 200
        zones = listed.json()
        assert zones and zones[0]["zone_type"] == ZoneType.SHELF.value

        zone_id = zones[0]["id"]
        patched = client.patch(
            f"/api/v1/cameras/zones/{zone_id}", json={"is_active": False}
        )
        assert patched.status_code == 200
        assert patched.json()["is_active"] is False


# ── products ─────────────────────────────────────────────────────────────────
class TestProductsAPI:
    def test_products_list_and_detail(self, client, seeded_store):
        listed = client.get("/api/v1/products")
        assert listed.status_code == 200
        body = listed.json()
        assert set(body) >= {"items", "total", "limit", "offset"}
        skus = {p["sku"] for p in body["items"]}
        assert "TEST-SKU-1" in skus

        detail = client.get("/api/v1/products/by-sku/TEST-SKU-1")
        assert detail.status_code == 200
        assert detail.json()["name"] == "Test product"

        by_id = client.get(f"/api/v1/products/{detail.json()['id']}")
        assert by_id.status_code == 200
        assert by_id.json()["sku"] == "TEST-SKU-1"

    def test_unknown_product_is_404(self, client):
        response = client.get("/api/v1/products/by-sku/NOPE-1")
        assert response.status_code == 404


# ── alerts lifecycle ─────────────────────────────────────────────────────────
class TestAlertLifecycle:
    def _low_stock_outcome(self, db_session, camera_id: str):
        """A real :class:`RuleOutcome` as the real rules module produces it."""
        from app.services.alert_service import RuleOutcome
        from app.vision.rules import StockState, build_dedup_key

        return RuleOutcome(
            action="raise",
            alert_type=AlertType.LOW_STOCK.value,
            severity=AlertSeverity.WARNING.value,
            title="Low stock",
            message="SKU TEST-SKU-1 is below its configured threshold.",
            payload={"sku": "TEST-SKU-1", "stock_state": StockState.LOW_STOCK.value},
            dedup_key=build_dedup_key(
                AlertType.LOW_STOCK.value,
                camera_id=camera_id,
                shelf_id=None,
                sku="TEST-SKU-1",
                zone_id=None,
            ),
        )

    def test_apply_creates_then_resolves(self, db_session, seeded_store):
        from app.services.alert_service import alert_service

        camera_id = seeded_store["camera"]
        alert = alert_service.apply(
            self._low_stock_outcome(db_session, camera_id),
            camera_id=camera_id,
            shelf_id=None,
            zone_id=None,
            sku="TEST-SKU-1",
            store_id=seeded_store["store"],
        )
        assert alert is not None
        assert alert.status == AlertStatus.ACTIVE.value
        first_id = alert.id

        # Re-applying the same condition must refresh, not duplicate.
        again = alert_service.apply(
            self._low_stock_outcome(db_session, camera_id),
            camera_id=camera_id,
            shelf_id=None,
            sku="TEST-SKU-1",
            store_id=seeded_store["store"],
        )
        assert again is not None and again.id == first_id
        assert db_session.query(Alert).filter(Alert.id == first_id).count() == 1

        alert.status = AlertStatus.ACKNOWLEDGED.value
        db_session.commit()
        resolved = alert_service.resolve(first_id, username="tester", note="Restocked")
        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED.value
        assert resolved.resolved_at is not None

    def test_cooldown_suppresses_repeat_alerts(self, db_session, seeded_store):
        from app.services.alert_service import alert_service

        camera_id = seeded_store["camera"]
        alert_service.apply(
            self._low_stock_outcome(db_session, camera_id),
            camera_id=camera_id,
            shelf_id=None,
            sku="TEST-SKU-1",
            store_id=seeded_store["store"],
        )
        db_session.commit()
        before = db_session.query(Alert).count()
        refreshed = alert_service.apply(
            self._low_stock_outcome(db_session, camera_id),
            camera_id=camera_id,
            shelf_id=None,
            sku="TEST-SKU-1",
            store_id=seeded_store["store"],
        )
        # Inside the cooldown the service refreshes the existing alert; it must
        # NOT create a second one (that is the alert-storm failure mode).
        assert db_session.query(Alert).count() == before
        assert refreshed is not None
        assert refreshed.occurrence_count >= 2

    def test_alerts_api_lists_open_alerts(self, client, db_session, seeded_store):
        from app.services.alert_service import alert_service

        alert_service.apply(
            self._low_stock_outcome(db_session, seeded_store["camera"]),
            camera_id=seeded_store["camera"],
            shelf_id=None,
            sku="TEST-SKU-1",
            store_id=seeded_store["store"],
        )
        db_session.commit()
        response = client.get("/api/v1/alerts", params={"status": "active"})
        assert response.status_code == 200
        items = response.json()["items"]
        assert any(a["alert_type"] == AlertType.LOW_STOCK.value for a in items)

        active = client.get("/api/v1/alerts/active")
        assert active.status_code == 200
        assert any(a["alert_type"] == AlertType.LOW_STOCK.value for a in active.json())


# ── analytics ────────────────────────────────────────────────────────────────
class TestAnalyticsAPI:
    def test_footfall_endpoint_shape(self, client):
        response = client.get("/api/v1/customers/footfall", params={"range": "24h"})
        assert response.status_code == 200, response.text
        body = response.json()
        # Every aggregate must be present even with no recorded crossings, so
        # the dashboard never has to invent a zero.
        assert "total_entries" in body
        assert "total_exits" in body

    def test_occupancy_and_queue_endpoints(self, client):
        occupancy = client.get("/api/v1/customers/occupancy")
        assert occupancy.status_code == 200, occupancy.text
        assert "active_tracks" in occupancy.json()

        queues = client.get("/api/v1/queues")
        assert queues.status_code == 200, queues.text

    def test_system_stats_reports_workers(self, client):
        body = client.get("/api/v1/system/stats").json()
        assert "cameras" in body and "alerts" in body
        assert body["cameras"]["workers"] >= 0

    def test_config_endpoint_exposes_runtime_settings(self, client):
        body = client.get("/api/v1/system/config").json()
        assert "confidence_threshold" in body or "settings" in body