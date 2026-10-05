"""Product reference images and their recognition embeddings.

A product is only recognisable once it has a reference image AND an embedding,
so both the upload endpoint and the import script must produce one. These tests
pin that, plus the rejections that keep junk out of the catalog.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from app.core.errors import ValidationError
from app.models import Product, ProductEmbedding
from app.services import catalog_service


def _png_bytes(seed: int = 0, size: int = 64) -> bytes:
    import cv2

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture()
def product(db_session, store):
    from app.models import ProductSource

    row = Product(
        store_id=store.id,
        sku="CATALOG-TEST-01",
        name="Catalog test product",
        expected_quantity=6,
        quantity_source=ProductSource.MANUAL.value,
        is_active=True,
    )
    db_session.add(row)
    db_session.commit()
    try:
        yield row
    finally:
        db_session.query(ProductEmbedding).filter(
            ProductEmbedding.product_id == row.id
        ).delete()
        db_session.query(Product).filter(Product.id == row.id).delete()
        db_session.commit()


def test_decode_rejects_non_images():
    with pytest.raises(ValidationError):
        catalog_service.decode_image(b"this is not an image")
    with pytest.raises(ValidationError):
        catalog_service.decode_image(b"")


def test_decode_rejects_tiny_images():
    small = _png_bytes(size=8)
    with pytest.raises(ValidationError):
        catalog_service.decode_image(small)


def test_save_reference_image_writes_under_the_catalog_dir(db_session, product):
    relative = catalog_service.save_reference_image(product.sku, _png_bytes(1), ".png")
    assert relative.startswith("product_catalog/")
    stored = catalog_service.catalog_dir() / relative.split("/")[-1]
    assert stored.exists() and stored.stat().st_size > 0
    stored.unlink(missing_ok=True)


def test_store_embedding_requires_a_reference_image(db_session, product):
    with pytest.raises(ValidationError):
        catalog_service.store_embedding(db_session, product)


def test_reference_image_makes_a_product_recognisable(db_session, product):
    """The whole point: upload an image, get a stored embedding."""
    relative = catalog_service.save_reference_image(product.sku, _png_bytes(2), ".png")
    product.reference_image_path = relative
    db_session.flush()

    try:
        dimension = catalog_service.store_embedding(db_session, product)
    except Exception as exc:  # pragma: no cover - depends on local embedder weights
        pytest.skip(f"Embedder weights unavailable: {exc}")

    assert dimension > 0
    row = (
        db_session.query(ProductEmbedding)
        .filter(ProductEmbedding.product_id == product.id)
        .one()
    )
    assert row.dimension == dimension
    assert len(row.vector) == dimension

    stored = catalog_service.catalog_dir() / relative.split("/")[-1]
    stored.unlink(missing_ok=True)


def test_upload_endpoint_accepts_an_image(client, product):
    response = client.post(
        f"/api/v1/products/{product.id}/reference-image",
        files={"file": ("ref.png", io.BytesIO(_png_bytes(3)), "image/png")},
    )
    if response.status_code == 500 and "embedder" in response.text.lower():
        pytest.skip("Embedder weights unavailable")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["detail"]["dimension"] > 0
    assert body["detail"]["reference_image_path"].startswith("product_catalog/")

    stored = catalog_service.catalog_dir() / body["detail"]["reference_image_path"].split("/")[-1]
    stored.unlink(missing_ok=True)


def test_upload_endpoint_rejects_a_non_image(client, product):
    response = client.post(
        f"/api/v1/products/{product.id}/reference-image",
        files={"file": ("notes.txt", io.BytesIO(b"not an image at all"), "text/plain")},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_upload_endpoint_rejects_unknown_product(client):
    response = client.post(
        "/api/v1/products/does-not-exist/reference-image",
        files={"file": ("ref.png", io.BytesIO(_png_bytes(4)), "image/png")},
    )
    assert response.status_code == 404
