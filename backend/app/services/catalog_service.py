"""Product reference images and their recognition embeddings.

SKU recognition works by comparing a detected crop with the reference image of
each catalog product, so a product is only recognisable once it has a reference
image AND a stored embedding. This module is the one place that writes either,
used by both the API and the import script so they cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import ValidationError
from app.core.logging import get_logger

logger = get_logger("app")

#: Reference images live under the data dir here; the path stored on the product
#: is always relative to the data dir (never absolute), like every other path.
CATALOG_SUBDIR = "product_catalog"

ALLOWED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def catalog_dir() -> Path:
    directory = settings.resolve_path(settings.data_dir) / CATALOG_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def decode_image(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image, or raise a clear error."""
    if not raw:
        raise ValidationError("The uploaded file is empty.")
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValidationError("The uploaded file is not a decodable image.")
    if min(image.shape[:2]) < 24:
        raise ValidationError(
            "The image is too small to use as a reference (needs at least 24x24)."
        )
    return image


def save_reference_image(sku: str, raw: bytes, suffix: str, *, overwrite: bool = True) -> str:
    """Persist reference-image bytes and return the data-dir-relative path."""
    suffix = suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValidationError(
            f"Unsupported image type {suffix!r}. Use one of: {', '.join(ALLOWED_SUFFIXES)}."
        )
    safe_sku = "".join(ch for ch in sku if ch.isalnum() or ch in "-_.") or "product"
    target = catalog_dir() / f"{safe_sku}{suffix}"
    if target.exists() and not overwrite:
        raise ValidationError(f"A reference image for {sku} already exists.")
    target.write_bytes(raw)
    logger.info("Reference image stored", extra={"sku": safe_sku, "bytes": len(raw)})
    return f"{CATALOG_SUBDIR}/{target.name}"


def store_embedding(session: Any, product: Any) -> int:
    """Compute and store the product's embedding from its reference image.

    Returns the embedding dimension. Raises ``ValidationError`` when the
    reference image is missing or undecodable, so the caller can report why the
    SKU is not recognisable instead of failing silently.
    """
    from app.models import ProductEmbedding
    from app.vision.models_registry import registry

    if not product.reference_image_path:
        raise ValidationError(f"Product {product.sku} has no reference image yet.")

    root = settings.resolve_path(settings.data_dir).resolve()
    path = (root / product.reference_image_path).resolve()
    if root not in path.parents or not path.exists():
        raise ValidationError(f"Reference image for {product.sku} is missing on disk.")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError(f"Reference image for {product.sku} could not be decoded.")

    embedder = registry.get("embedder")
    vector = embedder.embed(image)
    if vector is None:
        raise ValidationError("The embedder did not return a vector for this image.")

    model_key = embedder.model_key
    payload = [float(v) for v in vector]
    existing = (
        session.query(ProductEmbedding)
        .filter(
            ProductEmbedding.product_id == product.id,
            ProductEmbedding.model_id == model_key,
        )
        .first()
    )
    if existing is None:
        session.add(
            ProductEmbedding(
                product_id=product.id,
                model_id=model_key,
                dimension=len(payload),
                vector=payload,
                source_image=str(product.reference_image_path),
            )
        )
    else:
        existing.vector = payload
        existing.dimension = len(payload)
    session.flush()
    return len(payload)


def reload_recognition_index() -> int:
    """Reload the live recognition catalog; never let this fail a request."""
    try:
        from app.services.camera_manager import camera_manager

        return camera_manager.reload_recognizer_catalog()
    except Exception as exc:  # pragma: no cover - recognition must not block writes
        logger.warning("Could not refresh recognition catalog", exc_info=exc)
        return 0
