"""Product catalog and category endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from sqlalchemy import func

from app.api.deps import AdminUser, CurrentUser, DbSession, ManagerUser, PageParams, write_audit
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models import Product, ProductCategory, Shelf
from app.schemas import (
    ActionResponse,
    MessageResponse,
    Page,
    ProductCategoryCreate,
    ProductCategoryRead,
    ProductCreate,
    ProductRead,
    ProductUpdate,
)
from app.services.db_service import paginate

router = APIRouter(prefix="/products", tags=["products"])
categories_router = APIRouter(prefix="/product-categories", tags=["products"])
logger = get_logger("app")


# ── Categories ───────────────────────────────────────────────────────────────
@categories_router.get("", response_model=list[ProductCategoryRead], summary="List categories")
def list_categories(session: DbSession, _: CurrentUser) -> list[ProductCategoryRead]:
    return [
        ProductCategoryRead.model_validate(c)
        for c in session.query(ProductCategory).order_by(ProductCategory.name).all()
    ]


@categories_router.post(
    "", response_model=ProductCategoryRead, status_code=201, summary="Create a category (admin)"
)
def create_category(
    payload: ProductCategoryCreate, request: Request, session: DbSession, principal: AdminUser
) -> ProductCategoryRead:
    existing = (
        session.query(ProductCategory)
        .filter(ProductCategory.name == payload.name, ProductCategory.store_id.is_(None))
        .first()
    )
    if existing is not None:
        raise ConflictError(f"Category {payload.name!r} already exists.")
    category = ProductCategory(**payload.model_dump())
    session.add(category)
    session.flush()
    write_audit(session, principal, "category.created", entity_type="product_category",
                entity_id=category.id, request=request)
    return ProductCategoryRead.model_validate(category)


@categories_router.delete(
    "/{category_id}", response_model=MessageResponse, summary="Delete a category (admin)"
)
def delete_category(
    category_id: str, request: Request, session: DbSession, principal: AdminUser
) -> MessageResponse:
    category = session.get(ProductCategory, category_id)
    if category is None:
        raise NotFoundError(f"Category {category_id!r} does not exist.")
    session.delete(category)
    write_audit(session, principal, "category.deleted", entity_type="product_category",
                entity_id=category_id, request=request)
    return MessageResponse(message=f"Category {category_id} deleted.")


# ── Products ─────────────────────────────────────────────────────────────────
@router.get("", response_model=Page[ProductRead], summary="List products")
def list_products(
    session: DbSession,
    page: PageParams,
    _: CurrentUser,
    store_id: str | None = None,
    shelf_id: str | None = None,
    category_id: str | None = None,
    search: str | None = None,
    is_active: bool | None = None,
) -> Page[ProductRead]:
    query = session.query(Product)
    if store_id:
        query = query.filter(Product.store_id == store_id)
    if shelf_id:
        query = query.filter(Product.shelf_id == shelf_id)
    if category_id:
        query = query.filter(Product.category_id == category_id)
    if is_active is not None:
        query = query.filter(Product.is_active.is_(is_active))
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(
            func.lower(Product.name).like(pattern.lower())
            | func.lower(Product.sku).like(pattern.lower())
            | Product.barcode.ilike(pattern)
        )
    rows, total = paginate(query.order_by(Product.sku.asc()), page.limit, page.offset)
    return Page.build([ProductRead.model_validate(p) for p in rows], total, page.limit, page.offset)


@router.post("", response_model=ProductRead, status_code=201, summary="Create a product (manager)")
def create_product(
    payload: ProductCreate, request: Request, session: DbSession, principal: ManagerUser
) -> ProductRead:
    existing = session.query(Product).filter(Product.sku == payload.sku).first()
    if existing is not None:
        raise ConflictError(f"SKU {payload.sku!r} already exists.")
    data = payload.model_dump()
    if data.get("shelf_id") and session.get(Shelf, data["shelf_id"]) is None:
        raise NotFoundError(f"Shelf {data['shelf_id']!r} does not exist.")
    product = Product(**data)
    session.add(product)
    session.flush()
    write_audit(session, principal, "product.created", entity_type="product",
                entity_id=product.id, detail={"sku": product.sku}, request=request)
    refresh_recognizer()  # keep the recognition catalog in step
    return ProductRead.model_validate(product)


def refresh_recognizer() -> None:
    """Rebuild the embedding catalog so a new SKU can be recognised at once."""
    try:
        from app.services.camera_manager import camera_manager as manager

        manager.reload_recognizer_catalog()
    except Exception as exc:  # pragma: no cover - recognition must not block creation
        logger.warning("Could not refresh recognition catalog", exc_info=exc)


def camera_manager() -> None:
    """Backwards-compatible alias for :func:`refresh_recognizer`."""
    refresh_recognizer()


@router.get("/{product_id}", response_model=ProductRead, summary="Product detail")
def get_product(product_id: str, session: DbSession, _: CurrentUser) -> ProductRead:
    product = session.get(Product, product_id)
    if product is None:
        raise NotFoundError(f"Product {product_id!r} does not exist.")
    return ProductRead.model_validate(product)


@router.get("/by-sku/{sku}", response_model=ProductRead, summary="Product detail by SKU")
def get_by_sku(sku: str, session: DbSession, _: CurrentUser) -> ProductRead:
    product = session.query(Product).filter(Product.sku == sku).first()
    if product is None:
        raise NotFoundError(f"SKU {sku!r} is not in the catalog.")
    return ProductRead.model_validate(product)


@router.patch("/{product_id}", response_model=ProductRead, summary="Update a product (manager)")
def update_product(
    product_id: str, payload: ProductUpdate, request: Request,
    session: DbSession, principal: ManagerUser,
) -> ProductRead:
    product = session.get(Product, product_id)
    if product is None:
        raise NotFoundError(f"Product {product_id!r} does not exist.")
    for key, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(product, key, value)
    session.flush()
    write_audit(session, principal, "product.updated", entity_type="product",
                entity_id=product.id, detail={"fields": sorted(payload.model_dump(exclude_unset=True))},
                request=request)
    return ProductRead.model_validate(product)


@router.delete("/{product_id}", response_model=MessageResponse, summary="Delete a product (admin)")
def delete_product(
    product_id: str, request: Request, session: DbSession, principal: AdminUser
) -> MessageResponse:
    product = session.get(Product, product_id)
    if product is None:
        raise NotFoundError(f"Product {product_id!r} does not exist.")
    sku = product.sku
    session.delete(product)
    write_audit(session, principal, "product.deleted", entity_type="product",
                entity_id=product_id, detail={"sku": sku}, request=request)
    return MessageResponse(message=f"Product {sku} deleted.")


@router.post(
    "/embeddings/{product_id}/build", response_model=ActionResponse,
    summary="Compute and store the product's visual embedding (admin)",
)
def build_embedding(
    product_id: str, request: Request, session: DbSession, principal: AdminUser
) -> ActionResponse:
    """Embed the product's reference image so SKU recognition can match it."""
    import cv2
    import numpy as np

    from app.core.errors import ValidationError
    from app.models import ProductEmbedding
    from app.vision.models_registry import registry

    product = session.get(Product, product_id)
    if product is None:
        raise NotFoundError(f"Product {product_id!r} does not exist.")
    if not product.reference_image_path:
        raise ValidationError("This product has no reference_image_path configured.")

    from pathlib import Path

    from app.core.config import settings as cfg

    path = (cfg.resolve_path(cfg.data_dir) / product.reference_image_path).resolve()
    root = cfg.resolve_path(cfg.data_dir).resolve()
    if root not in path.parents or not path.exists():
        raise NotFoundError(f"Reference image not found for {product.sku}.")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError("The reference image could not be decoded.")

    embedder = registry.get("embedder")
    vector = embedder.embed(image)
    if vector is None:
        raise ValidationError("The embedder did not return a vector for this image.")

    model_key = embedder.model_key
    existing = (
        session.query(ProductEmbedding)
        .filter(
            ProductEmbedding.product_id == product.id,
            ProductEmbedding.model_id == model_key,
        )
        .first()
    )
    payload = [float(v) for v in vector]
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
    write_audit(session, principal, "product.embedding_built", entity_type="product",
                entity_id=product.id, detail={"model": model_key, "dim": len(payload)},
                request=request)
    camera_manager()
    _ = np
    return ActionResponse(
        action="build_embedding",
        message=f"Stored a {len(payload)}-d embedding for {product.sku}.",
        detail={"product_id": product.id, "model_id": model_key, "dimension": len(payload)},
    )


@router.post(
    "/{product_id}/reference-image",
    response_model=ActionResponse,
    summary="Upload a reference image for a product (manager)",
)
async def upload_reference_image(
    product_id: str,
    request: Request,
    session: DbSession,
    principal: ManagerUser,
    file: UploadFile = File(...),
) -> ActionResponse:
    """Attach a photo of a product so SKU recognition can name it.

    Recognition compares a detected crop against each product's reference image.
    So this single step — upload a photo — is what makes an item (chips,
    snacks, anything) recognisable by name: the image is stored and its
    embedding computed immediately, no restart required.
    """
    from app.core.config import settings
    from app.services import catalog_service

    product = session.get(Product, product_id)
    if product is None:
        raise NotFoundError(f"Product {product_id!r} does not exist.")

    raw = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(raw) > limit:
        raise ValidationError(
            f"Image is larger than the {settings.max_upload_mb} MB upload limit."
        )
    # Fail fast on a non-image before anything is written to disk.
    catalog_service.decode_image(raw)
    suffix = Path(file.filename or "").suffix or ".jpg"
    relative = catalog_service.save_reference_image(product.sku, raw, suffix)
    product.reference_image_path = relative
    session.flush()
    dimension = catalog_service.store_embedding(session, product)
    write_audit(
        session, principal, "product.reference_image_uploaded", entity_type="product",
        entity_id=product.id, detail={"sku": product.sku, "path": relative}, request=request,
    )
    catalog_service.reload_recognition_index()
    return ActionResponse(
        action="upload_reference_image",
        message=f"{product.sku} can now be recognised ({dimension}-d embedding).",
        detail={
            "product_id": product.id,
            "sku": product.sku,
            "reference_image_path": relative,
            "dimension": dimension,
        },
    )


@router.post(
    "/embeddings/rebuild", response_model=ActionResponse,
    summary="Rebuild the whole recognition index (admin)",
)
def rebuild_embeddings(request: Request, session: DbSession, principal: AdminUser) -> ActionResponse:
    count = camera_manager_count()
    write_audit(session, principal, "product.embeddings_rebuilt", entity_type="system",
                detail={"products": count}, request=request)
    return ActionResponse(
        action="rebuild_embeddings",
        message=f"Recognition index rebuilt from {count} stored embedding(s).",
        detail={"products": count},
    )


def camera_manager_count() -> int:
    from app.services.camera_manager import camera_manager as manager

    try:
        return manager.reload_recognizer_catalog()
    except Exception:  # pragma: no cover
        return 0


_ = Any