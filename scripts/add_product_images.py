"""Attach reference images to catalog products so SKU recognition names them.

Recognition compares a detected crop against each product's reference image, so
a product becomes recognisable exactly when it has an image. This script is the
bulk path: put one photo per product in a folder, named after the product's SKU,
and every image is stored and embedded in one pass.

Usage::

    # images already exist as catalog SKUs (CHIPS-01.jpg -> SKU CHIPS-01)
    python scripts/add_product_images.py data/product_catalog/incoming

    # also create a product for any image whose SKU is not in the catalog
    python scripts/add_product_images.py photos --create --category "Snacks"

    # attach one image to one existing SKU
    python scripts/add_product_images.py --sku CHIPS-01 --image chip.jpg

Nothing here invents stock or accuracy. It only makes a product's *reference*
image available, which is what lets the recogniser put a name on a detection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import settings  # noqa: E402
from app.core.errors import ValidationError  # noqa: E402
from app.db.session import SessionLocal, create_all  # noqa: E402
from app.models import Product, ProductCategory, ProductSource, Store  # noqa: E402
from app.services import catalog_service  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def log(message: str) -> None:
    print(f"  {message}")


def _product_by_sku(session, sku: str) -> Product | None:
    return session.query(Product).filter(Product.sku == sku).first()


def _resolve_store(session, code: str | None) -> Store:
    store = (
        session.query(Store).filter(Store.code == code).first()
        if code
        else session.query(Store).order_by(Store.created_at).first()
    )
    if store is None:
        raise SystemExit(
            "No store found. Create one first, or pass --store-code after seeding."
        )
    return store


def _resolve_category(session, name: str | None) -> ProductCategory | None:
    if not name:
        return None
    category = session.query(ProductCategory).filter(ProductCategory.name == name).first()
    if category is None:
        category = ProductCategory(name=name, code=name[:8].upper().replace(" ", "-"))
        session.add(category)
        session.flush()
    return category


def _create_product(session, store, category, sku: str, expected: int, name: str | None) -> Product:
    product = Product(
        store_id=store.id,
        category_id=category.id if category else None,
        sku=sku,
        name=name or sku,
        expected_quantity=expected,
        min_threshold=max(1, expected // 4),
        facing_capacity=expected,
        quantity_source=ProductSource.MANUAL.value,
        is_active=True,
        notes="Reference image added by add_product_images.py",
    )
    session.add(product)
    session.flush()
    return product


def _attach(session, product: Product, image_path: Path) -> int:
    """Store the image on the product and compute its embedding."""
    raw = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    catalog_service.decode_image(raw)  # reject non-images with a clear message
    relative = catalog_service.save_reference_image(product.sku, raw, suffix)
    product.reference_image_path = relative
    session.flush()
    return catalog_service.store_embedding(session, product)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, help="folder of <SKU>.<ext> images")
    parser.add_argument("--sku", help="attach a single image to this SKU")
    parser.add_argument("--image", type=Path, help="single image to attach")
    parser.add_argument("--create", action="store_true", help="create missing products")
    parser.add_argument("--category", help="category name for created products")
    parser.add_argument("--expected", type=int, default=12, help="expected quantity for created products")
    parser.add_argument("--store-code", help="store code (defaults to the first store)")
    args = parser.parse_args(argv)

    if not args.sku and not args.folder:
        parser.error("provide a folder, or --sku with --image")
    if args.sku and not args.image:
        parser.error("--sku requires --image")

    settings.ensure_directories()
    create_all()

    session = SessionLocal()
    created = 0
    updated = 0
    skipped: list[str] = []
    failed: list[str] = []
    try:
        store = _resolve_store(session, args.store_code)
        category = _resolve_category(session, args.category)

        jobs: list[tuple[str, Path]] = []
        if args.sku and args.image:
            jobs.append((args.sku, args.image))
        elif args.folder:
            if not args.folder.exists():
                raise SystemExit(f"No such folder: {args.folder}")
            for path in sorted(args.folder.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    jobs.append((path.stem, path))
            if not jobs:
                raise SystemExit(f"No images found in {args.folder}")

        for sku, image_path in jobs:
            product = _product_by_sku(session, sku)
            if product is None:
                if not args.create:
                    skipped.append(f"{sku} (no such SKU — rerun with --create)")
                    continue
                product = _create_product(session, store, category, sku, args.expected, None)
                created += 1
            elif product.reference_image_path:
                # Re-attaching is allowed and is how a bad photo is replaced.
                pass
            try:
                dimension = _attach(session, product, image_path)
            except ValidationError as exc:
                failed.append(f"{sku}: {exc}")
                continue
            session.commit()
            updated += 1
            log(f"{sku:24} -> {dimension}-d embedding")

        from app.models import ProductEmbedding

        indexed = session.query(ProductEmbedding).count()
    finally:
        session.close()

    print()
    print("Reference images")
    print("-" * 72)
    log(f"products created : {created}")
    log(f"images attached  : {updated}")
    log(f"embeddings stored: {indexed}")
    log(
        "A running backend picks these up when a reference image is uploaded "
        "through the API; for files added by this script, restart the backend "
        "or call POST /api/v1/products/embeddings/rebuild to refresh it."
    )
    if skipped:
        log(f"skipped ({len(skipped)}): " + "; ".join(skipped))
    if failed:
        log(f"failed ({len(failed)}): " + "; ".join(failed))
    print("-" * 72)
    print(
        "\nNOTE: an image makes a product RECOGNISABLE (a name for a crop). It does\n"
        "not make it DETECTABLE — the detector must still produce a box. For\n"
        "packaged goods the COCO baseline is weak; supply a fine-tuned retail model\n"
        "via IRIS_DETECTOR_WEIGHTS for reliable boxes."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
