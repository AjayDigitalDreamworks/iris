"""Seed IRIS with the minimum real configuration.

By default this creates ONLY what a real deployment needs: business-rule
settings, user accounts, one neutral store and the model registry. No demo
cameras, no demo products. Register real sources (RTSP, phone IP Webcam, USB
webcam, video file, image or web link) afterwards from the Cameras page.

Pass ``--demo`` to additionally seed clearly-labelled DEMO data:

* the synthetic camera renders watermarked ``DEMO MODE`` frames;
* detections, estimates and events derived from them are stored with
  ``source='demo'`` / ``source='demo_camera_estimate'``;
* product reference images are **real crops of a real photograph**, so the
  recognition mechanism is exercised on genuine pixels — but the SKUs are
  invented, so no retail accuracy may be inferred from them.

Usage::

    python scripts/seed.py                 # real configuration only
    python scripts/seed.py --demo          # also create labelled demo data
    python scripts/seed.py --reset         # drop and recreate
    python scripts/seed.py --no-start      # do not start camera workers
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import settings  # noqa: E402
from app.core.security import encrypt_secret, hash_password, redact_url  # noqa: E402
from app.db.session import SessionLocal, create_all  # noqa: E402
from app.models import (  # noqa: E402
    Camera,
    CameraRole,
    CameraStatus,
    CameraZone,
    ModelVersion,
    Planogram,
    PlanogramItem,
    Product,
    ProductCategory,
    ProductEmbedding,
    ProductSource,
    Shelf,
    ShelfSection,
    Store,
    User,
    ZoneType,
    default_settings_payload,
    Setting,
)

DEMO_TAG = "DEMO DATA"

# Real photographs used to build product reference embeddings.
REAL_PHOTOS = (
    REPO_ROOT / "data" / "sample_images" / "wikimedia_grocery.jpg",
    REPO_ROOT / "data" / "sample_images" / "wikimedia_store_interior.jpg",
)

CATALOG_DIR = REPO_ROOT / "data" / "product_catalog"

SHELF_POLYGON = [[0.05, 0.10], [0.95, 0.10], [0.95, 0.72], [0.05, 0.72]]
QUEUE_POLYGON = [[0.10, 0.55], [0.90, 0.55], [0.90, 0.97], [0.10, 0.97]]
ENTRANCE_LINE = [[0.02, 0.85], [0.98, 0.85]]
#: The pan clip moves content horizontally, so its doorway is a VERTICAL line.
#: x=0.34 is where the tracked person's foot point actually traverses this clip
#: (measured with the real detector+ByteTrack, not guessed): the subject pans
#: from ~0.42 to ~0.27 of frame width, so a line at 0.50 would never be crossed.
VERTICAL_ENTRANCE_LINE = [[0.34, 0.00], [0.34, 1.00]]


def log(message: str) -> None:
    print(f"  {message}")


# ── Product reference images from real photographs ───────────────────────────
def build_reference_images(limit: int = 6) -> list[dict]:
    """Crop real objects out of real photos and use them as product references.

    This makes the recognizer operate on genuine image features.  The SKU names
    are invented demo labels; no claim about retail accuracy is implied.
    """
    import cv2

    from app.vision.models_registry import registry
    from app.vision.types import Frame

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        detector = registry.get("product_detector")
        detector.confidence = 0.18
        detector.warmup()
    except Exception as exc:
        log(f"detector unavailable ({exc}); reference images will be skipped")
        return []

    entries: list[dict] = []
    counter = 0
    for photo in REAL_PHOTOS:
        if not photo.exists():
            log(f"sample photo missing: {photo.name} (run scripts/fetch_models.py)")
            continue
        image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if image is None:
            continue
        detections, _ = detector.predict(Frame(image=image, index=1, timestamp=0.0))
        # A shopper is not a product. Keeping people out of the reference
        # catalog is what stops recognition from reporting stock that a person
        # is merely standing in front of.
        detections = [
            d for d in detections if d.class_name not in {"person", "bus", "truck", "car"}
        ]
        detections.sort(key=lambda d: -d.bbox.area)
        for detection in detections[:3]:
            crop = detector.crop(image, detection.bbox, pad=0.08)
            if crop.size == 0 or min(crop.shape[:2]) < 32:
                continue
            counter += 1
            if counter > limit:
                break
            sku = f"DEMO-{detection.class_name.upper()[:6]}-{counter:02d}"
            filename = f"{sku}.jpg"
            path = CATALOG_DIR / filename
            cv2.imwrite(str(path), crop)
            entries.append(
                {
                    "sku": sku,
                    "name": f"{DEMO_TAG} {detection.class_name.title()} unit #{counter}",
                    "category": detection.class_name,
                    "detector_class_name": detection.class_name,
                    "reference_image_path": f"product_catalog/{filename}",
                    "source_photo": photo.name,
                    "source_confidence": round(detection.confidence, 4),
                }
            )
    detector.confidence = settings.confidence_threshold
    return entries


# ── Seeding ──────────────────────────────────────────────────────────────────
def seed(*, reset: bool = False, build_embeddings: bool = True, demo: bool = False) -> dict:
    settings.ensure_directories()
    create_all()

    session = SessionLocal()
    created: dict[str, int] = {}
    try:
        if reset:
            log("--reset: removing existing configuration rows")
            for model in (ProductEmbedding, PlanogramItem, Planogram, ShelfSection, Shelf,
                          Product, ProductCategory, CameraZone, Camera, Store, User, ModelVersion):
                session.query(model).delete(synchronize_session=False)
            session.commit()

        created["settings"] = _seed_settings(session)

        store_code = "DEMO-STORE" if demo else "MAIN-STORE"
        store = session.query(Store).filter(Store.code == store_code).first()
        if store is None:
            if demo:
                store = Store(
                    code=store_code,
                    name=f"{DEMO_TAG} Store (synthetic configuration)",
                    location="Local development machine",
                    timezone="UTC",
                    is_active=True,
                )
            else:
                store = Store(
                    code=store_code,
                    name="My Store",
                    timezone="UTC",
                    is_active=True,
                )
            session.add(store)
            session.flush()
            log(f"created store {store.code}")
        created["store"] = 1

        created["users"] = _seed_users(session)

        if not demo:
            created["model_versions"] = _seed_model_registry(session)
            session.commit()
            log("demo data skipped (run with --demo to seed labelled demo cameras/products)")
            return created

        category = session.query(ProductCategory).filter(ProductCategory.name == "Demo goods").first()
        if category is None:
            category = ProductCategory(name="Demo goods", code="DEMO", description=f"{DEMO_TAG} category")
            session.add(category)
            session.flush()

        cameras = _seed_cameras(session, store)
        created["cameras"] = len(cameras)
        created["zones"] = _seed_zones(session, cameras)

        shelves = _seed_shelves(session, store, cameras)
        created["shelves"] = len(shelves)

        entries = build_reference_images()
        created["products"] = _seed_products(session, store, category, shelves, entries)
        created["planograms"] = _seed_planogram(session, shelves)

        created["model_versions"] = _seed_model_registry(session)
        session.commit()

        if build_embeddings and entries:
            created["embeddings"] = _build_embeddings(session)
            session.commit()
    finally:
        session.close()

    return created


def _seed_settings(session) -> int:
    existing = {row.key for row in session.query(Setting).all()}
    created = 0
    for payload in default_settings_payload():
        if payload["key"] not in existing:
            session.add(Setting(**payload))
            created += 1
    session.flush()
    log(f"settings: {created} business rule(s) written, {len(existing)} already present")
    return created


def _seed_users(session) -> int:
    created = 0
    accounts = (
        ("admin", "Store Administrator", "admin"),
        ("manager", "Operations Manager", "manager"),
        ("staff", "Floor Staff", "staff"),
    )
    for username, full_name, role in accounts:
        if session.query(User).filter(User.username == username).first():
            continue
        password = settings.default_admin_password if username == "admin" else "IrisDemo!2026"
        session.add(
            User(username=username, full_name=full_name, role=role, password_hash=hash_password(password))
        )
        created += 1
        log(f"user {username} ({role}) password: {password if username != 'admin' else 'see IRIS_DEFAULT_ADMIN_PASSWORD'}")
    session.flush()
    return created


def _seed_cameras(session, store: Store) -> dict[str, Camera]:
    definitions = (
        {
            "key": "shelf_demo",
            "name": f"{DEMO_TAG} Shelf Camera A (synthetic frames)",
            "role": CameraRole.SHELF.value,
            "source_type": "synthetic",
            "capture_fps": 4.0,
            "is_enabled": True,
            "notes": "DEMO MODE. Renders a synthetic shelf scene; frames are watermarked.",
        },
        {
            "key": "entrance_photo",
            "name": "Entrance Camera (real photograph, looped)",
            "role": CameraRole.ENTRANCE.value,
            "source_type": "image",
            "image": "wikimedia_store_interior.jpg",
            "capture_fps": 2.0,
            "is_enabled": True,
            "notes": "Real retail photograph re-emitted as a static scene for testing person detection.",
        },
        {
            "key": "grocery_shelf_photo",
            "name": "Grocery Shelf Camera (real photograph, looped)",
            "role": CameraRole.SHELF.value,
            "source_type": "image",
            "image": "wikimedia_grocery.jpg",
            "capture_fps": 2.0,
            "is_enabled": True,
            "notes": (
                "Real grocery photograph. The shelf zone sits on this camera so stock "
                "estimates come from genuinely detected products, not from a synthetic scene."
            ),
        },
        {
            "key": "checkout_demo",
            "name": f"{DEMO_TAG} Checkout Camera (synthetic frames)",
            "role": CameraRole.CHECKOUT.value,
            "source_type": "synthetic",
            "capture_fps": 4.0,
            "is_enabled": True,
            "notes": "DEMO MODE. Synthetic scene with a configurable queue zone.",
        },
        {
            "key": "pan_clip",
            "name": f"{DEMO_TAG} Pan Clip Camera (video file)",
            "role": CameraRole.ENTRANCE.value,
            "source_type": "video_file",
            "capture_fps": 5.0,
            "is_enabled": True,
            "notes": (
                "DEMO MODE. Loops a rendered clip of a real photograph so tracking "
                "and footfall analytics can be exercised. Not real footage."
            ),
        },
        {
            "key": "rtsp_template",
            "name": "RTSP camera (template — disabled)",
            "role": CameraRole.SHELF.value,
            "source_type": "rtsp",
            "capture_fps": 5.0,
            "is_enabled": False,
            "notes": (
                "Template only. Add real credentials through the API; the URL is stored "
                "encrypted and is never returned by the API."
            ),
        },
    )
    cameras: dict[str, Camera] = {}
    for definition in definitions:
        camera = (
            session.query(Camera)
            .filter(Camera.store_id == store.id, Camera.name == definition["name"])
            .first()
        )
        if camera is None:
            camera = Camera(
                store_id=store.id,
                name=definition["name"],
                role=definition["role"],
                source_type=definition["source_type"],
                capture_fps=definition["capture_fps"],
                frame_skip=1,
                is_enabled=definition["is_enabled"],
                status=CameraStatus.OFFLINE.value,
                notes=definition["notes"],
            )
            if definition["source_type"] == "image":
                photo = (
                    REPO_ROOT / "data" / "sample_images" / definition["image"]
                )
                camera.image_path = str(photo) if photo.exists() else None
            if definition["source_type"] == "rtsp":
                template_url = "rtsp://username:password@192.168.1.50:554/Streaming/Channels/101"
                camera.stream_url_encrypted = encrypt_secret(template_url)
                camera.stream_url_preview = redact_url(template_url)
            if definition["source_type"] == "video_file":
                clip = _demo_clip()
                if clip is None:
                    log(
                        f"  camera {definition['key']:16} skipped: no demo clip. "
                        "Run: python scripts/make_demo_video.py"
                    )
                    session.delete(camera)
                    session.flush()
                    continue
                camera.file_path = str(clip)
            session.add(camera)
            session.flush()
            log(f"camera {definition['key']:16} -> {camera.id} ({definition['source_type']}, "
                f"{'enabled' if definition['is_enabled'] else 'disabled'})")
        cameras[definition["key"]] = camera
    return cameras


def _demo_clip() -> Path | None:
    """Locate a rendered DEMO clip produced by scripts/make_demo_video.py."""
    clips = sorted((REPO_ROOT / "data" / "sample_videos").glob("pan*.mp4"))
    return clips[0] if clips else None


def _seed_zones(session, cameras: dict[str, Camera]) -> int:
    created = 0
    definitions = [
        (cameras["shelf_demo"], "Shelf A region", ZoneType.SHELF.value, SHELF_POLYGON, None, 1),
        (cameras["grocery_shelf_photo"], "Grocery shelf region", ZoneType.SHELF.value, SHELF_POLYGON, None, 1),
        (cameras["entrance_photo"], "Store entrance line", ZoneType.ENTRANCE_LINE.value, [], ENTRANCE_LINE, 1),
        (cameras["checkout_demo"], "Checkout queue zone", ZoneType.QUEUE.value, QUEUE_POLYGON, None, 2),
    ]
    if "pan_clip" in cameras:
        definitions.append(
            (cameras["pan_clip"], "Pan clip doorway line", ZoneType.ENTRANCE_LINE.value,
             [], VERTICAL_ENTRANCE_LINE, 1)
        )
    for camera, name, zone_type, polygon, line, stations in definitions:
        existing = (
            session.query(CameraZone)
            .filter(CameraZone.camera_id == camera.id, CameraZone.name == name)
            .first()
        )
        if existing is not None:
            # Re-seeding must also refresh geometry: a stale line/polygon
            # silently disables footfall and queue analytics.
            existing.zone_type = zone_type
            existing.polygon = polygon
            existing.line = line
            existing.service_stations = stations
            existing.is_active = True
            continue
        session.add(
            CameraZone(
                camera_id=camera.id,
                name=name,
                zone_type=zone_type,
                polygon=polygon,
                line=line,
                service_stations=stations,
                is_active=True,
                description=f"{DEMO_TAG} {zone_type} zone",
            )
        )
        created += 1
    session.flush()
    log(f"zones: {created} created")
    return created


def _seed_shelves(session, store: Store, cameras: dict[str, Camera]) -> list[Shelf]:
    shelf = session.query(Shelf).filter(Shelf.code == "A-01").first()
    # The shelf is watched by the REAL grocery photograph, not the synthetic
    # scene: stock estimates are only meaningful where the detector actually
    # finds products.
    shelf_camera = cameras["grocery_shelf_photo"]
    if shelf is None:
        zone = (
            session.query(CameraZone)
            .filter(CameraZone.camera_id == shelf_camera.id,
                    CameraZone.zone_type == ZoneType.SHELF.value)
            .first()
        )
        shelf = Shelf(
            store_id=store.id,
            camera_id=shelf_camera.id,
            zone_id=zone.id if zone else None,
            code="A-01",
            name=f"{DEMO_TAG} Shelf A-01",
            aisle="Aisle 1",
            level=1,
            capacity_units=48,
            polygon=SHELF_POLYGON,
            is_active=True,
            description="DEMO shelf with 48 configured facing positions.",
        )
        session.add(shelf)
        session.flush()
        for index in range(3):
            session.add(
                ShelfSection(
                    shelf_id=shelf.id,
                    code=f"A-01-{index + 1}",
                    name=f"Level {index + 1}",
                    position_index=index,
                    capacity_units=16,
                    bbox=[0.05, 0.10 + index * 0.206, 0.95, 0.10 + (index + 1) * 0.206],
                )
            )
        session.flush()
        log(f"shelf A-01 -> {shelf.id} (capacity 48, 3 sections)")
    else:
        # Re-point an existing shelf if the seed configuration changed.
        zone = (
            session.query(CameraZone)
            .filter(CameraZone.camera_id == shelf_camera.id,
                    CameraZone.zone_type == ZoneType.SHELF.value)
            .first()
        )
        shelf.camera_id = shelf_camera.id
        shelf.zone_id = zone.id if zone else shelf.zone_id
    return [shelf]


def _seed_products(
    session, store: Store, category: ProductCategory, shelves: list[Shelf], entries: list[dict]
) -> int:
    shelf = shelves[0]
    created = 0
    # Demo SKUs derived from real photo crops.
    for entry in entries:
        if session.query(Product).filter(Product.sku == entry["sku"]).first():
            continue
        session.add(
            Product(
                store_id=store.id,
                category_id=category.id,
                shelf_id=shelf.id,
                sku=entry["sku"],
                name=entry["name"],
                brand="Demo Brand",
                detector_class_name=entry["detector_class_name"],
                reference_image_path=entry["reference_image_path"],
                expected_quantity=24,
                min_threshold=8,
                facing_capacity=12,
                quantity_source=ProductSource.MANUAL.value,
                last_counted_at=datetime.now(tz=timezone.utc),
                last_counted_by="seed",
                is_active=True,
                notes=(
                    f"{DEMO_TAG}. Reference image cropped from {entry['source_photo']} "
                    f"(detector confidence {entry['source_confidence']}). "
                    "SKU name is invented; no retail accuracy is implied."
                ),
            )
        )
        created += 1
    # A SKU with no reference image: recognition must report 'unknown' for it.
    if not session.query(Product).filter(Product.sku == "DEMO-UNLABELLED-01").first():
        session.add(
            Product(
                store_id=store.id,
                shelf_id=shelf.id,
                sku="DEMO-UNLABELLED-01",
                name=f"{DEMO_TAG} SKU without a reference image",
                expected_quantity=12,
                min_threshold=4,
                facing_capacity=12,
                quantity_source=ProductSource.MANUAL.value,
                is_active=True,
                notes=(
                    f"{DEMO_TAG}. Deliberately has no reference image so the recognizer must "
                    "return the 'unknown' state rather than invent a SKU."
                ),
            )
        )
        created += 1
    session.flush()
    log(f"products: {created} created")
    return created


def _seed_planogram(session, shelves: list[Shelf]) -> int:
    shelf = shelves[0]
    existing = session.query(Planogram).filter(Planogram.shelf_id == shelf.id).first()
    if existing is not None:
        return 0
    sections = (
        session.query(ShelfSection)
        .filter(ShelfSection.shelf_id == shelf.id)
        .order_by(ShelfSection.position_index.asc())
        .all()
    )
    planogram = Planogram(
        shelf_id=shelf.id,
        version=1,
        name=f"{DEMO_TAG} planogram v1",
        status="active",
        created_by="seed",
        activated_at=datetime.now(tz=timezone.utc),
    )
    session.add(planogram)
    session.flush()
    products = (
        session.query(Product)
        .filter(Product.shelf_id == shelf.id, Product.reference_image_path.isnot(None))
        .order_by(Product.sku.asc())
        .all()
    )
    for index, product in enumerate(products):
        section = sections[index % len(sections)] if sections else None
        session.add(
            PlanogramItem(
                planogram_id=planogram.id,
                product_id=product.id,
                section_id=section.id if section else None,
                position_index=index,
                expected_facings=max(1, (product.facing_capacity or 12) // 4),
            )
        )
    session.flush()
    log(f"planogram v1 -> {planogram.id} with {len(products)} expected placement(s)")
    return 1


def _seed_model_registry(session) -> int:
    from app.vision.models_registry import detect_hardware, registry

    hardware = detect_hardware()
    created = 0
    for spec in registry.specs():
        weights = spec.resolve_weights()
        existing = (
            session.query(ModelVersion)
            .filter(ModelVersion.name == spec.name, ModelVersion.version == spec.version)
            .first()
        )
        if existing is not None:
            continue
        session.add(
            ModelVersion(
                name=spec.name,
                task=spec.task,
                version=spec.version,
                status=spec.status,
                weights_path=str(weights) if weights else f"(no weights required: {spec.provider})",
                weights_sha256=registry.checksum(weights) if weights else None,
                source=spec.source,
                provider=spec.provider,
                input_size=spec.input_size or None,
                labels=list(spec.classes) or None,
                confidence_threshold=spec.confidence_threshold or None,
                device=hardware.device,
                metrics={},
                notes=f"{spec.notes} {spec.validation_note}".strip(),
                is_active=True,
            )
        )
        created += 1
    session.flush()
    log(f"model registry: {created} entr(ies) written (device={hardware.device})")
    return created


def _build_embeddings(session) -> int:
    """Compute and store the real product embeddings used by SKU recognition."""
    try:
        from app.vision.embedder import build_entry
        from app.vision.models_registry import registry
    except Exception as exc:  # pragma: no cover
        log(f"embedder unavailable: {exc}")
        return 0

    try:
        embedder = registry.get("embedder")
    except Exception as exc:
        log(f"embedder weights missing ({exc}); run scripts/fetch_models.py")
        return 0

    import cv2

    created = 0
    products = session.query(Product).filter(Product.reference_image_path.isnot(None)).all()
    for product in products:
        path = (settings.resolve_path(settings.data_dir) / product.reference_image_path).resolve()
        root = settings.resolve_path(settings.data_dir).resolve()
        if root not in path.parents or not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        vector = embedder.embed(image)
        if vector is None:
            continue
        entry = build_entry(product.id, product.sku, product.name, vector, embedder.model_key)
        existing = (
            session.query(ProductEmbedding)
            .filter(
                ProductEmbedding.product_id == product.id,
                ProductEmbedding.model_id == embedder.model_key,
            )
            .first()
        )
        payload = [float(v) for v in entry.vector]
        if existing is None:
            session.add(
                ProductEmbedding(
                    product_id=product.id,
                    model_id=embedder.model_key,
                    dimension=len(payload),
                    vector=payload,
                    source_image=product.reference_image_path,
                )
            )
        else:
            existing.vector = payload
            existing.dimension = len(payload)
        created += 1
    session.flush()
    log(f"embeddings: {created} product vector(s) computed and stored")
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed IRIS with the minimum real configuration.")
    parser.add_argument("--reset", action="store_true", help="delete existing configuration first")
    parser.add_argument("--demo", action="store_true", help="also seed clearly-labelled DEMO cameras and products")
    parser.add_argument("--no-embeddings", action="store_true", help="skip embedding computation")
    parser.add_argument("--start", action="store_true", help="start the camera workers afterwards")
    args = parser.parse_args(argv)

    print("=" * 88)
    print("IRIS SEED — real configuration" + (f" + {DEMO_TAG}" if args.demo else ""))
    print("=" * 88)
    created = seed(
        reset=args.reset,
        build_embeddings=not args.no_embeddings,
        demo=args.demo,
    )
    print()
    print("Created")
    print("-" * 88)
    for key, value in created.items():
        print(f"  {key:16} {value}")
    print("-" * 88)

    if args.start:
        from app.services.camera_manager import camera_manager

        started = camera_manager.start_all()
        print(f"Started {started} camera worker(s). Press Ctrl+C to stop.")
        try:
            import time

            while True:
                time.sleep(5)
                for camera_id, worker in camera_manager.workers.items():
                    health = worker.health.as_dict()
                    print(
                        f"  {worker.health.name:45} {health['status']:10} "
                        f"fps={health['inference_fps']:5.2f} lat={health['inference_latency_ms']}ms"
                    )
        except KeyboardInterrupt:
            camera_manager.stop_all()
            print("\nStopped.")

    print()
    print("Run the backend:  cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())