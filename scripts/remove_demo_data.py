"""Remove all DEMO data and cameras so only real sources remain.

Deletes every camera and everything derived from them (zones, detections,
tracks, footfall, queue snapshots, inventory snapshots, alerts, retail events,
health samples, shelves, planograms, demo products and embeddings), then
renames the placeholder demo store to a neutral one. Users, business-rule
settings and the model registry are kept — the system needs them to run.

Usage::

    python scripts/remove_demo_data.py            # from the repository root

Stop the backend before running this so no worker re-creates state mid-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.db.session import SessionLocal, create_all  # noqa: E402


def log(message: str) -> None:
    print(f"  {message}")


#: Delete order matters: children before the rows they reference.
DELETE_ORDER = (
    "FootfallEvent",
    "QueueSnapshot",
    "Track",
    "Detection",
    "RetailEvent",
    "Alert",
    "InventorySnapshot",
    "InventoryTransaction",
    "CameraHealthSample",
    "ProcessingJob",
    "CameraZone",
    "PlanogramItem",
    "Planogram",
    "ShelfSection",
    "Shelf",
    "ProductEmbedding",
    "Product",
    "Camera",
)


def main() -> int:
    create_all()
    session = SessionLocal()
    try:
        import app.models as m

        print("=" * 88)
        print("IRIS — remove demo data and cameras")
        print("=" * 88)
        total = 0
        for name in DELETE_ORDER:
            model = getattr(m, name, None)
            if model is None:
                continue
            count = session.query(model).delete(synchronize_session=False)
            if count:
                log(f"{name:22} {count} row(s) deleted")
                total += count
        session.flush()

        # Demo product categories ("Demo goods") are only meaningful with demo
        # products; drop any category that no longer has products.
        for category in list(session.query(m.ProductCategory).all()):
            if session.query(m.Product).filter(m.Product.category_id == category.id).count() == 0:
                session.delete(category)
                log(f"ProductCategory      '{category.name}' deleted (no products left)")

        # Give the operator a neutral store to register real cameras against.
        store = session.query(m.Store).first()
        if store is not None and "demo" in (store.code or "").lower():
            store.code, store.name = "MAIN-STORE", "My Store"
            store.location = None
            log(f"Store                 renamed to '{store.code}' — '{store.name}'")
        session.commit()
        log(f"TOTAL                 {total} row(s) deleted")
        print()
        print("Cameras are now empty. Register real sources (RTSP, phone IP Webcam,\n"
              "USB webcam, video file, image or web link) from the Cameras page or the API.")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
