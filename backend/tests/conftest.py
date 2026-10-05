"""Shared pytest fixtures.

Every test runs against a **real, isolated SQLite database** created fresh per
session, so tests exercise the actual ORM, migrations-free schema and services
rather than mocks.  Model weights are optional: tests that need inference are
skipped (not silently passed) when the weights are absent.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Point the whole app at an isolated test database BEFORE app imports settings.
os.environ.setdefault("IRIS_ENV", "test")
os.environ.setdefault(
    "IRIS_DATABASE_URL",
    f"sqlite:///{(Path(tempfile.gettempdir()) / 'iris_test.db').as_posix()}",
)
os.environ.setdefault("IRIS_DATA_DIR", str(REPO_ROOT / "data"))
os.environ.setdefault("IRIS_LOG_DIR", str(REPO_ROOT / "logs"))
os.environ.setdefault("IRIS_LOG_LEVEL", "WARNING")
os.environ.setdefault("IRIS_LOG_JSON", "false")
os.environ.setdefault("IRIS_AUTH_ENABLED", "false")
os.environ.setdefault("IRIS_SECRET_KEY", "test-secret-key-not-for-production-use-only")
os.environ.setdefault("IRIS_JWT_SECRET", "test-jwt-secret-not-for-production-use-only")

from app.core.config import settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
import app.models  # noqa: F401,E402


@pytest.fixture(scope="session", autouse=True)
def _schema() -> None:
    """Create the schema once for the whole session."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    """A session bound to a clean, rolled-back transaction."""
    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def client():
    """A FastAPI test client with the full application lifespan.

    Session-scoped on purpose: application startup loads the detector and the
    recogniser index, and re-running that for every test would dominate the
    suite runtime without testing anything new.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def store(db_session):
    """A committed store, deleted afterwards so tests stay isolated."""
    from app.models import Store

    row = Store(code="TEST-STORE", name="Test store", timezone="UTC")
    db_session.add(row)
    db_session.commit()
    try:
        yield row
    finally:
        db_session.query(Store).filter(Store.id == row.id).delete()
        db_session.commit()


@pytest.fixture()
def detector_or_skip():
    """The real detector, or a skip when weights are unavailable."""
    from app.vision.models_registry import registry

    try:
        model = registry.get("product_detector")
        if not model.is_ready:
            model.load()
        model.warmup()
    except Exception as exc:  # pragma: no cover - depends on local weights
        pytest.skip(f"Detector weights unavailable: {exc}")
    return model


@pytest.fixture()
def sample_image_path() -> Path:
    path = REPO_ROOT / "data" / "sample_images" / "bus.jpg"
    if not path.exists():
        pytest.skip("Sample image missing. Run: python scripts/fetch_models.py")
    return path


@pytest.fixture()
def clean_settings():
    """Restore the runtime business rules after a test mutates them."""
    from app.services.db_service import settings_service

    snapshot = settings_service.snapshot()
    yield snapshot
    for key, value in snapshot.items():
        settings_service.upsert(key, value, category="general")
    settings_service.invalidate()


__all__ = ["settings", "engine"]