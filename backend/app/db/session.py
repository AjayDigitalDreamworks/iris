"""Database engine, session factory and FastAPI session dependency."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

import time

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("app")


def _engine_kwargs(database_url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        kwargs["pool_size"] = max(10, settings.db_pool_size * 2)
        kwargs["max_overflow"] = 20
        kwargs["pool_timeout"] = 60
    else:
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_pool_size * 2,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return kwargs


def _normalise_sqlite_url(database_url: str) -> str:
    """Make relative SQLite paths absolute so the DB does not depend on cwd."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return database_url
    raw = database_url[len(prefix) :]
    if raw in ("", ":memory:") or raw.startswith(":memory:"):
        return database_url
    from pathlib import Path

    path = settings.resolve_path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


DATABASE_URL = _normalise_sqlite_url(settings.database_url)

engine: Engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
        """WAL + foreign keys + a generous busy timeout.

        WAL lets readers proceed while a writer commits.  Write serialisation is
        handled in ``app.services.db_service`` (a process-wide write lock), which
        is safe because nested transactions would deadlock under
        ``BEGIN IMMEDIATE``.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any failure."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session.

    Read-only requests end their transaction as soon as the handler returns, so
    a long response body never holds the single SQLite writer lock.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database() -> dict[str, Any]:
    """Cheap liveness probe used by the system-health endpoint."""
    started = time.perf_counter()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            if DATABASE_URL.startswith("sqlite"):
                mode = connection.execute(text("PRAGMA journal_mode")).scalar()
            else:
                mode = DATABASE_URL.split("://", 1)[0]
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "healthy",
            "dialect": engine.dialect.name,
            "journal_mode": mode,
            "latency_ms": round(latency_ms, 3),
        }
    except Exception as exc:  # pragma: no cover - exercised via failure injection
        logger.error("Database health check failed", exc_info=exc)
        return {"status": "offline", "dialect": engine.dialect.name, "error": type(exc).__name__}


def create_all() -> None:
    """Create every table directly (used by tests and first-run bootstrap)."""
    import app.models  # noqa: F401  (ensures every model is registered)
    from app.db.base import Base

    Base.metadata.create_all(engine)