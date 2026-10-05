"""Small database helpers shared by services and workers.

Worker threads do not have a request-scoped session, so every write goes
through :func:`run_in_session`, which opens, commits and closes its own session.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import SessionLocal

logger = get_logger("app")

T = TypeVar("T")

#: Process-wide write serialisation (see :func:`_run_once`).
_WRITE_LOCK = threading.RLock()


#: SQLite allows a single writer; concurrent camera workers can briefly collide.
#: Retrying with a short backoff turns a hard failure into a small delay.
LOCK_RETRY_ATTEMPTS = 4
LOCK_RETRY_DELAY_SECONDS = 0.25


def _is_lock_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "database is locked" in text or "database table is locked" in text


def run_in_session(work: Callable[[Session], T], session_factory: Any = None) -> T:
    """Run ``work`` inside its own transaction and return its result.

    Safe to call from a worker thread (each call opens and closes its own
    session), and safe to call from a request (pass ``session_factory`` to reuse
    the request session).  Transient SQLite lock contention is retried with a
    bounded backoff rather than surfaced as an error.
    """
    import time

    last_error: Exception | None = None
    for attempt in range(LOCK_RETRY_ATTEMPTS):
        try:
            return _run_once(work, session_factory)
        except Exception as exc:
            if not _is_lock_error(exc) or attempt == LOCK_RETRY_ATTEMPTS - 1:
                raise
            last_error = exc
            logger.warning(
                "Database busy; retrying",
                extra={"attempt": attempt + 1, "error": str(exc)},
            )
            time.sleep(LOCK_RETRY_DELAY_SECONDS * (attempt + 1))
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error
    raise RuntimeError("unreachable")


def _run_once(work: Callable[[Session], T], session_factory: Any) -> T:
    """Run ``work`` once.

    ``session_factory`` is an optional *already-open* session to reuse. Anything
    else (including a session factory callable such as ``SessionLocal``) means
    "use a fresh session": we then own it, serialise the write and always commit
    or roll back and close it. Ownership is derived from what was actually
    passed, never from truthiness, which previously leaked an open transaction
    and locked the database permanently.
    """
    owns_session = not isinstance(session_factory, Session)
    session = SessionLocal() if owns_session else session_factory
    try:
        if owns_session:
            # Single-process write serialisation: SQLite permits one writer and
            # several camera workers persist concurrently. Writes here are small,
            # so serialising them removes lock contention entirely.
            with _WRITE_LOCK:
                result = work(session)
                session.commit()
            return result
        return work(session)
    except Exception:
        if owns_session:
            session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


class UnitOfWork:
    """Context manager form of :func:`run_in_session`."""

    def __init__(self, session: Session | None = None) -> None:
        self.session = session
        self._owns = session is None

    def __enter__(self) -> Session:
        if self.session is None:
            self.session = SessionLocal()
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.session is not None
        try:
            if exc_type is None and self._owns:
                self.session.commit()
            elif exc_type is not None:
                self.session.rollback()
        finally:
            if self._owns:
                self.session.close()


def paginate(query: Any, limit: int, offset: int) -> tuple[list[Any], int]:
    """Return (rows, total) for a SQLAlchemy query."""
    total = query.order_by(None).count()
    rows = query.limit(max(1, limit)).offset(max(0, offset)).all()
    return list(rows), total


# ── Runtime settings ─────────────────────────────────────────────────────────
class SettingsService:
    """Reads and writes runtime business rules stored in the ``settings`` table.

    Values are cached briefly so a busy pipeline does not hit the database on
    every frame, and the cache is invalidated on write.
    """

    def __init__(self, cache_seconds: float = 5.0) -> None:
        self.cache_seconds = cache_seconds
        self._cache: dict[str, Any] = {}
        self._cache_at: float = 0.0

    # ── defaults ────────────────────────────────────────────────────────────
    @staticmethod
    def defaults() -> dict[str, Any]:
        from app.core.config import settings as cfg

        return {
            "low_stock_threshold_pct": cfg.default_low_stock_threshold,
            "out_of_stock_threshold_pct": cfg.default_out_of_stock_threshold,
            "alert_min_confidence": cfg.alert_min_confidence,
            "alert_required_observations": cfg.alert_required_observations,
            "alert_cooldown_seconds": cfg.alert_cooldown_seconds,
            "queue_busy_threshold": cfg.default_queue_busy,
            "queue_critical_threshold": cfg.default_queue_critical,
            "queue_alert_cooldown_seconds": 180,
            "queue_service_time_seconds": 150.0,
            "footfall_dwell_alert_seconds": 900.0,
            "planogram_min_compliance_pct": 85.0,
            "camera_offline_alert_seconds": 60.0,
            "retention_days_detections": cfg.detection_retention_days,
            "retention_days_evidence": cfg.evidence_retention_days,
            "evidence_enabled": cfg.evidence_enabled,
            "anonymize_faces": cfg.anonymize_faces,
        }

    # ── cache ───────────────────────────────────────────────────────────────
    def _valid(self) -> bool:
        import time

        return (time.time() - self._cache_at) < self.cache_seconds

    def invalidate(self) -> None:
        self._cache.clear()
        self._cache_at = 0.0

    def snapshot(self, session: Session | None = None) -> dict[str, Any]:
        """Return every rule value, falling back to documented defaults."""
        if self._valid() and self._cache:
            return dict(self._cache)
        import time

        from app.models import Setting

        values = self.defaults()
        try:
            if session is not None:
                rows = session.query(Setting).filter(Setting.is_secret.is_(False)).all()
            else:

                def _load(sess: Session) -> list[Setting]:
                    return list(sess.query(Setting).filter(Setting.is_secret.is_(False)).all())

                rows = run_in_session(_load)
            for row in rows:
                try:
                    values[row.key] = row.typed_value
                except (TypeError, ValueError):
                    logger.warning("Ignoring malformed setting", extra={"key": row.key})
        except Exception as exc:  # pragma: no cover - DB unavailable at boot
            logger.warning("Falling back to default settings", exc_info=exc)

        self._cache = values
        self._cache_at = time.time()
        return dict(values)

    def get(self, key: str, default: Any = None, session: Session | None = None) -> Any:
        values = self.snapshot(session)
        if key in values:
            return values[key]
        return self.defaults().get(key, default)

    def get_float(self, key: str, default: float = 0.0, session: Session | None = None) -> float:
        try:
            return float(self.get(key, default, session))
        except (TypeError, ValueError):
            return default

    def get_int(self, key: str, default: int = 0, session: Session | None = None) -> int:
        try:
            return int(self.get(key, default, session))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False, session: Session | None = None) -> bool:
        value = self.get(key, default, session)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    # ── derived rule objects ────────────────────────────────────────────────
    def stock_thresholds(self, session: Session | None = None):
        from app.vision.rules import StockThresholds

        return StockThresholds(
            low_stock_pct=self.get_float("low_stock_threshold_pct", 20.0, session),
            out_of_stock_pct=self.get_float("out_of_stock_threshold_pct", 5.0, session),
            min_confidence=self.get_float("alert_min_confidence", 0.8, session),
            required_observations=self.get_int("alert_required_observations", 3, session),
        )

    def alert_thresholds(self, session: Session | None = None):
        from app.vision.rules import AlertThresholds

        return AlertThresholds(
            cooldown_seconds=self.get_int("alert_cooldown_seconds", 300, session),
            min_confidence=self.get_float("alert_min_confidence", 0.8, session),
            required_observations=self.get_int("alert_required_observations", 3, session),
        )

    def queue_thresholds(self, session: Session | None = None):
        from app.vision.rules import QueueThresholds

        return QueueThresholds(
            busy=self.get_int("queue_busy_threshold", 4, session),
            critical=self.get_int("queue_critical_threshold", 7, session),
            service_time_seconds=self.get_float("queue_service_time_seconds", 150.0, session),
        )

    # ── mutation ────────────────────────────────────────────────────────────
    def upsert(
        self,
        key: str,
        value: Any,
        *,
        category: str = "general",
        value_type: str | None = None,
        description: str | None = None,
        updated_by: str | None = None,
        session: Session | None = None,
    ) -> None:
        from app.models import Setting

        inferred = value_type or _infer_type(value)

        def _write(sess: Session) -> None:
            row = sess.query(Setting).filter(Setting.key == key).first()
            if row is None:
                sess.add(
                    Setting(
                        key=key,
                        value=value,
                        value_type=inferred,
                        category=category,
                        description=description,
                        updated_by=updated_by,
                    )
                )
            else:
                row.value = value
                row.value_type = inferred
                row.category = category
                if description is not None:
                    row.description = description
                row.updated_by = updated_by

        run_in_session(_write, session)
        self.invalidate()

    def ensure_defaults(self, session: Session | None = None) -> int:
        """Seed any missing default rule. Safe to call on every startup."""
        from app.models import Setting, default_settings_payload

        def _write(sess: Session) -> int:
            existing = {row.key for row in sess.query(Setting).all()}
            created = 0
            for payload in default_settings_payload():
                if payload["key"] not in existing:
                    sess.add(Setting(**payload))
                    created += 1
            return created

        created = run_in_session(_write, session)
        self.invalidate()
        return created


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "json"


settings_service = SettingsService()


def bootstrap_database(session_factory: Any = None) -> dict[str, Any]:
    """Create tables when needed and ensure the default rules exist."""
    from app.db.session import create_all, engine

    created_tables = False
    try:
        with engine.connect() as connection:
            from sqlalchemy import inspect

            inspector = inspect(connection)
            created_tables = "stores" not in inspector.get_table_names()
    except Exception:  # pragma: no cover - unreachable DB
        created_tables = False
    if created_tables:
        create_all()

    created_settings = settings_service.ensure_defaults(session_factory)
    return {"created_tables": created_tables, "created_settings": created_settings}