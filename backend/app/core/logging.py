"""Structured logging with separated channels.

IRIS writes four independent streams so operational, AI, camera and security
concerns can be routed, retained and scanned separately::

    iris.app       application / request / service logs
    iris.ai        model loading, inference latency, model errors
    iris.camera    capture lifecycle, reconnects, stream health
    iris.security  authentication, authorization, audit, input rejection

Every record is emitted as a single line of JSON with a stable field set:
``timestamp, level, channel, module, message`` plus optional
``request_id, camera_id, model_id, event_id, exception``.

Passwords, bearer tokens and RTSP credentials are redacted before a record ever
reaches a handler.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings

# ── Request correlation ──────────────────────────────────────────────────────
request_id_var: ContextVar[str | None] = ContextVar("iris_request_id", default=None)
camera_id_var: ContextVar[str | None] = ContextVar("iris_camera_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(request_id: str | None) -> None:
    request_id_var.set(request_id)


def set_camera_id(camera_id: str | None) -> None:
    camera_id_var.set(camera_id)


# ── Redaction ────────────────────────────────────────────────────────────────
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # rtsp://user:password@host  ->  rtsp://user:***@host
    re.compile(r"(?P<scheme>rtsp|rtsps|http|https)://(?P<user>[^:/@\s]+):(?P<pw>[^@\s]+)@"),
    # key=value / key: value for well-known secret names
    re.compile(
        r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|authorization|jwt)\b"
        r"(\s*[:=]\s*|\"\s*:\s*\")(?P<val>[^\s,;}\"']+)"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
)

REDACTED = "***"


def redact(text: str) -> str:
    """Remove credentials from arbitrary text before it is logged."""
    if not text:
        return text
    out = text
    out = _SECRET_PATTERNS[0].sub(lambda m: f"{m.group('scheme')}://{m.group('user')}:{REDACTED}@", out)
    out = _SECRET_PATTERNS[1].sub(
        lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out
    )
    out = _SECRET_PATTERNS[2].sub("Bearer " + REDACTED, out)
    return out


# ── Formatter ────────────────────────────────────────────────────────────────
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class RedactingFilter(logging.Filter):
    """Last-line defence: scrub secrets from the final formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(str(record.msg))
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: redact(str(v)) for k, v in record.args.items()}
                else:
                    record.args = tuple(redact(str(a)) for a in record.args)
        except Exception:  # pragma: no cover - logging must never raise
            pass
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "channel": getattr(record, "channel", record.name),
            "module": record.module,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_") and key != "channel":
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        for ctx_key, var in (("request_id", request_id_var), ("camera_id", camera_id_var)):
            ctx_value = var.get()
            if ctx_value and ctx_key not in payload:
                payload[ctx_key] = ctx_value
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = f"{exc_type.__name__}: {redact(str(exc_value))}"
            payload["traceback"] = redact(
                "".join(traceback.format_exception(*record.exc_info))
            )
        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Readable console formatter (development convenience)."""

    default_fmt = "%(asctime)s %(levelname)-7s [%(channel)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.default_fmt, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.channel = getattr(record, "channel", record.name)  # type: ignore[attr-defined]
        text = super().format(record)
        if record.exc_info:
            text = f"{text}\n{redact(''.join(traceback.format_exception(*record.exc_info)))}"
        return redact(text)


class ChannelFilter(logging.Filter):
    """Attach the ``channel`` attribute used by both formatters."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "channel"):
            record.channel = record.name.split(".")[-1]  # type: ignore[attr-defined]
        return True


# ── Configuration ────────────────────────────────────────────────────────────
_configured = False


class SafeExtraLogger(logging.Logger):
    """A logger that cannot be crashed by a reserved ``extra`` key.

    ``logging.Logger.makeRecord`` raises ``KeyError`` when ``extra`` contains a
    key that already exists on the LogRecord (``name``, ``filename``, ``msg``
    and ~20 others). That exception propagates into whatever code was doing the
    logging — here that meant creating a camera returned a 500. Reserved keys
    are now renamed instead of crashing.
    """

    def makeRecord(  # noqa: N802 - stdlib naming
        self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None
    ) -> logging.LogRecord:
        if extra:
            safe = {}
            for key, value in extra.items():
                safe[f"field_{key}" if key in _RESERVED else key] = value
            extra = safe
        return super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )


def _make_logger(channel: str) -> logging.Logger:
    logger = logging.getLogger(f"iris.{channel}")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    # Guard every IRIS channel against reserved ``extra`` keys.
    if not isinstance(logger, SafeExtraLogger):
        logger.__class__ = SafeExtraLogger
    logger.propagate = False
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = JsonFormatter() if settings.log_json else HumanFormatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(ChannelFilter())
    stream.addFilter(RedactingFilter())
    logger.addHandler(stream)

    log_dir = settings.resolve_path(settings.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{channel}.log",
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ChannelFilter())
        file_handler.addFilter(RedactingFilter())
        logger.addHandler(file_handler)
    except OSError:  # pragma: no cover - read-only filesystem is survivable
        pass

    # Third-party libraries are noisy and untrusted; keep them on the app channel
    # at WARNING so our own structured records dominate.
    return logger


def configure_logging() -> None:
    """Idempotently configure every IRIS logging channel."""
    global _configured
    if _configured:
        return
    for channel in ("app", "ai", "camera", "security"):
        _make_logger(channel)
    for noisy in ("uvicorn.access", "uvicorn.error", "sqlalchemy.engine", "matplotlib", "PIL"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = False
    _configured = True


def get_logger(channel: str = "app") -> logging.Logger:
    """Return a configured IRIS logger for the given channel."""
    configure_logging()
    return logging.getLogger(f"iris.{channel}")


# Convenience accessors -------------------------------------------------------
app_log = get_logger("app")
ai_log = get_logger("ai")
camera_log = get_logger("camera")
security_log = get_logger("security")