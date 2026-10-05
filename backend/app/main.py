"""IRIS FastAPI application.

Run with::

    uvicorn app.main:app --reload --port 8000     (from backend/)

or the convenience script ``python -m app.main``.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, new_request_id, request_id_var, set_request_id
from app.db.session import check_database
from app.schemas import ComponentHealth, SystemHealthResponse
from app.services.db_service import bootstrap_database
from app.services.event_service import event_service, manager
from app.websocket.manager import router as websocket_router

logger = get_logger("app")

DESCRIPTION = """
**IRIS — Intelligent Retail Insight System**

Local-first retail intelligence: camera ingestion, AI inference, retail
analytics and alerting, all on one machine with no cloud dependency.

### Important honesty notes

* Camera-derived numbers are **estimates of visible shelf stock**. They never
  overwrite authoritative inventory (POS/ERP/manual counts).
* Shipped models are **baselines**. Retail accuracy is unvalidated until
  measured against store ground truth — see `docs/AI_MODELS.md`.
* Authentication is disabled by default for local development. Enable it with
  `IRIS_AUTH_ENABLED=true` before exposing IRIS beyond this machine.
* `DEMO MODE` synthetic frames are watermarked and tagged `source=demo` so they
  can never be mistaken for real footage.
"""


async def _background_monitor() -> None:
    """Periodic task: flush coalesced events, record health, run retention."""
    from app.services.health_service import record_sample
    from app.services.retention_service import cleanup

    health_every = 60.0
    retention_every = 3600.0
    started = time.time()
    last_health = 0.0
    last_retention = 0.0
    try:
        while True:
            await asyncio.sleep(5.0)
            now = time.time()
            if now - last_health >= health_every:
                last_health = now
                try:
                    sample = await asyncio.to_thread(record_sample)
                    if sample is not None:
                        await event_service.publish(
                            "system.health",
                            severity="info",
                            message="Periodic health sample recorded.",
                            payload={
                                "status": sample.status,
                                "cpu_percent": sample.cpu_percent,
                                "memory_percent": sample.memory_percent,
                                "queue_depth": sample.queue_depth,
                            },
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Health sampling failed", exc_info=exc)
            if now - last_retention >= retention_every:
                last_retention = now
                try:
                    await asyncio.to_thread(cleanup)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Retention pass failed", exc_info=exc)
    except asyncio.CancelledError:
        raise
    finally:
        _ = started


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown: prepare directories, database, workers."""
    settings.ensure_directories()
    # Bound CPU intra-op threads BEFORE any model loads. With several cameras
    # each defaulting to every core, the OS oversubscribes the CPU and every
    # inference slows down; a small fixed budget keeps total throughput higher
    # and the live feed responsive.
    if settings.torch_num_threads > 0:
        try:
            import torch  # noqa: PLC0415

            torch.set_num_threads(settings.torch_num_threads)
            logger.info("Torch threads bounded", extra={"threads": settings.torch_num_threads})
        except Exception:  # pragma: no cover - torch optional in some deploys
            pass
    logger.info(
        "IRIS starting",
        extra={
            "version": settings.app_version,
            "env": settings.env,
            "device": settings.device,
            "auth_enabled": settings.auth_enabled,
        },
    )
    for warning in settings.warn_if_insecure():
        logger.warning("Insecure configuration", extra={"warning": warning})

    bootstrap = await asyncio.to_thread(bootstrap_database)
    logger.info("Database ready", extra=bootstrap)

    from app.services.camera_manager import camera_manager

    if settings.env != "test":
        started = await asyncio.to_thread(camera_manager.start_all)
        logger.info("Camera workers started", extra={"started": started})

    monitor = asyncio.create_task(_background_monitor())
    app.state.started_at = time.time()
    logger.info("IRIS ready", extra={"startup_seconds": round(time.time() - app.state.started_at, 2)})
    try:
        yield
    finally:
        monitor.cancel()
        await manager.close_all()
        await asyncio.to_thread(camera_manager.stop_all)
        logger.info("IRIS stopped")


app = FastAPI(
    title="IRIS API",
    version=settings.app_version,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "IRIS"},
    license_info={"name": "Internal use"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and emit an access log line for every request."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    request_id_var.set(request_id)
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed",
            extra={"path": request.url.path, "method": request.method},
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "Request completed",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response


register_exception_handlers(app)
app.include_router(api_router, prefix="/api/v1")
app.include_router(websocket_router)


@app.get("/", tags=["meta"], summary="Service banner")
def root() -> dict[str, Any]:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": "/api/v1",
        "websocket": "/ws/events",
        "health": "/api/v1/system/health",
        "auth_enabled": settings.auth_enabled,
        "demo_mode": settings.demo_mode,
    }


@app.get("/health", response_model=SystemHealthResponse, include_in_schema=False)
def liveness() -> SystemHealthResponse:
    """Liveness probe used by Docker and the verification scripts."""
    from app.services.health_service import collect

    snapshot = collect()
    components = [
        ComponentHealth(
            name=name,
            status=data.get("status", "healthy"),
            detail=_short_detail(name, data),
            latency_ms=(data.get("latency_ms") if isinstance(data, dict) else None),
            metrics={k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
            if isinstance(data, dict)
            else {},
        )
        for name, data in snapshot["components"].items()
    ]
    return SystemHealthResponse(
        status=snapshot["status"],
        service=snapshot["service"],
        version=snapshot["version"],
        environment=snapshot["environment"],
        timestamp=snapshot["timestamp"],
        uptime_seconds=snapshot["metrics"]["uptime_seconds"],
        components=components,
        metrics=snapshot["metrics"],
    )


def _short_detail(name: str, data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("detail", "status", "latency_ms"):
        if key in data and data[key] is not None:
            return f"{key}={data[key]}" if not isinstance(data[key], str) else data[key]
    return None


@app.get("/api/v1/system/ready", tags=["meta"], summary="Readiness probe")
def readiness() -> JSONResponse:
    db = check_database()
    ready = db.get("status") == "healthy"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "database": db, "timestamp": datetime.now(tz=timezone.utc).isoformat()},
    )


def run() -> None:
    """Entry point for ``python -m app.main``."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config=None,  # IRIS owns logging configuration
    )


if __name__ == "__main__":
    set_request_id(new_request_id())
    run()