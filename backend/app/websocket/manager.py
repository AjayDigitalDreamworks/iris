"""WebSocket gateway for live dashboard events (``/ws/events``).

Protocol (README §6.3):

* Server pushes compact event envelopes: ``camera.status``,
  ``detection.summary``, ``inventory.updated``, ``alert.created``,
  ``alert.updated``, ``queue.updated``, ``system.health``.
* Client may send ``{"action": "subscribe", "camera_ids": [...]}`` to filter,
  ``{"action": "ping"}`` for a heartbeat, or ``{"action": "snapshot"}`` to ask
  for the current state.

Video frames are **not** sent here — they use the authorised MJPEG endpoint.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging import get_logger, new_request_id, request_id_var
from app.core.security import decode_access_token
from app.services.event_service import event_service, manager, parse_subscription

router = APIRouter(tags=["websocket"])
logger = get_logger("app")

HEARTBEAT_SECONDS = 20.0


async def _authorise(token: str | None) -> str | None:
    """Return an error message when the connection must be refused."""
    if not settings.auth_enabled:
        return None
    if not token:
        return "Authentication required: pass ?token=<access_token>."
    try:
        decode_access_token(token)
    except Exception:
        return "Invalid or expired access token."
    return None


def _current_state() -> dict[str, Any]:
    """Snapshot used by ``action: snapshot``."""
    from app.services.analytics_service import analytics_service
    from app.services.camera_manager import camera_manager

    try:
        data = analytics_service.dashboard()
        return {
            "cameras": data["cameras"],
            "inventory": data["inventory"],
            "alerts": data["alerts"],
            "ai": {k: v for k, v in data["ai"].items() if k in {"device", "provider"}},
            "queues": data["queues"],
            "queues_configured": len(data["queues"]) > 0,
        }
    except Exception as exc:  # pragma: no cover - must never kill the socket
        logger.warning("Snapshot failed", exc_info=exc)
        return {}


@router.websocket("/ws/events")
async def events_socket(
    websocket: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    request_id_var.set(new_request_id())
    refusal = await _authorise(token)
    if refusal:
        await websocket.close(code=4401, reason=refusal[:120])
        return

    if not await manager.connect(websocket):
        return

    logger.info("Dashboard connected", extra={"clients": manager.client_count})
    await manager.send_to(
        websocket,
        {
            "event_type": "connection.ready",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "payload": {
                "heartbeat_seconds": HEARTBEAT_SECONDS,
                "event_types": list(_event_types()),
                "auth_enabled": settings.auth_enabled,
            },
        },
    )

    heartbeat = asyncio.create_task(_heartbeat(websocket))
    try:
        while True:
            raw = await websocket.receive_text()
            await _handle_client_message(websocket, raw)
    except WebSocketDisconnect:
        logger.info("Dashboard disconnected", extra={"clients": manager.client_count})
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("WebSocket error", exc_info=exc)
    finally:
        heartbeat.cancel()
        await manager.disconnect(websocket)


async def _handle_client_message(websocket: WebSocket, raw: str) -> None:
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        await manager.send_to(websocket, {"event_type": "error", "message": "Invalid JSON."})
        return
    if not isinstance(message, dict):
        await manager.send_to(websocket, {"event_type": "error", "message": "Expected an object."})
        return

    action = message.get("action")
    if action == "ping":
        await manager.send_to(
            websocket,
            {"event_type": "pong", "timestamp": datetime.now(tz=timezone.utc).isoformat()},
        )
    elif action == "subscribe":
        manager.update_subscription(websocket, parse_subscription(message))
        await manager.send_to(websocket, {"event_type": "subscribed", "payload": message})
    elif action == "snapshot":
        await manager.send_to(
            websocket,
            {
                "event_type": "snapshot",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "payload": _current_state(),
            },
        )
    else:
        await manager.send_to(
            websocket,
            {"event_type": "error", "message": f"Unknown action: {action!r}"},
        )


async def _heartbeat(websocket: WebSocket) -> None:
    """Periodic coalesced flush + keepalive so proxies do not drop the socket."""
    try:
        while True:
            await asyncio.sleep(2.0)
            await event_service.flush_coalesced()
            await asyncio.sleep(HEARTBEAT_SECONDS - 2.0)
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover
        return


def _event_types() -> list[str]:
    from app.services.event_service import iter_event_types

    return list(iter_event_types())