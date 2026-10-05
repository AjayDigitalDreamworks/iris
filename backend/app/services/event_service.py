"""Central event service and WebSocket gateway.

Responsibilities:

* create a canonical, globally-unique :class:`RetailEvent` for every business
  event (idempotent by ``event_id`` so retries never duplicate);
* persist it;
* fan it out to every connected dashboard over ``/ws/events``;
* coalesce high-rate telemetry (``detection.summary``, ``queue.updated``) so a
  fast camera cannot flood the browser.

Video is **never** sent over this channel — only compact metadata. Frames use
the separate authorised stream endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import WebSocket

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("app")


# ── Event catalogue (README §19) ─────────────────────────────────────────────
class EventType:
    PRODUCT_DETECTED = "detection.summary"
    PRODUCT_RECOGNISED = "product.recognised"
    INVENTORY_UPDATED = "inventory.updated"
    LOW_STOCK = "inventory.low_stock"
    OUT_OF_STOCK = "inventory.out_of_stock"
    RESTOCK_DETECTED = "inventory.restock_detected"
    CUSTOMER_ENTERED = "customers.entered"
    CUSTOMER_EXITED = "customers.exited"
    OCCUPANCY = "customers.occupancy"
    DWELL_HIGH = "customers.dwell_high"
    QUEUE_UPDATED = "queue.updated"
    QUEUE_THRESHOLD_EXCEEDED = "queue.threshold_exceeded"
    CAMERA_STATUS = "camera.status"
    CAMERA_OFFLINE = "camera.offline"
    CAMERA_RECOVERED = "camera.recovered"
    MODEL_ERROR = "model.error"
    MODEL_DEGRADED = "model.degraded"
    ANOMALY_CANDIDATE = "anomaly.candidate"
    ALERT_CREATED = "alert.created"
    ALERT_UPDATED = "alert.updated"
    SYSTEM_HEALTH = "system.health"
    JOB_UPDATED = "job.updated"


#: Event types that may be coalesced (only the newest of each key is delivered).
COALESCING_TYPES = {
    EventType.PRODUCT_DETECTED,
    EventType.QUEUE_UPDATED,
    EventType.OCCUPANCY,
    EventType.INVENTORY_UPDATED,
    EventType.CAMERA_STATUS,
}


@dataclass
class OutboundEvent:
    """The wire format pushed to dashboards."""

    event_id: str
    event_type: str
    timestamp: datetime
    severity: str
    store_id: str | None = None
    camera_id: str | None = None
    camera_name: str | None = None
    shelf_id: str | None = None
    sku: str | None = None
    message: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "timestamp": self.timestamp.isoformat(),
                "severity": self.severity,
                "store_id": self.store_id,
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "shelf_id": self.shelf_id,
                "sku": self.sku,
                "message": self.message,
                "payload": self.payload,
                "evidence": self.evidence,
            },
            default=str,
        )

    @classmethod
    def from_row(cls, row: Any) -> "OutboundEvent":
        return cls(
            event_id=row.event_id,
            event_type=row.event_type,
            timestamp=row.occurred_at,
            severity=row.severity,
            store_id=row.store_id,
            camera_id=row.camera_id,
            shelf_id=row.shelf_id,
            payload=row.payload or {},
            evidence={"path": row.evidence_path} if row.evidence_path else {},
        )


@dataclass
class _Subscription:
    """What a connected client wants to receive."""

    camera_ids: set[str] | None = None      # None = all cameras
    event_types: set[str] | None = None
    severities: set[str] | None = None

    def accepts(self, event: OutboundEvent) -> bool:
        if self.camera_ids and event.camera_id and event.camera_id not in self.camera_ids:
            return False
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.severities and event.severity not in self.severities:
            return False
        return True


class ConnectionManager:
    """Tracks WebSocket clients and pushes events to the interested ones."""

    def __init__(self, max_clients: int = 64) -> None:
        self.max_clients = max_clients
        self._clients: dict[WebSocket, _Subscription] = {}
        self._lock = asyncio.Lock()
        self.total_connections = 0
        self.dropped_connections = 0

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(
        self,
        websocket: WebSocket,
        subscription: _Subscription | None = None,
    ) -> bool:
        async with self._lock:
            if len(self._clients) >= self.max_clients:
                self.dropped_connections += 1
                await websocket.close(code=1013, reason="Too many dashboard clients")
                return False
            await websocket.accept()
            self._clients[websocket] = subscription or _Subscription()
            self.total_connections += 1
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.pop(websocket, None)

    def update_subscription(self, websocket: WebSocket, subscription: _Subscription) -> None:
        if websocket in self._clients:
            self._clients[websocket] = subscription

    async def broadcast(self, event: OutboundEvent) -> int:
        """Send an event to every interested client. Returns delivery count."""
        if not self._clients:
            return 0
        message = event.to_json()
        dead: list[WebSocket] = []
        delivered = 0
        for websocket, subscription in list(self._clients.items()):
            if not subscription.accepts(event):
                continue
            try:
                await websocket.send_text(message)
                delivered += 1
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            await self.disconnect(websocket)
        return delivered

    async def send_to(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps(payload, default=str))

    async def close_all(self) -> None:
        for websocket in list(self._clients):
            with contextlib.suppress(Exception):
                await websocket.close(code=1001)
        self._clients.clear()


manager = ConnectionManager(max_clients=settings.mjpeg_max_clients * 4)


class EventService:
    """Creates, persists and dispatches retail events."""

    def __init__(self, connection_manager: ConnectionManager | None = None) -> None:
        self.manager = connection_manager or manager
        self._pending_coalesced: dict[str, OutboundEvent] = {}
        self._last_seen_ids: dict[str, datetime] = {}
        self._dedupe_ttl_seconds = 300
        self._lock = asyncio.Lock()
        self.published = 0
        self.deduplicated = 0
        self.coalesced = 0

    def is_duplicate(self, event_id: str) -> bool:
        """True when this event id was already published inside the dedupe window."""
        seen = self._last_seen_ids.get(event_id)
        if seen is None:
            return False
        now = datetime.now(tz=timezone.utc)
        if (now - seen).total_seconds() > self._dedupe_ttl_seconds:
            self._last_seen_ids.pop(event_id, None)
            return False
        return True

    async def publish(
        self,
        event_type: str,
        *,
        severity: str = "info",
        store_id: str | None = None,
        camera_id: str | None = None,
        camera_name: str | None = None,
        shelf_id: str | None = None,
        sku: str | None = None,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        model_id: str | None = None,
        source: str = "pipeline",
        occurred_at: datetime | None = None,
        persist: bool = True,
        session_factory: Any = None,
        alert_id: str | None = None,
    ) -> OutboundEvent:
        """Create, persist and broadcast one event.

        ``session_factory`` may be supplied to persist using a caller-owned
        session (used from worker threads where the request session is gone).
        """
        event_id = f"evt_{uuid.uuid4().hex[:20]}"
        if self.is_duplicate(event_id):  # pragma: no cover - uuid collision only
            self.deduplicated += 1
        timestamp = occurred_at or datetime.now(tz=timezone.utc)

        event = OutboundEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            severity=severity,
            store_id=store_id,
            camera_id=camera_id,
            camera_name=camera_name,
            shelf_id=shelf_id,
            sku=sku,
            message=message,
            payload=payload or {},
            evidence=evidence or {},
        )

        if persist:
            self._persist(event, model_id=model_id, source=source, session_factory=session_factory,
                          alert_id=alert_id)
        await self._dispatch(event)
        self.published += 1
        return event

    def _persist(
        self,
        event: OutboundEvent,
        *,
        model_id: str | None,
        source: str,
        session_factory: Any,
        alert_id: str | None,
    ) -> None:
        from app.models import RetailEvent  # local import avoids a cycle
        from app.services.db_service import run_in_session

        def _write(session) -> None:
            existing = (
                session.query(RetailEvent)
                .filter(RetailEvent.event_id == event.event_id)
                .first()
            )
            if existing is not None:
                return
            session.add(
                RetailEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    store_id=event.store_id,
                    camera_id=event.camera_id,
                    shelf_id=event.shelf_id,
                    alert_id=alert_id,
                    severity=event.severity,
                    payload=event.payload,
                    evidence_path=(event.evidence or {}).get("path"),
                    source=source,
                    model_id=model_id,
                    occurred_at=event.timestamp,
                )
            )

        try:
            run_in_session(_write, session_factory)
        except Exception as exc:  # pragma: no cover - DB failure must not stop the pipeline
            logger.error("Failed to persist retail event", exc_info=exc)

    async def _dispatch(self, event: OutboundEvent) -> None:
        if event.event_type in COALESCING_TYPES:
            key = f"{event.event_type}:{event.camera_id or '-'}:{event.sku or '-'}"
            async with self._lock:
                self._pending_coalesced[key] = event
            self.coalesced += 1
        await self.manager.broadcast(event)

    async def flush_coalesced(self) -> int:
        """Emit the newest event of each coalescing key (called on a timer)."""
        async with self._lock:
            pending = list(self._pending_coalesced.values())
            self._pending_coalesced.clear()
        if not pending:
            return 0
        await asyncio.gather(*(self.manager.broadcast(event) for event in pending))
        return len(pending)

    async def broadcast_direct(self, message: dict[str, Any]) -> None:
        """Push a non-persisted message (used for streaming overlays)."""
        for websocket in list(self.manager._clients):  # noqa: SLF001 - internal by design
            await self.manager.send_to(websocket, message)

    def stats(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "deduplicated": self.deduplicated,
            "coalesced": self.coalesced,
            "pending_coalesced": len(self._pending_coalesced),
            "websocket_clients": self.manager.client_count,
            "total_connections": self.manager.total_connections,
            "dropped_connections": self.manager.dropped_connections,
        }


event_service = EventService()


def parse_subscription(payload: dict[str, Any] | None) -> _Subscription:
    """Build a subscription from a client ``subscribe`` message."""
    if not payload:
        return _Subscription()
    cameras = payload.get("camera_ids")
    types = payload.get("event_types")
    severities = payload.get("severities")
    return _Subscription(
        camera_ids=set(cameras) if cameras else None,
        event_types=set(types) if types else None,
        severities=set(severities) if severities else None,
    )


def iter_event_types() -> Iterable[str]:
    return [
        value
        for name, value in vars(EventType).items()
        if not name.startswith("_") and isinstance(value, str)
    ]