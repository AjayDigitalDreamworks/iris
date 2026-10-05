"""Analytics service — aggregate footfall, queue, shelf and dashboard metrics.

All queries read persisted rows only, so historical charts work for any time
range regardless of whether a camera is currently running.  Bucket size adapts
to the requested range so a 24-hour view is readable and a 7-day view is not
a wall of points.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import func

from app.core.logging import get_logger
from app.models import (
    ACTIVE_STATUSES,
    Alert,
    Camera,
    CameraStatus,
    Detection,
    FootfallEvent,
    Product,
    QueueSnapshot,
    RetailEvent,
    Shelf,
    Track,
)
from app.services.db_service import run_in_session, settings_service
from app.vision.rules import QueueStatus, queue_status

logger = get_logger("app")

BUCKET_GRANULARITIES = (
    ("minute", 60),
    ("5min", 300),
    ("15min", 900),
    ("hour", 3600),
    ("day", 86400),
)


def choose_granularity(range_seconds: float) -> tuple[str, int]:
    """Pick a bucket size that yields a readable number of points."""
    target_points = 60
    bucket = max(60, int(range_seconds / target_points))
    for name, seconds in BUCKET_GRANULARITIES:
        if seconds >= bucket:
            return name, seconds
    return BUCKET_GRANULARITIES[-1]


def floor_to_bucket(moment: datetime, seconds: int) -> datetime:
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=timezone.utc)


@dataclass
class RangeQuery:
    range_start: datetime
    range_end: datetime
    camera_id: str | None = None
    shelf_id: str | None = None
    zone_id: str | None = None
    store_id: str | None = None


class AnalyticsService:
    """Read-only aggregation over persisted observations."""

    # ── footfall ────────────────────────────────────────────────────────────
    def footfall(self, query: RangeQuery) -> dict[str, Any]:
        span = (query.range_end - query.range_start).total_seconds()
        granularity, bucket_seconds = choose_granularity(span)

        def _load(session) -> dict[str, Any]:
            events_query = session.query(FootfallEvent).filter(
                FootfallEvent.occurred_at >= query.range_start,
                FootfallEvent.occurred_at <= query.range_end,
            )
            if query.camera_id:
                events_query = events_query.filter(FootfallEvent.camera_id == query.camera_id)
            if query.zone_id:
                events_query = events_query.filter(FootfallEvent.zone_id == query.zone_id)
            events = events_query.order_by(FootfallEvent.occurred_at.asc()).all()

            buckets: dict[datetime, dict[str, int]] = defaultdict(lambda: {"entries": 0, "exits": 0})
            for event in events:
                key = floor_to_bucket(event.occurred_at, bucket_seconds)
                if event.event_type == "in":
                    buckets[key]["entries"] += 1
                elif event.event_type == "out":
                    buckets[key]["exits"] += 1

            track_query = session.query(Track).filter(Track.first_seen <= query.range_end)
            if query.camera_id:
                track_query = track_query.filter(Track.camera_id == query.camera_id)
            tracks = track_query.all()
            dwell_values = [t.dwell_seconds for t in tracks if t.dwell_seconds is not None]

            camera = session.get(Camera, query.camera_id) if query.camera_id else None
            active_tracks = (
                session.query(func.count(Track.id))
                .filter(Track.camera_id == query.camera_id, Track.is_active.is_(True))
                .scalar()
                if query.camera_id
                else 0
            )

            entries = sum(b["entries"] for b in buckets.values())
            exits = sum(b["exits"] for b in buckets.values())
            points = [
                {
                    "bucket": bucket,
                    "entries": data["entries"],
                    "exits": data["exits"],
                    "net": data["entries"] - data["exits"],
                }
                for bucket, data in sorted(buckets.items())
            ]
            return {
                "camera_id": query.camera_id,
                "camera_name": camera.name if camera else None,
                "granularity": granularity,
                "total_entries": entries,
                "total_exits": exits,
                "current_occupancy": int(active_tracks or 0),
                "peak_occupancy": max((p["net"] for p in points), default=0),
                "mean_dwell_seconds": (
                    round(sum(dwell_values) / len(dwell_values), 2) if dwell_values else None
                ),
                "max_dwell_seconds": round(max(dwell_values), 2) if dwell_values else None,
                "active_tracks": int(active_tracks or 0),
                "points": points,
                "note": (
                    "Anonymous aggregate counts from temporary track IDs. "
                    "No identity, face or biometric information is stored."
                ),
            }

        return run_in_session(_load)

    # ── queue ───────────────────────────────────────────────────────────────
    def queues(self, query: RangeQuery) -> list[dict[str, Any]]:
        span = (query.range_end - query.range_start).total_seconds()
        granularity, bucket_seconds = choose_granularity(span)
        thresholds = settings_service.queue_thresholds()

        def _load(session) -> list[dict[str, Any]]:
            snapshots_query = session.query(QueueSnapshot).filter(
                QueueSnapshot.measured_at >= query.range_start,
                QueueSnapshot.measured_at <= query.range_end,
            )
            if query.camera_id:
                snapshots_query = snapshots_query.filter(QueueSnapshot.camera_id == query.camera_id)
            if query.zone_id:
                snapshots_query = snapshots_query.filter(QueueSnapshot.zone_id == query.zone_id)
            snapshots = snapshots_query.order_by(QueueSnapshot.measured_at.asc()).all()
            if not snapshots:
                return []

            grouped: dict[str, list[QueueSnapshot]] = defaultdict(list)
            for snapshot in snapshots:
                grouped[snapshot.zone_id or "unknown"].append(snapshot)

            results: list[dict[str, Any]] = []
            for zone_id, rows in grouped.items():
                zone = _zone_name(session, zone_id)
                buckets: dict[datetime, list[QueueSnapshot]] = defaultdict(list)
                for row in rows:
                    buckets[floor_to_bucket(row.measured_at, bucket_seconds)].append(row)
                points = [
                    {
                        "bucket": bucket,
                        "queue_length": float(sum(r.queue_length for r in group) / len(group)),
                        "estimated_wait_seconds": round(
                            sum(r.estimated_wait_seconds or 0 for r in group) / len(group), 1
                        ),
                        "status": queue_status(
                            int(round(sum(r.queue_length for r in group) / len(group))), thresholds
                        ).value,
                        "service_zone_occupancy": max(r.service_zone_occupancy for r in group),
                    }
                    for bucket, group in sorted(buckets.items())
                ]
                latest = rows[-1]
                lengths = [r.queue_length for r in rows]
                results.append(
                    {
                        "zone_id": zone_id,
                        "zone_name": zone,
                        "camera_id": latest.camera_id,
                        "granularity": granularity,
                        "current_queue_length": latest.queue_length,
                        "service_zone_occupancy": latest.service_zone_occupancy,
                        "service_stations": latest.service_stations,
                        "estimated_wait_seconds": latest.estimated_wait_seconds,
                        "status": latest.status,
                        "busy_threshold": thresholds.busy,
                        "critical_threshold": thresholds.critical,
                        "mean_queue_length": round(sum(lengths) / len(lengths), 2),
                        "peak_queue_length": max(lengths),
                        "points": points,
                        "note": (
                            "Queue length counts tracked people inside the configured zone; "
                            "waiting time is an M/M/c estimate, not a guaranteed wait."
                        ),
                    }
                )
            return sorted(results, key=lambda r: r["zone_id"] or "")

        return run_in_session(_load)

    # ── shelves ─────────────────────────────────────────────────────────────
    def shelves(
        self, query: RangeQuery, *, within_minutes: int = 15
    ) -> list[dict[str, Any]]:
        from app.models import InventorySnapshot
        from app.vision.retail_detector import ShelfStockView, shelf_stock_view
        from app.vision.rules import classify_stock

        thresholds = settings_service.stock_thresholds()
        since = query.range_end - timedelta(minutes=within_minutes)

        def _load(session) -> list[dict[str, Any]]:
            shelf_query = session.query(Shelf)
            if query.store_id:
                shelf_query = shelf_query.filter(Shelf.store_id == query.store_id)
            if query.shelf_id:
                shelf_query = shelf_query.filter(Shelf.id == query.shelf_id)
            shelves = shelf_query.all()
            rows: list[dict[str, Any]] = []
            for shelf in shelves:
                snapshots = (
                    session.query(InventorySnapshot)
                    .filter(InventorySnapshot.shelf_id == shelf.id, InventorySnapshot.observed_at >= since)
                    .order_by(InventorySnapshot.observed_at.asc())
                    .all()
                )
                if not snapshots:
                    rows.append(
                        {
                            "shelf_id": shelf.id,
                            "shelf_code": shelf.code,
                            "shelf_name": shelf.name,
                            "camera_id": shelf.camera_id,
                            "capacity_units": shelf.capacity_units,
                            "observed_units": 0,
                            "estimated_quantity": 0.0,
                            "stock_percentage": 0.0,
                            "occupancy_pct": None,
                            "confidence": 0.0,
                            "detected_skus": 0,
                            "empty_positions": None,
                            "planogram_compliance_pct": None,
                            "last_observed_at": None,
                            "state": "unknown",
                            # No observation at all: NOT out of stock, simply
                            # unknown. Reporting an unobserved shelf as empty
                            # would be a fabricated alert.
                            "is_out_of_stock": False,
                            "out_of_stock_reason": "no observation in the last window",
                        }
                    )
                    continue
                latest = snapshots[-1]
                occupancy = latest.occupancy_pct
                shelf_view = shelf_stock_view(
                    ShelfStockView(
                        occupancy_pct=float(occupancy or 0.0),
                        detected_units=int(latest.observed_quantity or 0),
                        empty_positions=0,
                        total_positions=0,
                    )
                )
                state = classify_stock(
                    latest.estimated_quantity,
                    shelf.capacity_units,
                    latest.confidence,
                    thresholds,
                )
                rows.append(
                    {
                        "shelf_id": shelf.id,
                        "shelf_code": shelf.code,
                        "shelf_name": shelf.name,
                        "camera_id": shelf.camera_id,
                        "capacity_units": shelf.capacity_units,
                        "observed_units": latest.observed_quantity,
                        "estimated_quantity": round(latest.estimated_quantity, 3),
                        "stock_percentage": round(latest.stock_percentage, 2),
                        "occupancy_pct": round(occupancy, 2) if occupancy is not None else None,
                        "confidence": round(latest.confidence, 4),
                        "detected_skus": len({s.sku for s in snapshots}),
                        "empty_positions": None,
                        "planogram_compliance_pct": _planogram_compliance(session, shelf.id),
                        "last_observed_at": latest.observed_at,
                        "state": state.value,
                        # Decided by the same rule the vision module uses, so
                        # the badge on screen and the alert in the log can never
                        # disagree.
                        "is_out_of_stock": shelf_view.is_empty,
                        "out_of_stock_reason": (
                            "no units visible and the shelf area is empty"
                            if shelf_view.is_empty
                            else None
                        ),
                    }
                )
            return rows

        return run_in_session(_load)

    # ── detections ──────────────────────────────────────────────────────────
    def detection_summary(
        self, query: RangeQuery, *, top_n: int = 10
    ) -> dict[str, Any]:
        def _load(session) -> dict[str, Any]:
            base = session.query(Detection).filter(
                Detection.frame_timestamp >= query.range_start,
                Detection.frame_timestamp <= query.range_end,
            )
            if query.camera_id:
                base = base.filter(Detection.camera_id == query.camera_id)
            if query.shelf_id:
                base = base.filter(Detection.shelf_id == query.shelf_id)

            total = base.count()
            recognised = base.filter(Detection.sku.isnot(None)).count()
            rows = (
                base.with_entities(Detection.class_name, func.count(Detection.id))
                .group_by(Detection.class_name)
                .order_by(func.count(Detection.id).desc())
                .limit(top_n)
                .all()
            )
            sku_rows = (
                base.filter(Detection.sku.isnot(None))
                .with_entities(Detection.sku, func.count(Detection.id))
                .group_by(Detection.sku)
                .order_by(func.count(Detection.id).desc())
                .limit(top_n)
                .all()
            )
            confidence = base.with_entities(func.avg(Detection.confidence)).scalar()
            return {
                "total_detections": total,
                "recognised_detections": recognised,
                "unknown_detections": total - recognised,
                "mean_confidence": round(float(confidence), 4) if confidence else 0.0,
                "top_classes": [{"class_name": name, "count": count} for name, count in rows],
                "top_skus": [{"sku": sku, "count": count} for sku, count in sku_rows],
            }

        return run_in_session(_load)

    # ── alerts ──────────────────────────────────────────────────────────────
    def alert_summary(self, query: RangeQuery) -> dict[str, Any]:
        def _load(session) -> dict[str, Any]:
            active = session.query(Alert).filter(Alert.status.in_(ACTIVE_STATUSES)).all()
            since = query.range_end - timedelta(days=1)
            resolved = session.query(Alert).filter(
                Alert.status == "resolved", Alert.resolved_at >= since
            ).all()
            by_severity: dict[str, int] = defaultdict(int)
            by_type: dict[str, int] = defaultdict(int)
            for alert in active:
                by_severity[alert.severity] += 1
                by_type[alert.alert_type] += 1
            oldest = min((a.created_at for a in active), default=None)
            return {
                "active": sum(1 for a in active if a.status == "active"),
                "acknowledged": sum(1 for a in active if a.status == "acknowledged"),
                "resolved_24h": len(resolved),
                "by_severity": dict(by_severity),
                "by_type": dict(by_type),
                "oldest_active_at": oldest,
            }

        return run_in_session(_load)

    # ── dashboard ───────────────────────────────────────────────────────────
    def dashboard(self, *, store_id: str | None = None, camera_id: str | None = None) -> dict[str, Any]:
        from app.services.camera_manager import camera_manager
        from app.services.inventory_service import inventory_service

        now = datetime.now(tz=timezone.utc)
        day_ago = now - timedelta(hours=24)
        query = RangeQuery(range_start=day_ago, range_end=now, store_id=store_id, camera_id=camera_id)

        def _load(session) -> dict[str, Any]:
            from app.models import Store

            store = session.get(Store, store_id) if store_id else session.query(Store).first()
            recent = (
                session.query(RetailEvent)
                .order_by(RetailEvent.occurred_at.desc())
                .limit(25)
                .all()
            )
            return {
                "store_name": store.name if store else None,
                "recent_events": [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type,
                        "timestamp": e.occurred_at,
                        "severity": e.severity,
                        "camera_id": e.camera_id,
                        "message": (e.payload or {}).get("message") or e.event_type,
                        "payload": e.payload,
                    }
                    for e in recent
                ],
                "skus_configured": session.query(func.count(Product.id)).scalar() or 0,
            }

        base = run_in_session(_load)
        camera_stats = camera_manager.summary()
        queue_rows = self.queues(query)
        return {
            "generated_at": now,
            "store_name": base["store_name"],
            "cameras": camera_stats,
            "inventory": inventory_service.summary(store_id=store_id),
            "alerts": self.alert_summary(query),
            "customers": self.footfall(query),
            "queues": queue_rows,
            "ai": camera_manager.ai_summary(),
            "recent_events": base["recent_events"],
            "skus_configured": base["skus_configured"],
        }

    # ── events ──────────────────────────────────────────────────────────────
    def recent_events(self, *, limit: int = 50, event_type: str | None = None,
                      severity: str | None = None, camera_id: str | None = None,
                      since: datetime | None = None) -> list[RetailEvent]:
        def _load(session) -> list[RetailEvent]:
            query = session.query(RetailEvent)
            if event_type:
                query = query.filter(RetailEvent.event_type == event_type)
            if severity:
                query = query.filter(RetailEvent.severity == severity)
            if camera_id:
                query = query.filter(RetailEvent.camera_id == camera_id)
            if since:
                query = query.filter(RetailEvent.occurred_at >= since)
            return list(query.order_by(RetailEvent.occurred_at.desc()).limit(limit).all())

        return run_in_session(_load)


def _zone_name(session: Any, zone_id: str) -> str:
    from app.models import CameraZone

    zone = session.get(CameraZone, zone_id)
    return zone.name if zone else zone_id


def _planogram_compliance(session: Any, shelf_id: str) -> float | None:
    from app.models import Planogram, RetailEvent

    planogram = (
        session.query(Planogram)
        .filter(Planogram.shelf_id == shelf_id, Planogram.status == "active")
        .first()
    )
    if planogram is None:
        return None
    latest = (
        session.query(RetailEvent)
        .filter(
            RetailEvent.event_type == "planogram.mismatch",
            RetailEvent.shelf_id == shelf_id,
        )
        .order_by(RetailEvent.occurred_at.desc())
        .first()
    )
    if latest is None:
        return None
    return (latest.payload or {}).get("compliance_pct")


#: Shared analytics service instance used by the API layer and report service.
analytics_service = AnalyticsService()


def camera_health_rows(session: Any, store_id: str | None = None) -> Sequence[Camera]:
    query = session.query(Camera)
    if store_id:
        query = query.filter(Camera.store_id == store_id)
    return query.all()


_ = (CameraStatus, QueueStatus)  # referenced by the API layer for enum parity