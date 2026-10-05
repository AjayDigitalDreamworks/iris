"""Report generation with CSV / JSON export."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.services.analytics_service import RangeQuery, analytics_service

logger = get_logger("app")

REPORT_TYPES = {"inventory", "footfall", "queue", "alerts", "shelves", "detections", "anomalies"}

#: Formats every export supports.
REPORT_FORMATS = {"csv", "json", "pdf"}


def _default_range() -> tuple[datetime, datetime]:
    now = datetime.now(tz=timezone.utc)
    return now - timedelta(days=7), now


def generate(
    report_type: str,
    *,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
    output_format: str = "csv",
    camera_id: str | None = None,
    shelf_id: str | None = None,
    store_id: str | None = None,
) -> dict[str, Any]:
    """Build a report and write it to the exports directory."""
    if report_type not in REPORT_TYPES:
        raise ValidationError(f"Unknown report type {report_type!r}.")
    if output_format not in REPORT_FORMATS:
        raise ValidationError(
            f"Unknown report format {output_format!r}. Use one of: "
            + ", ".join(sorted(REPORT_FORMATS))
        )
    start, end = _default_range()
    range_start = range_start or start
    range_end = range_end or end
    if range_end <= range_start:
        raise ValidationError("range_end must be after range_start.")

    query = RangeQuery(
        range_start=range_start,
        range_end=range_end,
        camera_id=camera_id,
        shelf_id=shelf_id,
        store_id=store_id,
    )

    if report_type == "inventory":
        rows = _inventory_rows(query)
    elif report_type == "footfall":
        rows = _footfall_rows(query)
    elif report_type == "queue":
        rows = _queue_rows(query)
    elif report_type == "alerts":
        rows = _alert_rows(query)
    elif report_type == "anomalies":
        rows = _anomaly_rows(query)
    elif report_type == "shelves":
        rows = _shelf_rows(query)
    else:
        rows = _detection_rows(query)

    export_dir = settings.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{report_type}_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{output_format}"
    path = export_dir / filename
    if output_format == "csv":
        path.write_text(_to_csv(rows), encoding="utf-8")
    elif output_format == "pdf":
        from app.services.pdf_export import render_report_pdf

        render_report_pdf(
            str(path),
            report_type=report_type,
            rows=rows,
            store_name=_store_name(query),
            range_start=range_start,
            range_end=range_end,
        )
    else:
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    logger.info(
        "Report generated",
        extra={"report_type": report_type, "rows": len(rows), "source_filename": filename},
    )
    return {
        "report_type": report_type,
        "format": output_format,
        "filename": filename,
        "row_count": len(rows),
        "generated_at": datetime.now(tz=timezone.utc),
        "download_path": f"exports/{filename}",
        "_path": str(path),
    }


def _to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _flatten(v) for k, v in row.items()})
    return buffer.getvalue()


def _flatten(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


def _inventory_rows(query: RangeQuery) -> list[dict[str, Any]]:
    from app.models import InventorySnapshot

    from app.services.db_service import run_in_session

    def _load(session) -> list[dict[str, Any]]:
        rows = (
            session.query(InventorySnapshot)
            .filter(
                InventorySnapshot.observed_at >= query.range_start,
                InventorySnapshot.observed_at <= query.range_end,
            )
            .order_by(InventorySnapshot.observed_at.asc())
            .all()
        )
        if query.camera_id:
            rows = [r for r in rows if r.camera_id == query.camera_id]
        if query.shelf_id:
            rows = [r for r in rows if r.shelf_id == query.shelf_id]
        return [
            {
                "observed_at": r.observed_at,
                "sku": r.sku,
                "observed_quantity": r.observed_quantity,
                "estimated_quantity": round(r.estimated_quantity, 3),
                "capacity_units": r.capacity_units,
                "stock_percentage": round(r.stock_percentage, 2),
                "confidence": round(r.confidence, 4),
                "shelf_id": r.shelf_id,
                "camera_id": r.camera_id,
                "source": r.source,
                "method": r.method,
            }
            for r in rows
        ]

    return run_in_session(_load)


def _footfall_rows(query: RangeQuery) -> list[dict[str, Any]]:
    data = analytics_service.footfall(query)
    return [
        {
            "bucket": point["bucket"],
            "entries": point["entries"],
            "exits": point["exits"],
            "net": point["net"],
        }
        for point in data["points"]
    ]


def _queue_rows(query: RangeQuery) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for zone in analytics_service.queues(query):
        for point in zone["points"]:
            rows.append(
                {
                    "bucket": point["bucket"],
                    "zone_id": zone["zone_id"],
                    "zone_name": zone["zone_name"],
                    "camera_id": zone["camera_id"],
                    "queue_length": point["queue_length"],
                    "estimated_wait_seconds": point["estimated_wait_seconds"],
                    "status": point["status"],
                }
            )
    return rows


def _store_name(query: RangeQuery) -> str | None:
    """Store label for the PDF header; None when the range spans many stores."""
    from app.models import Store
    from app.services.db_service import run_in_session

    def _load(session: Any) -> str | None:
        if query.store_id:
            store = session.get(Store, query.store_id)
            return store.name if store else None
        names = {
            s.name for s in session.query(Store).all()
        }
        return names.pop() if len(names) == 1 else None

    try:
        return run_in_session(_load)
    except Exception:  # pragma: no cover - a missing label must not fail the export
        return None


def _anomaly_rows(query: RangeQuery) -> list[dict[str, Any]]:
    """Shelf anomaly candidates raised in the range.

    Anomalies are stored as alerts of type ``anomaly.candidate``; the measured
    context lives in the alert payload, so it is flattened into real columns
    rather than left as an opaque JSON blob in a spreadsheet.
    """
    from app.models import Alert, AlertType
    from app.services.db_service import run_in_session

    def _load(session: Any) -> list[dict[str, Any]]:
        query_set = (
            session.query(Alert)
            .filter(
                Alert.alert_type == AlertType.ANOMALY_CANDIDATE.value,
                Alert.created_at >= query.range_start,
                Alert.created_at <= query.range_end,
            )
            .order_by(Alert.created_at.desc())
        )
        if query.camera_id:
            query_set = query_set.filter(Alert.camera_id == query.camera_id)
        if query.shelf_id:
            query_set = query_set.filter(Alert.shelf_id == query.shelf_id)
        if query.store_id:
            query_set = query_set.filter(Alert.store_id == query.store_id)
        rows = []
        for alert in query_set.limit(5000):
            payload = alert.payload or {}
            context = payload.get("context") or {}
            rows.append(
                {
                    "created_at": alert.created_at,
                    "kind": payload.get("kind"),
                    "severity": alert.severity,
                    "score": payload.get("score"),
                    "confidence": alert.confidence,
                    "observations": alert.observation_count,
                    "camera_id": alert.camera_id,
                    "shelf_id": alert.shelf_id,
                    "status": alert.status,
                    "evidence_path": alert.evidence_path,
                    "message": alert.message,
                    "occupancy_pct": context.get("occupancy_pct"),
                    "baseline_occupancy_pct": context.get("baseline_occupancy_pct"),
                }
            )
        return rows

    return run_in_session(_load)


def _alert_rows(query: RangeQuery) -> list[dict[str, Any]]:
    from app.models import Alert

    from app.services.db_service import run_in_session

    def _load(session) -> list[dict[str, Any]]:
        rows = (
            session.query(Alert)
            .filter(Alert.created_at >= query.range_start, Alert.created_at <= query.range_end)
            .order_by(Alert.created_at.asc())
            .all()
        )
        return [
            {
                "id": r.id,
                "created_at": r.created_at,
                "alert_type": r.alert_type,
                "severity": r.severity,
                "status": r.status,
                "title": r.title,
                "sku": r.sku,
                "camera_id": r.camera_id,
                "shelf_id": r.shelf_id,
                "confidence": r.confidence,
                "occurrence_count": r.occurrence_count,
                "acknowledged_by": r.acknowledged_by,
                "resolved_by": r.resolved_by,
                "resolution_note": r.resolution_note,
            }
            for r in rows
        ]

    return run_in_session(_load)


def _shelf_rows(query: RangeQuery) -> list[dict[str, Any]]:
    return analytics_service.shelves(query, within_minutes=15 * 24 * 60)


def _detection_rows(query: RangeQuery) -> list[dict[str, Any]]:
    from app.models import Detection

    from app.services.db_service import run_in_session

    def _load(session) -> list[dict[str, Any]]:
        rows = (
            session.query(Detection)
            .filter(
                Detection.frame_timestamp >= query.range_start,
                Detection.frame_timestamp <= query.range_end,
            )
            .order_by(Detection.frame_timestamp.asc())
            .limit(50_000)
            .all()
        )
        if query.camera_id:
            rows = [r for r in rows if r.camera_id == query.camera_id]
        return [
            {
                "frame_timestamp": r.frame_timestamp,
                "camera_id": r.camera_id,
                "class_name": r.class_name,
                "sku": r.sku,
                "similarity": r.similarity,
                "confidence": round(r.confidence, 4),
                "track_id": r.track_id,
                "bbox": [r.x1, r.y1, r.x2, r.y2],
                "model_id": r.model_id,
                "source": r.source,
            }
            for r in rows
        ]

    return run_in_session(_load)


def resolve_export(relative: str) -> Path | None:
    """Resolve an export path safely, refusing traversal."""
    root = settings.resolve_path(settings.data_dir).resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents or not candidate.exists():
        return None
    return candidate


__all__ = ["generate", "resolve_export", "REPORT_TYPES", "analytics_service"]