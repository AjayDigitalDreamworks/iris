"""Queue and checkout analytics.

Queue length = number of tracked people whose foot point falls inside a
configured queue zone.  Waiting time is an **estimate** from a queueing model
with stated assumptions; it is never presented as a guaranteed wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.vision.rules import QueueStatus, QueueThresholds, estimate_wait_seconds, queue_status
from app.vision.tracker import point_in_polygon
from app.vision.types import Frame, TrackedObject


@dataclass
class QueueMeasurement:
    """One queue-zone reading."""

    zone_id: str
    zone_name: str
    camera_id: str | None
    queue_length: int
    service_zone_occupancy: int
    service_stations: int
    estimated_wait_seconds: float | None
    status: QueueStatus
    confidence: float
    busy_threshold: int
    critical_threshold: int
    note: str = ""
    per_track: list[dict[str, Any]] = field(default_factory=list)

    def as_payload(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "camera_id": self.camera_id,
            "queue_length": self.queue_length,
            "service_zone_occupancy": self.service_zone_occupancy,
            "service_stations": self.service_stations,
            "estimated_wait_seconds": self.estimated_wait_seconds,
            "status": self.status.value,
            "confidence": self.confidence,
            "busy_threshold": self.busy_threshold,
            "critical_threshold": self.critical_threshold,
            "note": self.note,
        }


class QueueAnalyzer:
    """Computes queue metrics for every configured queue zone of a camera."""

    def __init__(
        self,
        camera_id: str | None,
        thresholds: QueueThresholds | None = None,
        *,
        zones: Sequence[dict[str, Any]] | None = None,
        confidence_floor: float = 0.35,
    ) -> None:
        self.camera_id = camera_id
        self.thresholds = thresholds or QueueThresholds()
        self.zones = list(zones or [])
        self.confidence_floor = confidence_floor

    def configure(self, zones: Sequence[dict[str, Any]] | None = None) -> None:
        if zones is not None:
            self.zones = list(zones)

    def analyse(self, frame: Frame, tracks: list[TrackedObject]) -> list[QueueMeasurement]:
        people = [t for t in tracks if t.class_name == "person"]
        measurements: list[QueueMeasurement] = []
        for zone in self.zones:
            if zone.get("zone_type") != "queue":
                continue
            polygon = [
                [float(p[0]) * frame.width, float(p[1]) * frame.height]
                for p in (zone.get("polygon") or [])
            ]
            if len(polygon) < 3:
                continue
            in_queue: list[TrackedObject] = []
            in_service: list[TrackedObject] = []
            for track in people:
                if point_in_polygon(track.bbox.bottom_center, polygon):
                    in_queue.append(track)
                elif point_in_polygon(track.bbox.center, self._service_polygon(zone, frame)):
                    in_service.append(track)

            stations = int(zone.get("service_stations") or 1)
            length = len(in_queue)
            status = queue_status(length, self.thresholds)
            wait = (
                estimate_wait_seconds(length, self.thresholds, servers=stations)
                if length
                else 0.0
            )
            confident = [t for t in in_queue if t.confidence >= self.confidence_floor]
            confidence = round(
                min(1.0, (len(confident) / length) if length else 0.5) * 0.9 + 0.1, 4
            )
            measurements.append(
                QueueMeasurement(
                    zone_id=zone["zone_id"],
                    zone_name=zone.get("name") or zone["zone_id"],
                    camera_id=self.camera_id,
                    queue_length=length,
                    service_zone_occupancy=len(in_service),
                    service_stations=stations,
                    estimated_wait_seconds=round(wait, 1),
                    status=status,
                    confidence=confidence,
                    busy_threshold=self.thresholds.busy,
                    critical_threshold=self.thresholds.critical,
                    note=(
                        "Queue length counts tracked people inside the configured zone. "
                        f"Waiting time is an ESTIMATE = counted people x "
                        f"{self.thresholds.service_time_seconds:.0f}s mean service time divided by "
                        f"{stations} service station(s); it is not a measured or promised wait."
                    ),
                    per_track=[
                        {
                            "track_id": t.track_id,
                            "bbox": t.bbox.as_list(),
                            "confidence": round(t.confidence, 3),
                        }
                        for t in in_queue
                    ],
                )
            )
        return measurements

    @staticmethod
    def _service_polygon(zone: dict[str, Any], frame: Frame) -> list[list[float]]:
        polygon = zone.get("service_polygon")
        if not polygon:
            return []
        return [
            [float(p[0]) * frame.width, float(p[1]) * frame.height] for p in polygon
        ]


def pool_measurements(measurements: Sequence[QueueMeasurement]) -> QueueMeasurement | None:
    """Median-pool several frames of one zone into a single reading."""
    if not measurements:
        return None
    import numpy as np

    ordered = sorted(measurements, key=lambda m: m.zone_id)
    queue_length = int(round(float(np.median([m.queue_length for m in ordered]))))
    service = int(round(float(np.median([m.service_zone_occupancy for m in ordered]))))
    stations = ordered[0].service_stations
    thresholds = QueueThresholds(
        busy=ordered[0].busy_threshold,
        critical=ordered[0].critical_threshold,
    )
    return QueueMeasurement(
        zone_id=ordered[0].zone_id,
        zone_name=ordered[0].zone_name,
        camera_id=ordered[0].camera_id,
        queue_length=queue_length,
        service_zone_occupancy=service,
        service_stations=stations,
        estimated_wait_seconds=ordered[0].estimated_wait_seconds,
        status=queue_status(queue_length, thresholds),
        confidence=round(float(np.median([m.confidence for m in ordered])), 4),
        busy_threshold=thresholds.busy,
        critical_threshold=thresholds.critical,
        note=ordered[0].note,
    )