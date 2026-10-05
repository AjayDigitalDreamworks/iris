"""Customer analytics: entry/exit counting, occupancy, dwell time.

Privacy by design (README §10.2):

* No facial recognition, no re-identification, no appearance embeddings.
* Track IDs are ephemeral and expire after a short TTL.
* Only aggregate counts and duration statistics are persisted by default.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.vision.tracker import (
    TrackState,
    line_crossed,
    line_side,
    point_in_polygon,
    scaled_line,
)
from app.vision.types import Frame, TrackedObject


@dataclass
class CrossingEvent:
    """A single direction-aware line crossing."""

    zone_id: str
    track_uid: str
    external_id: int
    event_type: str      # 'in' | 'out'
    direction: str       # '+' | '-'
    x: float
    y: float
    confidence: float
    occurred_at: float


@dataclass
class CustomerAnalytics:
    """Aggregate, per-camera customer metrics for one moment in time."""

    camera_id: str | None
    active_tracks: int = 0
    total_entries: int = 0
    total_exits: int = 0
    current_occupancy: int = 0
    crossings: list[CrossingEvent] = field(default_factory=list)
    dwell_seconds: list[float] = field(default_factory=list)
    zone_occupancy: dict[str, int] = field(default_factory=dict)
    mean_dwell_seconds: float | None = None
    confidence: float = 0.0
    note: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "active_tracks": self.active_tracks,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "current_occupancy": self.current_occupancy,
            "zone_occupancy": self.zone_occupancy,
            "mean_dwell_seconds": self.mean_dwell_seconds,
            "crossings": len(self.crossings),
        }


class CustomerAnalyticsEngine:
    """Tracks line crossings and dwell for one camera.

    Duplicate suppression is per (zone, track) — a person jitters across a line
    for several frames but is only counted once until they cross back.
    """

    def __init__(self, camera_id: str | None, *, occupancy_zones: Sequence[dict] | None = None) -> None:
        self.camera_id = camera_id
        self.lines: list[dict[str, Any]] = []
        self.occupancy_zones: list[dict[str, Any]] = list(occupancy_zones or [])
        self.entries = 0
        self.exits = 0
        self._dwell_history: deque[float] = deque(maxlen=500)
        self._pending: dict[tuple[str, int], tuple[float, float]] = {}

    # ── configuration ───────────────────────────────────────────────────────
    def configure(
        self,
        lines: Iterable[dict[str, Any]] | None = None,
        occupancy_zones: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        if lines is not None:
            self.lines = list(lines)
        if occupancy_zones is not None:
            self.occupancy_zones = list(occupancy_zones)

    def reset(self) -> None:
        self.entries = 0
        self.exits = 0
        self._dwell_history.clear()
        self._pending.clear()

    # ── per-frame update ────────────────────────────────────────────────────
    def update(
        self,
        frame: Frame,
        tracks: list[TrackedObject],
        states: dict[int, TrackState],
        *,
        now: float | None = None,
    ) -> CustomerAnalytics:
        timestamp = now if now is not None else time.time()
        crossings: list[CrossingEvent] = []
        zone_counts: dict[str, int] = {z["zone_id"]: 0 for z in self.occupancy_zones}
        seen_external: set[int] = set()

        for track in tracks:
            if track.class_name != "person":
                continue
            seen_external.add(track.track_id)
            state = states.get(track.track_id)
            foot = track.bbox.bottom_center

            for zone in self.occupancy_zones:
                polygon = _to_pixels(zone.get("polygon") or [], frame.width, frame.height)
                if len(polygon) >= 3 and point_in_polygon(foot, polygon):
                    zone_counts[zone["zone_id"]] = zone_counts.get(zone["zone_id"], 0) + 1
                    if state is not None:
                        state.zone_ids.add(zone["zone_id"])

            for line in self.lines:
                line_px = scaled_line(line.get("line"), frame.width, frame.height)
                if line_px is None:
                    continue
                key = (line["zone_id"], track.track_id)
                previous = self._pending.get(key)
                # A foot point landing exactly on the line has no definite side.
                # Keep the last point that did, so the crossing is not lost when
                # a track happens to land on the line for one frame.
                side_curr = line_side(foot, line_px)
                if side_curr == 0.0:
                    continue
                self._pending[key] = foot
                if previous is None:
                    continue
                crossed, direction = line_crossed(previous, foot, line_px)
                if not crossed:
                    continue
                if state is not None and key in state.crossed_lines:
                    continue  # duplicate suppression for this (zone, track)
                if state is not None:
                    state.crossed_lines.add(key)
                event_type = self._direction_to_event(line, direction)
                if event_type == "in":
                    self.entries += 1
                elif event_type == "out":
                    self.exits += 1
                crossings.append(
                    CrossingEvent(
                        zone_id=line["zone_id"],
                        track_uid=f"{self.camera_id}:{track.track_id}",
                        external_id=track.track_id,
                        event_type=event_type,
                        direction=direction or "",
                        x=float(foot[0]),
                        y=float(foot[1]),
                        confidence=track.confidence,
                        occurred_at=timestamp,
                    )
                )

        # Dwell statistics from tracks that have just expired.
        for external_id, state in list(states.items()):
            if external_id not in seen_external and state.age_seconds > 0:
                if not state.crossed_lines:
                    self._dwell_history.append(state.age_seconds)
                    state.dwell_seconds = state.age_seconds

        return CustomerAnalytics(
            camera_id=self.camera_id,
            active_tracks=len(seen_external),
            total_entries=self.entries,
            total_exits=self.exits,
            current_occupancy=len(seen_external),
            crossings=crossings,
            dwell_seconds=list(self._dwell_history),
            zone_occupancy=zone_counts,
            mean_dwell_seconds=(
                round(sum(self._dwell_history) / len(self._dwell_history), 2)
                if self._dwell_history
                else None
            ),
            confidence=round(
                min(1.0, len(seen_external) * 0.2 + (0.6 if tracks else 0.0)), 4
            ),
            note="Counts are anonymous aggregate metrics from temporary track IDs.",
        )

    def _direction_to_event(self, line: dict[str, Any], direction: str | None) -> str:
        """Map a crossing direction to 'in'/'out' using the configured polarity."""
        if direction is None:
            return "in"
        positive_side = str(line.get("positive_side", "+"))
        is_positive = direction == "+"
        declared_positive_is_entry = line.get("entry_on_positive", True)
        if is_positive == declared_positive_is_entry:
            return "in"
        _ = positive_side
        return "out"


def _to_pixels(polygon: Sequence[Sequence[float]], width: int, height: int) -> list[list[float]]:
    return [[float(p[0]) * width, float(p[1]) * height] for p in polygon]


def summarise_dwell(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None, "p90": None, "count": 0}
    import numpy as np

    return {
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "max": round(float(np.max(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "count": len(values),
    }