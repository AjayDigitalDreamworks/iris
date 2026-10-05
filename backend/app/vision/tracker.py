"""Multi-object tracking.

ByteTrack is used because it needs no appearance model (so no privacy-relevant
re-identification network is loaded), is fast on CPU, and degrades gracefully
with occlusion.  Tracks are **anonymous and ephemeral**: an ID identifies a blob
of pixels over a short window, never a person, and every track expires.

Also provides the geometry helpers used by footfall (line crossing) and zone
analytics (point-in-polygon).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.vision.models_registry import ModelSpec
from app.vision.types import BBox, TrackedObject

ai_log = get_logger("ai")


@dataclass
class TrackState:
    """Server-side state for one live track ID."""

    external_id: int
    class_name: str
    first_seen: float
    last_seen: float
    bbox: BBox
    confidence: float = 0.0
    frames_seen: int = 1
    centroids: list[tuple[float, float]] = field(default_factory=list)
    crossed_lines: set[str] = field(default_factory=set)
    zone_ids: set[str] = field(default_factory=set)
    velocity: tuple[float, float] = (0.0, 0.0)

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen


class ByteTrackWrapper:
    """ByteTrack with an explicit track TTL and a stable IRIS interface."""

    def __init__(self, spec: ModelSpec, ttl_seconds: float | None = None) -> None:
        self.spec = spec
        self.tracker_config = (spec.version or "bytetrack") + ".yaml"
        self.ttl_seconds = ttl_seconds or settings.track_max_ttl_seconds
        self._tracker: Any = None
        self._lock = threading.Lock()
        self.states: dict[int, TrackState] = {}
        self.total_created = 0
        self.total_expired = 0

    @property
    def model_key(self) -> str:
        return self.spec.key

    def _ensure(self) -> Any:
        if self._tracker is None:
            from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: PLC0415

            self._tracker = BYTETracker(args=_tracker_args(self.tracker_config))
            ai_log.info("Tracker initialised", extra={"tracker": self.tracker_config})
        return self._tracker

    def update(
        self,
        detections: Sequence[Any],
        frame: Any,
        *,
        camera_id: str | None = None,
        frame_size: tuple[int, int] | None = None,
        now: float | None = None,
        class_names: dict[int, str] | None = None,
    ) -> list[TrackedObject]:
        """Feed one frame's detections and get back tracked objects.

        ``detections`` is a sequence of objects exposing ``.bbox`` and
        ``.confidence`` (IRIS :class:`Detection`). ByteTrack is fed through an
        Ultralytics ``Boxes`` container because that is the interface its
        high/low-confidence split expects.
        """
        import time

        timestamp = now if now is not None else time.time()
        tracker = self._ensure()
        image = getattr(frame, "image", frame)
        results: list[TrackedObject] = []

        boxes_obj = _as_boxes(detections)
        if boxes_obj is not None and len(boxes_obj):
            tracks = tracker.update(boxes_obj, img=image)
            names = dict(class_names or {})
            for row in np.asarray(tracks).reshape(-1, 8):
                box = BBox(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                external_id = int(row[4])
                confidence = float(row[5])
                class_id = int(row[6])
                class_name = names.get(class_id) or _NAME_LOOKUP.get(class_id, f"class_{class_id}")
                results.append(
                    TrackedObject(
                        track_id=external_id,
                        bbox=box,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                )
                self._upsert(external_id, class_name, box, confidence, timestamp)

        self._expire(timestamp)
        return results

    # ── internal state ──────────────────────────────────────────────────────
    def _upsert(
        self, external_id: int, class_name: str, box: BBox, confidence: float, now: float
    ) -> None:
        with self._lock:
            state = self.states.get(external_id)
            center = box.bottom_center
            if state is None:
                self.states[external_id] = TrackState(
                    external_id=external_id,
                    class_name=class_name,
                    first_seen=now,
                    last_seen=now,
                    bbox=box,
                    confidence=confidence,
                    centroids=[center],
                )
                self.total_created += 1
                return
            previous = state.centroids[-1] if state.centroids else center
            state.velocity = (center[0] - previous[0], center[1] - previous[1])
            state.last_seen = now
            state.bbox = box
            state.confidence = confidence
            state.frames_seen += 1
            state.centroids.append(center)
            # Bounded trajectory memory — privacy: we do not keep long histories.
            if len(state.centroids) > 256:
                del state.centroids[:-256]

    def _expire(self, now: float) -> None:
        with self._lock:
            stale = [
                key for key, state in self.states.items()
                if now - state.last_seen > self.ttl_seconds
            ]
            for key in stale:
                del self.states[key]
                self.total_expired += 1

    def get_state(self, external_id: int) -> TrackState | None:
        return self.states.get(external_id)

    def active_count(self) -> int:
        return len(self.states)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "track_id": state.external_id,
                    "class_name": state.class_name,
                    "age_seconds": round(state.age_seconds, 2),
                    "frames_seen": state.frames_seen,
                    "confidence": round(state.confidence, 3),
                    "zone_ids": sorted(state.zone_ids),
                }
                for state in self.states.values()
            ]

    def reset(self) -> None:
        with self._lock:
            self.states.clear()
        self._tracker = None


_NAME_LOOKUP: dict[int, str] = {0: "person"}


def _tracker_args(config: str):
    """Build the namespace Ultralytics' BYTETracker expects."""
    from types import SimpleNamespace

    from app.core.config import settings as cfg

    return SimpleNamespace(
        tracker_type="bytetrack",
        tracker_path=config,
        track_high_thresh=min(0.30, cfg.confidence_threshold),
        track_low_thresh=0.10,
        new_track_thresh=cfg.confidence_threshold,
        track_buffer=60,
        match_thresh=0.8,
        fuse_score=True,
        mot20=False,
    )


def _as_boxes(detections: Sequence[Any]) -> Any:
    """Wrap IRIS detections in the Ultralytics ``Boxes`` container ByteTrack wants."""
    if not detections:
        return None
    try:
        import torch  # noqa: PLC0415
        from ultralytics.engine.results import Boxes  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - ultralytics always present here
        ai_log.warning("Tracker unavailable", exc_info=exc)
        return None

    if not detections:
        return None

    rows: list[list[float]] = []
    for det in detections:
        box = getattr(det, "bbox", None)
        if box is None:
            continue
        # Boxes expects rows of [x1, y1, x2, y2, (track_id,) conf, cls].
        rows.append(
            [
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                float(getattr(det, "confidence", 0.0)),
                float(getattr(det, "class_id", 0)),
            ]
        )
    if not rows:
        return None
    array = torch.tensor(rows, dtype=torch.float32)
    max_x = max(row[2] for row in rows)
    max_y = max(row[3] for row in rows)
    return Boxes(array, orig_shape=(int(max_y) + 1, int(max_x) + 1))


# ── Geometry helpers shared by footfall / zone analytics ─────────────────────
def point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test in pixel coordinates.

    Only the *strict* interior counts.  A point that lies exactly on an edge or
    a vertex is reported as outside: for retail zones an ambiguous boundary hit
    must not be silently counted as presence (queue length, occupancy), so the
    caller is forced to treat it as unknown rather than as a yes.
    """
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    if _on_boundary((x, y), polygon):
        return False
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _on_boundary(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """True when the point sits on (or within a pixel of) the polygon outline."""
    x, y = point
    tolerance = 1e-9
    count = len(polygon)
    j = count - 1
    for i in range(count):
        x1, y1 = float(polygon[j][0]), float(polygon[j][1])
        x2, y2 = float(polygon[i][0]), float(polygon[i][1])
        dx, dy = x2 - x1, y2 - y1
        cross = dx * (y - y1) - dy * (x - x1)
        if abs(cross) > tolerance * max(1.0, abs(dx) + abs(dy)):
            j = i
            continue
        # Collinear: check the point lies within the segment's bounding box.
        if (min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance) and (
            min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
        ):
            return True
        j = i
    return False


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Shoelace area (absolute)."""
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        total += (xj + xi) * (yi - yj)
        j = i
    return abs(total) / 2.0


def bbox_centroid(box: BBox) -> tuple[float, float]:
    return box.center


def bbox_intersection_ratio(box: BBox, polygon: Sequence[Sequence[float]]) -> float:
    """Fraction of the box's corners inside the polygon (cheap overlap proxy)."""
    xs = (box.x1, box.x2, (box.x1 + box.x2) / 2)
    ys = (box.y1, box.y2, (box.y1 + box.y2) / 2)
    points = [(x, y) for x in xs for y in ys]
    inside = sum(1 for p in points if point_in_polygon(p, polygon))
    return inside / len(points)


def line_side(point: tuple[float, float], line: Sequence[Sequence[float]]) -> float:
    """Signed side of ``point`` relative to the infinite line through ``line``.

    Sign tells which side the point is on; ``0.0`` means it lies exactly on the
    line, which is a genuinely ambiguous frame for crossing detection and is
    reported as such rather than guessed.
    """
    if len(line) < 2:
        return 0.0
    x1, y1 = float(line[0][0]), float(line[0][1])
    x2, y2 = float(line[1][0]), float(line[1][1])
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return dx * (float(point[1]) - y1) - dy * (float(point[0]) - x1)


def line_crossed(
    previous: tuple[float, float], current: tuple[float, float], line: Sequence[Sequence[float]]
) -> tuple[bool, str | None]:
    """Test segment ``previous -> current`` against an infinite line.

    Returns ``(crossed, direction)`` where direction is ``'+'``/``'-'`` based on
    which side of the line the motion went.  A point lying exactly on the line
    yields ``(False, None)`` — callers must remember the last point that had a
    definite side (see :func:`line_side`); dropping that history would silently
    lose real crossings whenever a track lands on the line for a frame.
    """
    if len(line) < 2:
        return False, None
    side_prev = line_side(previous, line)
    side_curr = line_side(current, line)
    if side_prev == 0.0 or side_curr == 0.0:
        return False, None
    if (side_prev > 0) == (side_curr > 0):
        return False, None
    return True, "+" if side_curr > 0 else "-"


def scaled_polygon(
    polygon: Iterable[Sequence[float]], width: int, height: int
) -> list[list[float]]:
    """Convert a normalised polygon into pixel coordinates."""
    return [[float(p[0]) * width, float(p[1]) * height] for p in polygon]


def scaled_line(line: Sequence[Sequence[float]] | None, width: int, height: int):
    if not line or len(line) < 2:
        return None
    return [[float(p[0]) * width, float(p[1]) * height] for p in line[:2]]