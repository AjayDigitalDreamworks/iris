"""Per-task inference scheduling (README §3, Pipeline A step 5).

Sending one frame through every model on every frame is wasteful: a shelf does
not need person tracking at 10 FPS, and nobody needs shelf segmentation at 10
FPS. The scheduler decides, per task, whether *this* frame is one it should run
on.

Two design choices worth stating:

**Frame-count gating, not wall-clock.** A task runs every Nth delivered frame,
where N is derived from the camera's delivery rate and the task's target rate.
Wall-clock gating looks more natural but makes behaviour depend on how fast the
machine happens to be, which makes it untestable and non-reproducible. Frame
counting gives the same rates on any machine, and a test can assert them
exactly.

**These are rates, not performance guarantees.** The numbers below are the
starting points from README §3. Real sustainable rates depend on the hardware,
so the effective rate is always ``min(target, what the CPU can actually
deliver)`` — and the scheduler reports what was achieved rather than claiming
the target was met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

ai_log = get_logger("ai")

# Task names. Kept as constants so a typo cannot silently create a new task that
# never runs.
DETECTION = "detection"
PERSON_TRACKING = "person_tracking"
FOOTFALL = "footfall"
QUEUE = "queue"
SHELF_ANALYSIS = "shelf_analysis"
RECOGNITION = "recognition"
PLANOGRAM = "planogram"

#: Task set for each camera role, with target rates in FPS.
#:
#: Entrance cameras follow people, so person work runs fast and product
#: recognition is pointless there. Shelf cameras care about stock, which changes
#: slowly, so segmentation and recognition run at a fraction of the rate.
#: Checkout cameras care about queue movement.
ROLE_PROFILES: dict[str, dict[str, float]] = {
    "entrance": {
        DETECTION: 10.0,
        PERSON_TRACKING: 10.0,
        FOOTFALL: 10.0,
    },
    "shelf": {
        DETECTION: 4.0,
        SHELF_ANALYSIS: 1.0,
        RECOGNITION: 1.0,
        PLANOGRAM: 0.05,
    },
    "checkout": {
        DETECTION: 8.0,
        PERSON_TRACKING: 8.0,
        QUEUE: 5.0,
    },
    "queue": {
        # A dedicated queue camera: the queue task IS the job, and queue length
        # needs person positions, so tracking runs at the same rate as queue.
        DETECTION: 8.0,
        PERSON_TRACKING: 8.0,
        QUEUE: 5.0,
    },
    "other": {
        DETECTION: 5.0,
        PERSON_TRACKING: 5.0,
        FOOTFALL: 5.0,
    },
}

#: Used when a camera's role is unknown or unset.
DEFAULT_PROFILE = ROLE_PROFILES["other"]


@dataclass
class TaskRate:
    """Target rate and observed rate for one task."""

    name: str
    target_fps: float
    frames_run: int = 0
    frames_seen: int = 0
    _last_run_frame: int = field(default=-10_000, repr=False)

    @property
    def run_every(self) -> int:
        """Run this task once every N delivered frames. 1 means every frame."""
        if self.target_fps <= 0:
            return 0  # never
        if self.delivery_fps <= 0:
            return 1
        return max(1, int(round(self.delivery_fps / self.target_fps)))

    def allowed(self, frame_index: int) -> bool:
        """True when `frame_index` is one this task should run on.

        The first frame always runs, so a clip shorter than the task's interval
        is still analysed once instead of being skipped entirely.
        """
        self.frames_seen += 1
        every = self.run_every
        if every == 0:
            return False
        if frame_index - self._last_run_frame >= every:
            self._last_run_frame = frame_index
            self.frames_run += 1
            return True
        return False

    @property
    def achieved_fps(self) -> float:
        if self.frames_seen <= 0:
            return 0.0
        return round(self.frames_run * self.delivery_fps / self.frames_seen, 3)

    delivery_fps: float = 5.0


class InferenceScheduler:
    """Decides which AI tasks run on which frame for one camera.

    Per-camera, exactly like the tracker: ByteTrack keeps internal state, so
    sharing it across cameras would mix identities.
    """

    def __init__(
        self,
        *,
        role: str = "other",
        capture_fps: float = 5.0,
        overrides: dict[str, float] | None = None,
    ) -> None:
        self.role = role if role in ROLE_PROFILES else "other"
        self.capture_fps = max(0.1, float(capture_fps))
        profile = dict(DEFAULT_PROFILE if role not in ROLE_PROFILES else ROLE_PROFILES[role])
        if overrides:
            profile.update({k: float(v) for k, v in overrides.items()})
        self.targets = profile
        self._tasks = {
            name: TaskRate(
                name=name,
                # A task can never run faster than frames are actually delivered.
                target_fps=min(fps, self.capture_fps) if fps > 0 else 0.0,
                delivery_fps=self.capture_fps,
            )
            for name, fps in profile.items()
        }
        self._frame_index = 0

    def begin_frame(self) -> int:
        """Advance to the next delivered frame and return its index."""
        self._frame_index += 1
        return self._frame_index

    def allow(self, task: str) -> bool:
        """True when `task` should run on the current frame."""
        rate = self._tasks.get(task)
        if rate is None:
            # Unknown task: never silently run something unconfigured.
            return False
        return rate.allowed(self._frame_index)

    def ensure_task(self, task: str, fps: float = 5.0) -> None:
        """Enable a task if the role profile does not already run it.

        The role decides the DEFAULT task set, but configuration can require
        more: a queue zone on an entrance camera must still be measured, or the
        queue looks configured yet never reports a number.
        """
        current = self._tasks.get(task)
        if current is not None and current.target_fps > 0:
            return
        self.set_target(task, fps)

    def target_for(self, task: str) -> float:
        rate = self._tasks.get(task)
        return rate.target_fps if rate else 0.0

    def set_target(self, task: str, fps: float) -> None:
        """Override a task's rate at runtime (Settings page)."""
        capped = min(float(fps), self.capture_fps) if fps > 0 else 0.0
        self.targets[task] = capped
        existing = self._tasks.get(task)
        if existing is None:
            self._tasks[task] = TaskRate(
                name=task, target_fps=capped, delivery_fps=self.capture_fps
            )
        else:
            existing.target_fps = capped

    def stats(self) -> dict[str, Any]:
        """What each task was asked to do, and what it actually did."""
        return {
            "role": self.role,
            "capture_fps": round(self.capture_fps, 3),
            "frames_seen": self._frame_index,
            "tasks": {
                name: {
                    "target_fps": round(rate.target_fps, 3),
                    "frames_run": rate.frames_run,
                    "frames_seen": rate.frames_seen,
                    "achieved_fps": rate.achieved_fps,
                }
                for name, rate in sorted(self._tasks.items())
            },
        }

    def reset(self) -> None:
        for rate in self._tasks.values():
            rate.frames_run = 0
            rate.frames_seen = 0
            rate._last_run_frame = -10_000
        self._frame_index = 0
