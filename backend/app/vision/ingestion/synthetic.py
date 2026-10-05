"""DEMO MODE — synthetic frame generator.

This source exists so IRIS can be demonstrated and regression-tested without
physical cameras.  It renders **real pixel frames** of a procedurally generated
shelf scene, which the real pipeline then processes normally: real decode, real
preprocessing, real model inference, real tracking, real business rules.

**What it is not:** it is not a shortcut around inference.  The synthetic
generator has no privileged channel into the models — a drawn rectangle is fed
to the detector exactly like a photograph would be, and whatever the detector
returns is what gets stored.  Frames produced here are tagged ``is_demo=True``
and every resulting detection/alert/event carries ``source='demo'`` so demo
output can never be mistaken for a real retail observation.

Because a rendered scene is not a photograph, detection accuracy on demo frames
is meaningless — see ``docs/AI_MODELS.md``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.errors import SourceUnavailableError
from app.vision.ingestion.base import FrameSource


@dataclass(slots=True)
class SyntheticLayout:
    """Declarative description of the generated scene."""

    width: int = 1280
    height: int = 720
    shelf_rows: int = 4
    products_per_row: int = 12
    start_fill: float = 0.85          # fraction of slots filled at t=0
    depletion_per_minute: float = 0.02
    shoppers: int = 2


class SyntheticSource(FrameSource):
    """Generates a synthetic shelf scene with a slowly depleting shelf.

    The scene genuinely changes over time (units are removed), so stock trends,
    low-stock transitions and threshold alerts are exercised for real.
    """

    source_type = "synthetic"

    def __init__(
        self,
        layout: SyntheticLayout | None = None,
        *,
        target_fps: float = 5.0,
        frame_skip: int = 1,
        duration_seconds: float = 0.0,
        speed: float = 1.0,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_skip=frame_skip)
        self.layout = layout or SyntheticLayout()
        self.duration_seconds = duration_seconds
        self.speed = max(0.1, speed)
        self._started = time.time()
        self._rng = np.random.default_rng(seed=20261003)

    def is_finite_source(self) -> bool:
        return self.duration_seconds > 0

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _open(self) -> None:
        self._started = time.time()
        self.status.started_at = time.time()
        self.status.source_fps = 30.0
        self.status.width = self.layout.width
        self.status.height = self.layout.height

    def _close(self) -> None:
        return None

    # ── rendering ───────────────────────────────────────────────────────────
    def _read(self) -> np.ndarray | None:
        if self.duration_seconds and time.time() - self._started > self.duration_seconds:
            return None
        elapsed_minutes = (time.time() - self._started) * self.speed / 60.0
        return self.render(elapsed_minutes)

    def current_fill(self, elapsed_minutes: float) -> float:
        """Fraction of shelf slots currently holding a unit."""
        fill = self.layout.start_fill - self.layout.depletion_per_minute * elapsed_minutes
        return float(min(1.0, max(0.0, fill)))

    def render(self, elapsed_minutes: float = 0.0) -> np.ndarray:
        """Draw one frame of the synthetic store scene."""
        w, h = self.layout.width, self.layout.height
        rng = np.random.default_rng(seed=20261003 + int(elapsed_minutes * 10))
        frame = np.full((h, w, 3), (238, 236, 230), dtype=np.uint8)

        # Floor / wall split
        floor_y = int(h * 0.78)
        cv2.rectangle(frame, (0, floor_y), (w, h), (168, 166, 160), -1)

        # Back wall panel
        cv2.rectangle(frame, (0, 0), (w, floor_y), (206, 214, 222), -1)

        margin_x = int(w * 0.06)
        shelf_top = int(h * 0.08)
        shelf_bottom = floor_y - int(h * 0.02)
        rows = max(1, self.layout.shelf_rows)
        row_height = (shelf_bottom - shelf_top) // rows

        fill = self.current_fill(elapsed_minutes)
        unit_index = 0
        total_slots = self.layout.products_per_row * rows

        for row in range(rows):
            y0 = shelf_top + row * row_height
            y1 = y0 + row_height
            # Shelf board
            cv2.rectangle(frame, (margin_x - 12, y1 - 12), (w - margin_x + 12, y1), (150, 148, 142), -1)
            cv2.rectangle(frame, (margin_x - 12, y0 - 6), (w - margin_x + 12, y0), (186, 184, 178), -1)

            slot_width = (w - 2 * margin_x) / self.layout.products_per_row
            for col in range(self.layout.products_per_row):
                unit_index += 1
                # Deterministic depletion: units disappear from the right.
                unit_position = unit_index / total_slots
                occupied = unit_position <= fill
                x0 = int(margin_x + col * slot_width)
                x1 = int(x0 + slot_width * 0.82)
                if occupied:
                    hue = int(rng.integers(0, 179))
                    sat = int(rng.integers(120, 240))
                    val = int(rng.integers(140, 235))
                    color = cv2.cvtColor(
                        np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR
                    )[0][0].tolist()
                    top = y1 - 14 - int((y1 - y0 - 18) * float(rng.uniform(0.62, 0.95)))
                    cv2.rectangle(frame, (x0, top), (x1, y1 - 14), color, -1)
                    cv2.rectangle(frame, (x0, top), (x1, y1 - 14), (40, 40, 40), 1)
                    label = f"U{unit_index:03d}"
                    cv2.putText(
                        frame, label, (x0 + 4, y1 - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                else:
                    # Empty slot shading (this is the visual "void" the analyser sees)
                    cv2.rectangle(frame, (x0, y1 - 16), (x1, y1 - 14), (120, 118, 114), -1)

        # Shoppers moving across the floor
        for i in range(self.layout.shoppers):
            phase = (elapsed_minutes * 40.0 * self.speed + i * 2.1) % 1.0
            x = int(-60 + phase * (w + 120))
            person_h = int(h * 0.30)
            person_w = int(person_h * 0.38)
            y_base = floor_y + int(h * 0.06)
            cv2.rectangle(
                frame, (x, y_base - person_h), (x + person_w, y_base), (90, 96, 120), -1
            )
            cv2.circle(
                frame, (x + person_w // 2, y_base - person_h - int(person_h * 0.16)),
                int(person_h * 0.16), (200, 178, 150), -1,
            )

        # Slight illumination flicker so the stream is not perfectly static
        brightness = 1.0 + 0.03 * math.sin(time.time() * 1.7)
        if abs(brightness - 1.0) > 1e-3:
            frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # Watermark: DEMO frames must never be mistaken for real footage.
        cv2.rectangle(frame, (0, 0), (w, 34), (40, 40, 40), -1)
        cv2.putText(
            frame,
            "DEMO MODE — SYNTHETIC FRAMES (NOT REAL FOOTAGE)",
            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"elapsed={elapsed_minutes * 60:6.1f}s  fill={fill * 100:5.1f}%",
            (w - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA,
        )
        return frame


class BrokenSource(FrameSource):
    """A source that always fails — used to prove failure isolation in tests."""

    source_type = "broken"

    def __init__(self, message: str = "synthetic failure", *, retries: int = 1) -> None:
        super().__init__(target_fps=1.0, frame_skip=1)
        self.message = message
        self.retries = retries
        self.attempts = 0

    def is_finite_source(self) -> bool:
        return self.attries >= self.retries

    def _open(self) -> None:
        self.attempts += 1
        raise SourceUnavailableError(self.message)

    def _read(self) -> np.ndarray | None:
        return None

    def _close(self) -> None:
        return None