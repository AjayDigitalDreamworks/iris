"""Verify visual anomaly detection end-to-end on a REAL video file.

This runs the actual pipeline (real shelf ROI analysis, real person detection)
over a real clip, then replays a controlled shelf condition to show that:

1. nothing is flagged while the detector is still learning the baseline;
2. a steady shelf produces no candidates at all;
3. a sustained, changed shelf condition does produce one, with a severity,
   a score and the measured context that produced it;
4. the evidence image is written and is a readable JPEG.

The step-3 condition is *injected* on purpose. A single real clip cannot prove
the detector fires on real shelf emptying, because emptying a real shelf needs
a real shop. Injecting a controlled change and detecting it is a real test of
the real code path; it is not evidence of retail accuracy, and nothing here
claims that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from _verify_common import REPO_ROOT, Report  # noqa: E402
from app.core.errors import SourceUnavailableError  # noqa: E402
from app.vision.anomaly import ShelfAnomalyDetector, save_evidence  # noqa: E402


def _clip() -> Path | None:
    """Prefer real footage the operator added; fall back to the demo clip."""
    user_clips = sorted((REPO_ROOT / "data" / "sample_videos").glob("*.mp4"))
    for clip in user_clips:
        if not clip.name.startswith("pan_"):
            return clip
    return user_clips[0] if user_clips else None


def _real_roi(clip: Path, frame_index: int = 30) -> np.ndarray | None:
    """A real shelf-sized crop taken from actual footage."""
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        # A shelf band across the middle of the frame.
        y1, y2 = int(height * 0.35), int(height * 0.75)
        x1, x2 = int(width * 0.15), int(width * 0.85)
        return frame[y1:y2, x1:x2]
    finally:
        capture.release()


def main() -> int:
    report = Report("IRIS - Visual anomaly detection verification")

    clip = _clip()
    if clip is None:
        report.add(
            "Real footage available",
            False,
            "no .mp4 in data/sample_videos - drop a real clip there to verify",
            skipped=True,
        )
        return report.finish()
    report.add("Real footage available", True, clip.name)

    roi = _real_roi(clip)
    if roi is None or roi.size == 0:
        report.add(
            "Real shelf ROI extracted",
            False,
            f"could not decode a frame from {clip.name}",
            skipped=True,
        )
        return report.finish()
    report.add("Real shelf ROI extracted", True, f"{roi.shape[1]}x{roi.shape[0]} from real footage")

    # ── 1 + 2: calibration and steady state ─────────────────────────────────
    detector = ShelfAnomalyDetector(baseline_observations=8, persistence_required=3)
    during_calibration = [
        detector.update("verify-shelf", roi, occupancy_pct=70.0, detected_units=12)
        for _ in range(7)
    ]
    report.add(
        "No candidate before the baseline exists",
        all(c is None for c in during_calibration),
        f"{sum(c is not None for c in during_calibration)} candidates in 7 warm-up observations",
    )

    # Vary occupancy slightly, as a real shelf measurement does.
    for index in range(10):
        detector.update(
            "verify-shelf", roi, occupancy_pct=70.0 + (index % 3), detected_units=12
        )
    report.add("Calibrated after 8 observations", detector.is_calibrated("verify-shelf"), "baseline established")

    steady = [
        detector.update(
            "verify-shelf", roi, occupancy_pct=70.0 + (i % 3), detected_units=12
        )
        for i in range(12)
    ]
    report.add(
        "A steady shelf raises nothing",
        all(c is None for c in steady),
        f"{sum(c is not None for c in steady)} candidates over 12 steady observations",
    )

    # ── 3: a sustained change must be caught ────────────────────────────────
    # The shelf region is emptied: same geometry, the goods removed.
    emptied = np.full_like(roi, 28)
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 10, size=emptied.shape, dtype=np.uint8)
    emptied = np.clip(emptied.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    emitted = []
    for _ in range(5):
        candidate = detector.update(
            "verify-shelf", emptied, occupancy_pct=6.0, detected_units=1
        )
        if candidate is not None:
            emitted.append(candidate)

    report.add(
        "A single changed frame is filtered out",
        emitted == [] or emitted[0].consecutive >= 3,
        "persistence requirement enforced before any candidate is emitted",
    )
    report.add(
        "A sustained change is flagged",
        bool(emitted),
        f"{len(emitted)} candidates once the change persisted",
    )

    if emitted:
        best = emitted[0]
        detail = (
            f"{best.kind.value} severity={best.severity.value} "
            f"score={best.score} consecutive={best.consecutive}"
        )
        report.add("Candidate carries severity and score", 0.0 < best.score <= 1.0, detail)
        report.add(
            "Candidate carries measured context",
            bool(best.context),
            f"context keys: {sorted(best.context)[:5]}",
        )

        # ── 4: evidence ─────────────────────────────────────────────────────
        path = save_evidence(str(REPO_ROOT / "data" / "evidence"), best, emptied)
        readable = bool(path) and Path(path).exists() and cv2.imread(path) is not None
        report.add(
            "Evidence image written and readable",
            readable,
            Path(path).name if path else "no evidence file",
        )
    else:
        report.add(
            "Evidence image written and readable",
            False,
            "no candidate was produced, so no evidence was expected",
        )

    # ── honesty: the module must not claim theft detection ──────────────────
    text = (REPO_ROOT / "backend" / "app" / "vision" / "anomaly.py").read_text(
        encoding="utf-8"
    )
    report.add(
        "Module states it cannot detect theft",
        "does not detect theft" in text,
        "docstring limits the claim",
    )

    try:
        from app.vision.anomaly import ShelfAnomalyDetector as _D  # noqa: F401

        report.add("Detector importable from app.vision.anomaly", True, "app.vision.anomaly")
    except SourceUnavailableError as exc:  # pragma: no cover
        report.add("Detector importable from app.vision.anomaly", False, str(exc))

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
