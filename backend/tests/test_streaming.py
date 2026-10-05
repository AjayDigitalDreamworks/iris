"""Regression tests for the live video stream and the Live overlay.

Each test here pins a bug that actually shipped to the dashboard:

* the MJPEG parts were delimited by ``--frame`` while the response declared
  ``boundary=irisframe``, so browsers parsed the whole body as one part and the
  feed froze on its first frame;
* ``LatestFrameBuffer.get_latest`` blocked for its full timeout on every call
  while frames were flowing, capping every stream at ~1 fps;
* ``CameraWorker.stop`` closed the frame buffer, and ``restart`` reused the same
  worker, so a restarted camera streamed nothing forever;
* the Live overlay read the stored detection history (many frames at once),
  which made the model look like it detected far more people than were visible;
* ``/analytics/detections`` returns a SUMMARY, not rows, so the overlay's boxes
  were silently always empty.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from app.api.v1 import streams as streams_module
from app.services.camera_manager import (
    MJPEG_BOUNDARY,
    CameraWorker,
    build_multipart_frame,
)
from app.vision.ingestion.base import LatestFrameBuffer
from app.vision.types import BBox, Detection, Frame
import numpy as np


def _frame(value: int = 0) -> Frame:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    return Frame(image=image, index=1, timestamp=time.time())


# ── MJPEG framing ────────────────────────────────────────────────────────────
def test_multipart_boundary_matches_the_declared_boundary():
    """The part delimiter MUST equal the boundary in the Content-Type header."""
    body = build_multipart_frame(b"\xff\xd8jpegdata\xff\xd9")
    assert body.startswith(f"--{MJPEG_BOUNDARY}\r\n".encode())
    # A browser divides the body by the DECLARED boundary. If the two differ it
    # never finds a second part and the image freezes.
    assert streams_module.BOUNDARY == MJPEG_BOUNDARY
    assert body.count(f"--{MJPEG_BOUNDARY}".encode()) == 1
    assert b"Content-Type: image/jpeg\r\n" in body


# ── LatestFrameBuffer freshness semantics ────────────────────────────────────
def test_get_latest_returns_immediately_when_a_frame_is_buffered():
    buffer = LatestFrameBuffer(maxsize=4)
    buffer.put(_frame(1))
    started = time.perf_counter()
    frame = buffer.get_latest(timeout=1.0)
    elapsed = time.perf_counter() - started
    assert frame is not None
    # The old implementation drained for the whole timeout window, so this
    # returned after ~1.0 s and the stream could never exceed ~1 fps.
    assert elapsed < 0.25, f"get_latest blocked for {elapsed:.3f}s with data available"


def test_get_latest_is_non_destructive():
    buffer = LatestFrameBuffer(maxsize=4)
    buffer.put(_frame(5))
    first = buffer.get_latest(timeout=1.0)
    second = buffer.get_latest(timeout=1.0)
    # Two readers (MJPEG + snapshot) must both see the newest frame.
    assert first is second is not None


def test_get_latest_waits_for_the_first_frame_then_gives_up():
    buffer = LatestFrameBuffer(maxsize=4)
    started = time.perf_counter()
    assert buffer.get_latest(timeout=0.2) is None
    elapsed = time.perf_counter() - started
    assert 0.15 <= elapsed < 1.5


# ── worker restart ───────────────────────────────────────────────────────────
class _DummyThread:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        pass


def test_restarting_a_worker_reopens_the_frame_buffer():
    worker = CameraWorker("camera-xyz", manager=None)  # type: ignore[arg-type]
    worker.stop()
    assert worker.frame_buffer._closed is True
    with patch("app.services.camera_manager.threading.Thread", _DummyThread):
        assert worker.start() is True
    # stop() closes the buffer; without a fresh one a restarted camera would
    # accept no frames at all and the dashboard tile would stay blank.
    assert worker.frame_buffer._closed is False
    assert worker.frame_buffer.put(_frame(2)) is True


# ── live overlay boxes ───────────────────────────────────────────────────────
def _result(detections):
    from app.services.pipeline import PipelineResult

    return PipelineResult(
        camera_id="cam",
        frame_index=1,
        captured_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        width=100,
        height=100,
        detections=list(detections),
        latency_ms=12.0,
    )


def _detection() -> Detection:
    return Detection(bbox=BBox(1, 2, 30, 40), class_id=0, class_name="person", confidence=0.9)


def test_publish_live_keeps_last_boxes_when_detection_was_gated():
    from app.services.pipeline import PipelineRunner

    runner = object.__new__(PipelineRunner)  # bypass heavy model construction
    runner.live_detections = []  # type: ignore[attr-defined]
    runner.live_frame = {}  # type: ignore[attr-defined]

    runner.publish_live(_result([_detection()]), detection_ran=True)
    assert len(runner.live_detections) == 1
    assert runner.live_detections[0]["is_person"] is True

    # A scheduler-gated frame carries no detections; the overlay must keep the
    # previous boxes rather than flickering empty.
    runner.publish_live(_result([]), detection_ran=False)
    assert len(runner.live_detections) == 1
    assert runner.live_frame["detections_fresh"] is False


def test_publish_live_replaces_boxes_when_detection_ran():
    from app.services.pipeline import PipelineRunner

    runner = object.__new__(PipelineRunner)
    runner.live_detections = []  # type: ignore[attr-defined]
    runner.live_frame = {}  # type: ignore[attr-defined]

    runner.publish_live(_result([_detection()]), detection_ran=True)
    runner.publish_live(_result([]), detection_ran=True)
    assert runner.live_detections == []
    assert runner.live_frame["detections_fresh"] is True


# ── shelf stage / source wiring ──────────────────────────────────────────────
def test_process_shelves_accepts_people():
    """Regression: the method referenced an out-of-scope ``people`` name and
    every shelf camera crashed with NameError, storing nothing."""
    import inspect

    from app.services.pipeline import PipelineRunner

    params = list(inspect.signature(PipelineRunner._process_shelves).parameters)
    assert "people" in params, params
    assert params.index("people") < params.index("result")


def test_video_file_cameras_loop_forever():
    """A clip camera must loop; otherwise it ends every few seconds, reopens
    from frame 0 and the live feed flickers/restarts."""
    from pathlib import Path
    from types import SimpleNamespace

    from app.services.camera_manager import build_source_for_camera

    clip = Path(__file__).resolve().parents[2] / "data" / "sample_videos" / "pan_0.mp4"
    if not clip.exists():
        import pytest

        pytest.skip("sample clip missing")
    camera = SimpleNamespace(
        source_type="video_file",
        capture_fps=5,
        frame_skip=1,
        reconnect_delay=1,
        reconnect_max_delay=5,
        file_path=str(clip),
        image_path=None,
    )
    source = build_source_for_camera(camera)  # type: ignore[arg-type]
    assert source.is_finite_source() is False


# ── API contract ─────────────────────────────────────────────────────────────
def test_live_detections_endpoint_rejects_unknown_camera(client):
    response = client.get("/api/v1/streams/does-not-exist/detections")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_custom_detector_classes_become_products():
    """A fine-tuned model's own classes (e.g. "chips") are not in the COCO
    allow-list, so the recognizer must accept them explicitly."""
    from app.vision.embedder import ProductRecognizer

    recognizer = object.__new__(ProductRecognizer)
    recognizer.set_custom_classes(["chips", "namkeen", "person", ""])
    assert "chips" in recognizer.custom_classes
    assert "namkeen" in recognizer.custom_classes
    # A person is never a product, whatever the model calls it.
    assert "person" not in recognizer.custom_classes
    assert "" not in recognizer.custom_classes


def test_scheduler_enables_configured_tasks_missing_from_a_role():
    """An entrance camera has a queue zone but its role profile has no queue
    task, so queue analysis never ran and the Queues page stayed empty."""
    from app.vision import scheduler as sched

    scheduler = sched.InferenceScheduler(role="entrance", capture_fps=10.0)
    assert scheduler.target_for(sched.QUEUE) == 0.0
    scheduler.ensure_task(sched.QUEUE, 5.0)
    assert scheduler.target_for(sched.QUEUE) > 0
    # It must not clobber a rate the role already set.
    scheduler.ensure_task(sched.DETECTION, 1.0)
    assert scheduler.target_for(sched.DETECTION) == 10.0


def test_dashboard_endpoint_is_healthy(client):
    """Regression: the dashboard built InventorySummary without the required
    ``generated_at`` field, so the whole Overview page 500'd."""
    response = client.get("/api/v1/analytics/dashboard")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "inventory" in body and "generated_at" in body["inventory"]
    assert "cameras" in body and "alerts" in body


def test_capabilities_endpoint_is_healthy(client):
    """Regression: ``ai_summary`` omitted ``task`` and this 500'd."""
    response = client.get("/api/v1/inference/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["models"], "expected at least one registered model"
    assert all("task" in model for model in body["models"])


def test_telemetry_endpoint_exists(client, store):
    created = client.post(
        "/api/v1/cameras",
        json={
            "store_id": store.id,
            "name": "Telemetry Camera",
            "role": "shelf",
            "source_type": "synthetic",
            "capture_fps": 2,
            "is_enabled": False,
        },
    )
    assert created.status_code == 201, created.text
    camera_id = created.json()["id"]
    response = client.get(f"/api/v1/inference/telemetry/{camera_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == camera_id
    assert "hardware" in body and "device" in body["hardware"]
    # The Live page reads this shape; a 404 here left it on fallback values.
    assert "capture_fps" in body and "inference_fps" in body
