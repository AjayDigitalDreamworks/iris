"""Object detector wrapper.

Wraps an Ultralytics YOLO model behind a stable interface so the rest of IRIS
never depends on the framework's output format.  Also owns preprocessing so
letterboxing and coordinate mapping are correct — a classic source of silently
wrong bounding boxes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.vision.models_registry import ModelSpec, detect_hardware, registry
from app.vision.types import BBox, Detection, Frame

ai_log = get_logger("ai")


class UltralyticsDetector:
    """A YOLO-family object detector with IRIS-native inputs and outputs.

    Instances are shared across cameras: the weights are loaded once and calls
    are serialised with a lock (PyTorch models are not re-entrant on a single
    thread pool without care).
    """

    def __init__(self, spec: ModelSpec, weights: Path | str | None) -> None:
        self.spec = spec
        self.weights = Path(weights) if weights else None
        self.device = detect_hardware().device if spec.device == "auto" else spec.device
        self.imgsz = spec.input_size or settings.detector_imgsz
        self.confidence = spec.confidence_threshold or settings.confidence_threshold
        self.iou = settings.iou_threshold
        self._lock = threading.Lock()
        self._model: Any = None
        self._names: dict[int, str] = {}
        self.load_error: str | None = None
        self._latency_samples: list[float] = []

    # ── lifecycle ───────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # noqa: PLC0415

        if self.weights is None:
            raise FileNotFoundError(f"No weights configured for {self.spec.key}")
        started = time.perf_counter()
        self._model = YOLO(str(self.weights))
        self._names = {int(k): str(v) for k, v in (self._model.names or {}).items()}
        elapsed = (time.perf_counter() - started) * 1000
        ai_log.info(
            "Detector loaded",
            extra={
                "model_id": self.spec.key,
                "device": self.device,
                "load_ms": round(elapsed, 2),
                "classes": len(self._names),
                "weights": str(self.weights),
            },
        )

    @property
    def model_id(self) -> str:
        return f"{self.spec.name}@{self.spec.version}"

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self.load_error is None

    @property
    def mean_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return round(float(np.mean(self._latency_samples[-50:])), 3)

    def warmup(self, size: int | None = None) -> float:
        """Run one throwaway inference so the first real frame is not an outlier."""
        self.load()
        dummy = np.zeros((size or self.imgsz, size or self.imgsz, 3), dtype=np.uint8)
        started = time.perf_counter()
        self._raw_predict(dummy)
        return (time.perf_counter() - started) * 1000

    # ── inference ───────────────────────────────────────────────────────────
    def _raw_predict(self, image: np.ndarray):
        assert self._model is not None
        return self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
            max_det=200,
        )

    def predict(
        self,
        frame: Frame,
        *,
        class_filter: set[int] | None = None,
        class_names: set[str] | None = None,
    ) -> tuple[list[Detection], float]:
        """Run detection on a frame. Returns (detections, latency_ms).

        Never raises: a failed inference is logged and yields an empty result so
        one bad frame cannot stop a stream.
        """
        pre_started = time.perf_counter()
        try:
            frame.validate()
        except ValueError as exc:
            ai_log.warning("Invalid frame rejected by detector", extra={"error": str(exc)})
            return [], 0.0

        try:
            self.load()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            ai_log.error(
                "Detector failed to load",
                extra={"model_id": self.spec.key, "error": self.load_error},
            )
            return [], 0.0

        try:
            with self._lock:
                started = time.perf_counter()
                results = self._raw_predict(frame.image)
                latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            ai_log.exception(
                "Inference failed",
                extra={"model_id": self.spec.key, "camera_id": frame.camera_id},
            )
            return [], (time.perf_counter() - pre_started) * 1000

        self._latency_samples.append(latency_ms)
        post_started = time.perf_counter()
        detections = self._postprocess(results, frame, class_filter, class_names)
        _ = (time.perf_counter() - post_started) * 1000
        return detections, latency_ms

    def _postprocess(
        self,
        results: Any,
        frame: Frame,
        class_filter: set[int] | None,
        class_names: set[str] | None,
    ) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        # Ultralytics already applied letterboxing; boxes map back to the original frame.
        scale_x = frame.width / max(1, result.orig_shape[1])
        scale_y = frame.height / max(1, result.orig_shape[0])

        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, clss, strict=False):
            name = self._names.get(int(class_id), f"class_{int(class_id)}")
            if class_filter is not None and int(class_id) not in class_filter:
                continue
            if class_names is not None and name not in class_names:
                continue
            box = BBox(
                float(np.clip(x1 * scale_x, 0, frame.width)),
                float(np.clip(y1 * scale_y, 0, frame.height)),
                float(np.clip(x2 * scale_x, 0, frame.width)),
                float(np.clip(y2 * scale_y, 0, frame.height)),
            )
            if box.width <= 1 or box.height <= 1:
                continue  # degenerate box
            detections.append(
                Detection(bbox=box, class_id=int(class_id), class_name=name, confidence=float(conf))
            )
        return detections

    # ── crops for recognition ───────────────────────────────────────────────
    @staticmethod
    def crop(image: np.ndarray, box: BBox, pad: float = 0.05) -> np.ndarray:
        """Crop a detection with a small context pad, clamped to the image."""
        height, width = image.shape[:2]
        pad_x = box.width * pad
        pad_y = box.height * pad
        x1 = int(max(0, box.x1 - pad_x))
        y1 = int(max(0, box.y1 - pad_y))
        x2 = int(min(width, box.x2 + pad_x))
        y2 = int(min(height, box.y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            return image[0:0, 0:0]
        return image[y1:y2, x1:x2].copy()

    def record_metrics(self) -> None:
        if self._latency_samples:
            samples = self._latency_samples[-200:]
            registry.record_metrics(
                self.spec.key,
                {
                    "device": self.device,
                    "input_size": self.imgsz,
                    "samples": len(samples),
                    "latency_ms_mean": round(float(np.mean(samples)), 3),
                    "latency_ms_p50": round(float(np.percentile(samples, 50)), 3),
                    "latency_ms_p95": round(float(np.percentile(samples, 95)), 3),
                    "fps_mean": round(1000.0 / float(np.mean(samples)), 3) if samples else 0.0,
                },
            )


def letterbox(image: np.ndarray, size: int, colour: tuple[int, int, int] = (114, 114, 114)):
    """Aspect-ratio preserving resize with padding (returns image, ratio, pad)."""
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), colour, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)