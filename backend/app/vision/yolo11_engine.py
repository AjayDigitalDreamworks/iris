"""Ultralytics YOLO11 Real-Time Retail Detection Engine.

Provides high-FPS, production-grade inference using YOLO11 (yolo11n / yolo11s)
with automatic hardware acceleration (CUDA/FP16 -> MPS -> CPU/ONNX Runtime),
warmup routines, decoupled frame parsing, and retail multipack combination logic
ported from Retail-product-detection-main.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.vision.models_registry import detect_hardware

logger = get_logger("ai")

# Distinct palette for retail bounding boxes and overlays
CLASS_PALETTE: dict[str, str] = {
    "bottle": "#10B981",       # Emerald
    "cup": "#06B6D4",          # Cyan
    "wine glass": "#8B5CF6",   # Violet
    "bowl": "#3B82F6",         # Blue
    "banana": "#F59E0B",       # Amber
    "apple": "#EF4444",        # Red
    "sandwich": "#EC4899",     # Pink
    "orange": "#F97316",       # Orange
    "book": "#6366F1",         # Indigo
    "multipack": "#EAB308",    # Yellow/Gold
    "person": "#818CF8",       # Soft Indigo
    "default": "#14B8A6",      # Teal
}


def get_color_for_label(label: str) -> str:
    """Return a consistent hex color for a given detection label."""
    return CLASS_PALETTE.get(label.lower(), CLASS_PALETTE["default"])


@dataclass
class NormalizedDetection:
    """Structured detection object ready for client-side rendering and metrics."""

    id: str
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in normalized 0.0-100.0 percentage space
    raw_bbox: list[int]  # [xmin, ymin, xmax, ymax] in absolute pixel coordinates
    color: str
    track_id: int | None = None
    is_multipack: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(coord, 2) for coord in self.bbox],
            "raw_bbox": self.raw_bbox,
            "color": self.color,
            "track_id": self.track_id,
            "is_multipack": self.is_multipack,
        }


@dataclass
class DetectionTelemetry:
    """Real-time performance measurements for model execution."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    effective_fps: float
    device: str
    model: str
    hardware_provider: str
    cuda_available: bool
    fp16_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "effective_fps": round(self.effective_fps, 1),
            "device": self.device,
            "model": self.model,
            "hardware_provider": self.hardware_provider,
            "cuda_available": self.cuda_available,
            "fp16_enabled": self.fp16_enabled,
        }


@dataclass
class YOLO11Result:
    """Complete prediction payload including detections, counts, and latency."""

    detections: list[NormalizedDetection]
    telemetry: DetectionTelemetry
    total_items: int
    class_breakdown: dict[str, int]
    frame_width: int
    frame_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "telemetry": self.telemetry.to_dict(),
            "total_items": self.total_items,
            "class_breakdown": self.class_breakdown,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }


class YOLO11Detector:
    """Production-grade YOLO11 inference engine with hardware auto-tuning and warmup."""

    _instances: dict[str, "YOLO11Detector"] = {}
    _init_lock = threading.Lock()

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        model_size: str = "yolo11n",
        device: str = "auto",
        imgsz: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.model_size = model_size
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self._lock = threading.Lock()
        self._model: Any = None
        self._names: dict[int, str] = {}
        self._latency_history: list[float] = []

        # Determine target weights path
        if weights_path:
            self.weights_path = Path(weights_path)
        else:
            resolved_dir = settings.resolved_model_dir
            candidate = resolved_dir / f"{model_size}.pt"
            self.weights_path = candidate if candidate.exists() else Path(f"{model_size}.pt")

        # Hardware probing
        hw = detect_hardware()
        if device == "auto":
            self.device = "cuda" if hw.cuda_available else "cpu"
        else:
            self.device = device

        self.cuda_available = hw.cuda_available
        self.fp16 = bool(self.cuda_available and self.device.startswith("cuda"))
        self.hardware_provider = hw.provider

        logger.info(
            "YOLO11Detector initialized",
            extra={
                "weights": str(self.weights_path),
                "device": self.device,
                "fp16": self.fp16,
                "imgsz": self.imgsz,
            },
        )

    @classmethod
    def get_instance(cls, model_size: str = "yolo11n", imgsz: int = 640) -> "YOLO11Detector":
        """Singleton factory pattern to share loaded model weights across requests."""
        key = f"{model_size}:{imgsz}"
        if key not in cls._instances:
            with cls._init_lock:
                if key not in cls._instances:
                    instance = cls(model_size=model_size, imgsz=imgsz)
                    instance.load()
                    instance.warmup()
                    cls._instances[key] = instance
        return cls._instances[key]

    def load(self) -> None:
        """Load the Ultralytics model weights into memory."""
        if self._model is not None:
            return

        from ultralytics import YOLO  # noqa: PLC0415

        start_time = time.perf_counter()
        self._model = YOLO(str(self.weights_path))
        self._names = {int(k): str(v) for k, v in (self._model.names or {}).items()}
        load_duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "YOLO11 model weights loaded",
            extra={
                "path": str(self.weights_path),
                "classes": len(self._names),
                "load_ms": round(load_duration_ms, 2),
                "device": self.device,
            },
        )

    def warmup(self, passes: int = 1) -> float:
        """Run throwaway dummy inference passes to eliminate cold-start latency."""
        self.load()
        dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        start_time = time.perf_counter()

        predict_kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
            "max_det": 20,
        }
        if self.fp16:
            predict_kwargs["half"] = True

        with self._lock:
            for _ in range(passes):
                self._model.predict(dummy_frame, **predict_kwargs)


        warmup_duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "YOLO11 warmup completed",
            extra={"passes": passes, "warmup_ms": round(warmup_duration_ms, 2)},
        )
        return warmup_duration_ms

    def predict(
        self,
        image: np.ndarray,
        *,
        conf: float | None = None,
        filter_classes: Sequence[str] | None = None,
    ) -> YOLO11Result:
        """Execute high-FPS inference on a BGR image array with normalized output.

        Returns structured detections, timing breakdown, and class counters.
        """
        self.load()
        h, w = image.shape[:2]
        effective_conf = conf if conf is not None else self.confidence

        t0 = time.perf_counter()
        # Preprocessing: format check
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Input image must be a valid 3-channel numpy array")
        t_pre = (time.perf_counter() - t0) * 1000

        # Model Inference
        t1 = time.perf_counter()
        predict_kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": effective_conf,
            "iou": self.iou,
            "device": self.device,
            "verbose": False,
            "max_det": 300,
        }
        if self.fp16:
            predict_kwargs["half"] = True

        with self._lock:
            results = self._model.predict(image, **predict_kwargs)
        t_infer = (time.perf_counter() - t1) * 1000

        # Postprocessing: coordinate normalization and breakdown
        t2 = time.perf_counter()
        parsed_detections: list[NormalizedDetection] = []
        class_breakdown: dict[str, int] = {}

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_indices = boxes.cls.cpu().numpy().astype(int)

                for idx, (coords, score, cls_id) in enumerate(zip(xyxy, confs, cls_indices)):
                    label = self._names.get(cls_id, f"item_{cls_id}")

                    if filter_classes and label not in filter_classes:
                        continue

                    xmin, ymin, xmax, ymax = coords.tolist()
                    xmin_clamped = max(0, min(w, int(xmin)))
                    ymin_clamped = max(0, min(h, int(ymin)))
                    xmax_clamped = max(0, min(w, int(xmax)))
                    ymax_clamped = max(0, min(h, int(ymax)))

                    # Normalized coordinates in 0-100% space
                    norm_x1 = (xmin_clamped / w) * 100.0 if w > 0 else 0.0
                    norm_y1 = (ymin_clamped / h) * 100.0 if h > 0 else 0.0
                    norm_x2 = (xmax_clamped / w) * 100.0 if w > 0 else 0.0
                    norm_y2 = (ymax_clamped / h) * 100.0 if h > 0 else 0.0

                    color = get_color_for_label(label)
                    detection_id = f"det_{int(time.time() * 1000)}_{idx}"

                    parsed_detections.append(
                        NormalizedDetection(
                            id=detection_id,
                            label=label,
                            confidence=float(score),
                            bbox=[norm_x1, norm_y1, norm_x2, norm_y2],
                            raw_bbox=[xmin_clamped, ymin_clamped, xmax_clamped, ymax_clamped],
                            color=color,
                            is_multipack=False,
                        )
                    )
                    class_breakdown[label] = class_breakdown.get(label, 0) + 1

        t_post = (time.perf_counter() - t2) * 1000
        total_time_ms = t_pre + t_infer + t_post

        # Track rolling latency for effective FPS
        self._latency_history.append(total_time_ms)
        if len(self._latency_history) > 60:
            self._latency_history.pop(0)

        mean_total = float(np.mean(self._latency_history)) if self._latency_history else total_time_ms
        effective_fps = float(1000.0 / mean_total) if mean_total > 0 else 0.0


        telemetry = DetectionTelemetry(
            preprocess_ms=t_pre,
            inference_ms=t_infer,
            postprocess_ms=t_post,
            total_ms=total_time_ms,
            effective_fps=effective_fps,
            device=self.device,
            model=f"{self.model_size}@{self.imgsz}",
            hardware_provider=self.hardware_provider,
            cuda_available=self.cuda_available,
            fp16_enabled=self.fp16,
        )

        return YOLO11Result(
            detections=parsed_detections,
            telemetry=telemetry,
            total_items=len(parsed_detections),
            class_breakdown=class_breakdown,
            frame_width=w,
            frame_height=h,
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        conf: float | None = None,
        filter_classes: Sequence[str] | None = None,
    ) -> YOLO11Result:
        """Decode image bytes and execute YOLO11 prediction."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from provided byte stream")
        return self.predict(image, conf=conf, filter_classes=filter_classes)

    @staticmethod
    def apply_multipack_precedence(
        general_detections: list[NormalizedDetection],
        multipack_detections: list[NormalizedDetection],
    ) -> tuple[list[NormalizedDetection], int]:
        """Combine general and multipack detections using the precedence rule.

        Ported from Retail-product-detection-main (combineDetect):
        Any general detection whose center point falls strictly inside a
        multipack bounding box is suppressed. Surviving general boxes are
        appended after multipack boxes.
        """
        if not multipack_detections:
            return list(general_detections), 0

        surviving: list[NormalizedDetection] = []
        suppressed_count = 0

        for gen in general_detections:
            cx = gen.bbox[0] + (gen.bbox[2] - gen.bbox[0]) / 2.0
            cy = gen.bbox[1] + (gen.bbox[3] - gen.bbox[1]) / 2.0

            inside = False
            for multi in multipack_detections:
                if (
                    multi.bbox[0] < cx < multi.bbox[2]
                    and multi.bbox[1] < cy < multi.bbox[3]
                ):
                    inside = True
                    break

            if not inside:
                surviving.append(gen)
            else:
                suppressed_count += 1

        for multi in multipack_detections:
            multi.is_multipack = True
            multi.color = CLASS_PALETTE["multipack"]

        return list(multipack_detections) + surviving, suppressed_count
