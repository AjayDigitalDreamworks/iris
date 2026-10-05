"""Visual embedding + SKU recognition.

Detection says *something is there*; recognition says *which SKU it is*.  IRIS
implements recognition as embedding retrieval against an operator-supplied
catalog of product reference images:

    crop -> embedder -> cosine similarity vs catalog -> best SKU or 'unknown'

This is a real algorithm running on real model features.  It is **not** a
fine-tuned retail classifier, so it is registered with ``status='baseline'``
and never presented as validated retail accuracy.  Swapping the embedder for
DINOv2 (README §4.1) requires no change outside this file.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.vision.models_registry import ModelSpec, detect_hardware, registry
from app.vision.types import BBox, Detection

ai_log = get_logger("ai")

UNKNOWN_SKU = "UNKNOWN"


@dataclass(slots=True)
class CatalogEmbedding:
    """One product reference image reduced to a unit-norm vector."""

    product_id: str
    sku: str
    name: str
    vector: np.ndarray
    model_key: str


@dataclass(slots=True)
class RecognitionResult:
    product_id: str | None
    sku: str | None
    similarity: float | None
    is_unknown: bool
    margin: float = 0.0
    candidate_sku: str | None = None

    @property
    def label(self) -> str:
        if self.is_unknown or not self.sku:
            return UNKNOWN_SKU
        return self.sku


class UltralyticsEmbedder:
    """Extracts a compact descriptor from an image using a YOLO backbone.

    The pre-classifier global feature vector is used because it is available
    from weights already downloaded and is a legitimate baseline descriptor.
    """

    def __init__(self, spec: ModelSpec, weights: Path | str | None) -> None:
        self.spec = spec
        self.weights = Path(weights) if weights else None
        self.device = detect_hardware().device if spec.device == "auto" else spec.device
        self.input_size = spec.input_size or 224
        self._lock = threading.Lock()
        self._model: Any = None
        self._loaded = False
        self._latency: list[float] = []

    @property
    def model_key(self) -> str:
        return self.spec.key

    def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO  # noqa: PLC0415

        if self.weights is None:
            raise FileNotFoundError("Embedder requires weights.")
        self._model = YOLO(str(self.weights))
        self._loaded = True
        ai_log.info(
            "Embedder loaded", extra={"model_id": self.spec.key, "device": self.device}
        )

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def embed(self, image: np.ndarray) -> np.ndarray | None:
        """Return a unit-norm float32 descriptor, or None when unavailable."""
        if image is None or image.size == 0 or min(image.shape[:2]) < 8:
            return None
        try:
            self.load()
        except Exception as exc:
            ai_log.error("Embedder load failed", exc_info=exc)
            return None

        prepared = self._prepare(image)
        try:
            with self._lock:
                started = time.perf_counter()
                results = self._model.embed(prepared, device=self.device, verbose=False)
                self._latency.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            ai_log.warning("Embedding failed", exc_info=exc)
            return None

        if not results:
            return None
        try:
            vector = np.asarray(results[0], dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """Square-pad to preserve aspect ratio, then letterbox to model size."""
        height, width = image.shape[:2]
        side = max(height, width)
        square = np.full((side, side, 3), 114, dtype=np.uint8)
        y0 = (side - height) // 2
        x0 = (side - width) // 2
        square[y0 : y0 + height, x0 : x0 + width] = image
        size = max(32, self.input_size)
        return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

    @property
    def mean_latency_ms(self) -> float:
        return round(float(np.mean(self._latency[-50:])), 3) if self._latency else 0.0


class ProductRecognizer:
    """Nearest-neighbour SKU retrieval against an in-memory catalog index."""

    def __init__(self, embedder: UltralyticsEmbedder, *, min_similarity: float | None = None) -> None:
        self.embedder = embedder
        self.min_similarity = (
            min_similarity if min_similarity is not None else settings.embedding_min_similarity
        )
        self._catalog: list[CatalogEmbedding] = []
        self._matrix: np.ndarray | None = None
        self._lock = threading.RLock()
        self.catalog_size = 0

    # ── index management ────────────────────────────────────────────────────
    def add(self, entry: CatalogEmbedding) -> None:
        with self._lock:
            self._catalog.append(entry)
            self._rebuild()

    def load_entries(self, entries: Sequence[CatalogEmbedding]) -> None:
        with self._lock:
            self._catalog = list(entries)
            self._rebuild()

    def clear(self) -> None:
        with self._lock:
            self._catalog.clear()
            self._matrix = None
            self.catalog_size = 0

    def _rebuild(self) -> None:
        if self._catalog:
            self._matrix = np.vstack([e.vector for e in self._catalog]).astype(np.float32)
            self.catalog_size = len(self._catalog)
        else:
            self._matrix = None
            self.catalog_size = 0

    @property
    def is_ready(self) -> bool:
        return self.catalog_size > 0

    # ── recognition ─────────────────────────────────────────────────────────
    def recognize_vector(self, vector: np.ndarray | None) -> RecognitionResult:
        """Match a pre-computed descriptor against the catalog."""
        if vector is None:
            return RecognitionResult(None, None, None, True)
        with self._lock:
            matrix = self._matrix
            catalog = list(self._catalog)
        if matrix is None or not catalog:
            return RecognitionResult(None, None, None, True)

        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return RecognitionResult(None, None, None, True)
        query = (vector / norm).astype(np.float32)
        scores = matrix @ query  # vectors are stored unit-norm
        order = np.argsort(-scores)
        best_idx = int(order[0])
        best_score = float(scores[best_idx])
        runner_up = float(scores[int(order[1])]) if len(order) > 1 else 0.0

        if best_score < self.min_similarity:
            return RecognitionResult(
                product_id=None,
                sku=None,
                similarity=round(best_score, 4),
                is_unknown=True,
                margin=round(best_score - runner_up, 4),
                candidate_sku=catalog[best_idx].sku,
            )
        match = catalog[best_idx]
        return RecognitionResult(
            product_id=match.product_id,
            sku=match.sku,
            similarity=round(best_score, 4),
            is_unknown=False,
            margin=round(best_score - runner_up, 4),
            candidate_sku=match.sku,
        )

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        return self.recognize_vector(self.embedder.embed(image))

    #: Classes that must NEVER receive a product SKU.
    NON_PRODUCT_CLASSES = frozenset({"person"})

    #: COCO classes that represent an individual item a shop actually sells.
    #: Recognition is an ALLOW-list, not a deny-list: with a deny-list, a
    #: backpack crop is happily "recognised" as the nearest catalog entry
    #: (measured on a real crowd clip: a backpack was labelled with a demo
    #: apple's SKU). Restricting retrieval to sellable item classes keeps
    #: strangers' belongings from inventing stock that is not on the shelf.
    PRODUCT_CLASSES = frozenset(
        {
            # Consumables
            "bottle", "wine glass", "cup", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            # Packaged / household goods
            "book", "vase", "toothbrush", "scissors", "clock",
            # Electronics on a shop counter
            "cell phone", "laptop", "mouse", "remote", "keyboard", "tv",
            "microwave", "oven", "toaster", "sink", "refrigerator",
            # Reusable / serveware
            "fork", "knife", "spoon",
        }
    )

    #: Extra product classes coming from a CUSTOM detector. When an operator
    #: supplies a fine-tuned retail model (e.g. classes named "chips",
    #: "namkeen"), those names are not in the COCO allow-list above, so without
    #: this the recognizer would skip every crop a custom model produces.
    custom_classes: set[str] = set()

    def set_custom_classes(self, names: Sequence[str]) -> None:
        """Register class names from a custom (non-COCO) detector as products."""
        self.custom_classes = {
            str(name) for name in names if name and name not in self.NON_PRODUCT_CLASSES
        }

    def annotate(self, detections: list[Detection], image: np.ndarray) -> list[Detection]:
        """Attach SKU candidates to detections using cropped embeddings."""
        from app.vision.detector import UltralyticsDetector

        for detection in detections:
            if detection.class_name in self.NON_PRODUCT_CLASSES:
                detection.extra["recognition"] = {"state": "skipped", "reason": "not_a_product"}
                continue
            if (
                detection.class_name not in self.PRODUCT_CLASSES
                and detection.class_name not in self.custom_classes
            ):
                # Measured evidence for the allow-list: on a real crowd clip,
                # backpack/handbag/suitcase crops were being assigned catalog
                # SKUs. An unrecognised scene is reported as such, never as
                # a product.
                detection.extra["recognition"] = {
                    "state": "skipped",
                    "reason": "class_not_a_product",
                }
                continue
            crop = UltralyticsDetector.crop(image, detection.bbox)
            if crop.size == 0:
                continue
            vector = self.embedder.embed(crop)
            detection.embedding = vector
            result = self.recognize_vector(vector)
            if result.is_unknown:
                detection.extra["recognition"] = {
                    "state": "unknown",
                    "best_similarity": result.similarity,
                    "candidate_sku": result.candidate_sku,
                    "min_similarity": self.min_similarity,
                }
                continue
            detection.product_id = result.product_id
            detection.sku = result.sku
            detection.similarity = result.similarity
            detection.extra["recognition"] = {
                "state": "matched",
                "similarity": result.similarity,
                "margin": result.margin,
                "min_similarity": self.min_similarity,
            }
        return detections

    def record_metrics(self) -> None:
        registry.record_metrics(
            self.embedder.model_key,
            {
                "device": self.embedder.device,
                "latency_ms_mean": self.embedder.mean_latency_ms,
                "catalog_size": self.catalog_size,
                "min_similarity": self.min_similarity,
            },
        )


def build_entry(
    product_id: str, sku: str, name: str, vector: np.ndarray, model_key: str
) -> CatalogEmbedding:
    """Normalise and wrap a catalog vector."""
    norm = float(np.linalg.norm(vector))
    return CatalogEmbedding(
        product_id=product_id,
        sku=sku,
        name=name,
        vector=(vector / norm).astype(np.float32) if norm else vector.astype(np.float32),
        model_key=model_key,
    )


def crop_for_recognition(image: np.ndarray, box: BBox, pad: float = 0.05) -> np.ndarray:
    from app.vision.detector import UltralyticsDetector

    return UltralyticsDetector.crop(image, box, pad=pad)