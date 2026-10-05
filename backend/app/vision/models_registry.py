"""Model registry and hardware detection.

Every model IRIS can run is described by a :class:`ModelSpec` and registered
here, so a model can be swapped by changing configuration rather than code.
The registry is the single place that knows:

* what models exist, their task, version, labels and confidence threshold
* where the weights live and how to obtain them
* which acceleration provider to use on this machine
* whether the model is a ``baseline`` (pretrained, unvalidated for this store)
  or ``validated`` (measured against store ground truth)

A model is never described as more capable than it has been measured to be.
"""

from __future__ import annotations

import functools
import hashlib
import platform
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

ai_log = get_logger("ai")


# ── Hardware detection ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class HardwareProfile:
    """What acceleration this machine actually offers."""

    device: str
    provider: str
    cuda_available: bool = False
    cuda_device_name: str | None = None
    onnx_available: bool = False
    onnx_providers: tuple[str, ...] = ()
    torch_version: str | None = None
    opencv_version: str | None = None
    cpu_count: int = 0
    platform: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@functools.lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Probe CUDA / ONNX / CPU support once per process."""
    torch_version = None
    cuda = False
    cuda_name = None
    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        if cuda:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - torch always present here
        ai_log.warning("PyTorch probe failed", exc_info=exc)

    providers: tuple[str, ...] = ()
    onnx_ok = False
    try:
        import onnxruntime as ort  # noqa: PLC0415

        providers = tuple(ort.get_available_providers())
        onnx_ok = "CPUExecutionProvider" in providers
    except Exception:  # pragma: no cover - optional dependency
        onnx_ok = False

    try:
        import cv2  # noqa: PLC0415

        opencv_version = cv2.__version__
    except Exception:  # pragma: no cover
        opencv_version = None

    requested = (settings.device or "auto").lower()
    if requested == "cuda" and not cuda:
        device = "cpu"
    elif requested in {"cpu", "cuda", "mps", "auto"}:
        device = "cuda" if (requested == "cuda" or (requested == "auto" and cuda)) else "cpu"
    else:
        device = requested

    provider = "onnxruntime" if (onnx_ok and device == "cpu") else "pytorch"

    profile = HardwareProfile(
        device=device,
        provider=provider,
        cuda_available=cuda,
        cuda_device_name=cuda_name,
        onnx_available=onnx_ok,
        onnx_providers=providers,
        torch_version=torch_version,
        opencv_version=opencv_version,
        cpu_count=_cpu_count(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )
    ai_log.info(
        "Hardware profile detected",
        extra={"device": profile.device, "provider": profile.provider, "cuda": profile.cuda_available},
    )
    return profile


def _cpu_count() -> int:
    import os

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 0


# ── Model specification ──────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    """Everything IRIS knows about one model artefact."""

    key: str
    name: str
    task: str                     # detect | classify | segment | embed | track | anomaly
    version: str
    weights: str | None                 # file name inside the model dir, or absolute
    source: str                         # provenance / URL
    provider: str = "ultralytics"
    status: str = "baseline"            # baseline | trained | validated | deprecated
    input_size: int = 640
    confidence_threshold: float = 0.25
    classes: tuple[str, ...] = ()
    device: str = "auto"
    is_pretrained: bool = True
    notes: str = ""
    validation_note: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def resolve_weights(self, model_dir: Path | None = None) -> Path | None:
        base = model_dir or settings.resolved_model_dir
        if not self.weights:
            return None
        candidate = Path(self.weights)
        if candidate.is_absolute():
            return candidate if candidate.exists() else None
        return (base / candidate) if (base / candidate).exists() else None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["classes"] = list(self.classes)
        return data


COCO_PRODUCT_CLASSES = (
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "book", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

PERSON_CLASSES = ("person",)


def _default_specs() -> dict[str, ModelSpec]:
    return {
        "product_detector": ModelSpec(
            key="product_detector",
            name=f"YOLO11 COCO detector ({Path(settings.detector_weights).stem})",
            task="detect",
            version=f"{Path(settings.detector_weights).stem}-coco",
            weights=settings.detector_weights,
            source=f"Ultralytics pretrained {settings.detector_weights} (COCO, 80 classes)",
            provider="ultralytics",
            status="baseline",
            input_size=settings.detector_imgsz,
            confidence_threshold=settings.confidence_threshold,
            classes=COCO_PRODUCT_CLASSES,
            notes=(
                "COCO-pretrained general detector. Detects real objects including "
                "'person'; it is NOT fine-tuned for this store's SKUs."
            ),
            validation_note=(
                "BASELINE ONLY. No retail accuracy claim. A SKU-110K / store-annotated "
                "fine-tune is required before retail metrics can be published."
            ),
        ),
        "retail_product_detector": ModelSpec(
            key="retail_product_detector",
            name="Retail product detector (multipack-precedence combination)",
            task="detect",
            version="retail-combine-1",
            weights=settings.detector_weights,
            source=(
                "Ultralytics YOLO11 general detector + the combination rule from "
                "Retail-product-detection-main (multipack takes precedence)"
            ),
            provider="ultralytics",
            status="baseline",
            input_size=settings.detector_imgsz,
            confidence_threshold=settings.confidence_threshold,
            classes=COCO_PRODUCT_CLASSES,
            notes=(
                "Two-stage: a general product detector, plus an optional multipack "
                "detector whose boxes take precedence. The reference repository's "
                "trained weights are on an external drive and are NOT present here, "
                "so the multipack stage is disabled unless "
                "IRIS_RETAIL_MULTIPACK_WEIGHTS points at a real file."
            ),
            validation_note=(
                "BASELINE. The reference project reports mAP@0.5 = 89% for ITS OWN "
                "SKU-110K fine-tune; that number is not a property of this system "
                "and is not claimed. The combination RULE is verified by unit tests; "
                "retail accuracy is not."
            ),
        ),
        "person_detector": ModelSpec(
            key="person_detector",
            name="YOLO11s person detector",
            task="detect",
            version="yolo11s-coco",
            weights=settings.person_weights,
            source="Ultralytics pretrained yolo11s.pt (COCO class 0 = person)",
            provider="ultralytics",
            status="baseline",
            input_size=settings.detector_imgsz,
            confidence_threshold=settings.confidence_threshold,
            classes=PERSON_CLASSES,
            notes=(
                "COCO mAP50-95 47.0 vs 39.5 for yolo11n, at roughly 2x the CPU cost. "
                "Chosen for person work because footfall counting fails when shoppers "
                "are missed."
            ),
            validation_note="BASELINE. Footfall accuracy must be validated against manual counts.",
        ),
        "shelf_segmenter": ModelSpec(
            key="shelf_segmenter",
            name="Shelf occupancy baseline (geometric ROI + foreground void analysis)",
            task="segment",
            version="roi-void-1",
            weights=None,
            source="IRIS geometric shelf ROI + OpenCV foreground/void estimation",
            provider="opencv",
            status="baseline",
            input_size=0,
            confidence_threshold=0.0,
            classes=("shelf", "product", "void"),
            notes=(
                "Uses the configured shelf polygon plus per-slot foreground analysis. "
                "Occupancy is an AREA ratio, not an item count."
            ),
            validation_note=(
                "BASELINE. Replace with a trained segmentation model per README §4.2 for "
                "IoU/Dice metrics."
            ),
        ),
        "embedder": ModelSpec(
            key="embedder",
            name="Visual product embedder (YOLO11n penultimate features)",
            task="embed",
            version="yolo11n-embed-1",
            weights=settings.detector_weights,
            source="Ultralytics yolo11n.pt global feature vector (pre-classifier)",
            provider="ultralytics",
            status="baseline",
            input_size=224,
            confidence_threshold=0.0,
            classes=(),
            notes=(
                "Extracts a 128/256-d descriptor from a cropped product image for "
                "nearest-neighbour SKU retrieval."
            ),
            validation_note=(
                "BASELINE. README §4.1 recommends DINOv2 for retrieval quality; the "
                "interface is provider-agnostic so swapping is configuration-only."
            ),
        ),
        "tracker": ModelSpec(
            key="tracker",
            name="ByteTrack",
            task="track",
            version=settings.tracker.replace(".yaml", ""),
            weights=None,
            source="Ultralytics ByteTrack implementation (lap assignment)",
            provider="ultralytics",
            status="baseline",
            input_size=0,
            confidence_threshold=0.0,
            classes=(),
            notes="Multi-object tracking with an explicit track TTL so IDs expire.",
            validation_note="IDF1 / ID-switch metrics require annotated sequences (see docs/AI_MODELS.md).",
        ),
    }


class ModelRegistry:
    """Registry of every model IRIS can load, with lazy instantiation."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = _default_specs()
        self._lock = threading.Lock()
        self._loaded: dict[str, Any] = {}
        self._measured: dict[str, dict[str, Any]] = {}

    # ── registry API ────────────────────────────────────────────────────────
    def register(self, spec: ModelSpec) -> ModelSpec:
        with self._lock:
            self._specs[spec.key] = spec
        return spec

    def spec(self, key: str) -> ModelSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            raise KeyError(f"Unknown model key: {key!r}") from exc

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def keys(self) -> list[str]:
        return list(self._specs.keys())

    def status_report(self) -> list[dict[str, Any]]:
        """Status of every model including whether weights are present."""
        report: list[dict[str, Any]] = []
        for spec in self.specs():
            weights = spec.resolve_weights()
            data = spec.as_dict()
            data["weights_present"] = True if not spec.weights else bool(weights)
            data["resolved_weights"] = str(weights) if weights else None
            data["measured"] = self._measured.get(spec.key, {})
            data["is_loaded"] = spec.key in self._loaded
            report.append(data)
        return report

    # ── instantiation ───────────────────────────────────────────────────────
    def get(self, key: str) -> Any:
        """Return a loaded model instance, loading it on first use."""
        if key in self._loaded:
            return self._loaded[key]
        spec = self.spec(key)
        with self._lock:
            if key in self._loaded:  # double-checked locking
                return self._loaded[key]
            instance = self._construct(spec)
            self._loaded[key] = instance
        return instance

    def _construct(self, spec: ModelSpec) -> Any:
        from app.vision.detector import UltralyticsDetector
        from app.vision.embedder import UltralyticsEmbedder
        from app.vision.tracker import ByteTrackWrapper

        if spec.key == "tracker":
            return ByteTrackWrapper(spec)
        weights = spec.resolve_weights()
        if spec.weights and weights is None:
            raise FileNotFoundError(
                f"Weights for model '{spec.key}' were not found at "
                f"'{settings.resolved_model_dir / (spec.weights or '')}'. "
                "Run: python scripts/fetch_models.py"
            )
        if spec.key == "embedder":
            return UltralyticsEmbedder(spec, weights)
        return UltralyticsDetector(spec, weights)

    def record_metrics(self, key: str, metrics: dict[str, Any]) -> None:
        """Store MEASURED metrics only (never estimated)."""
        self._measured[key] = {**metrics, "measured_at": datetime.now(tz=timezone.utc).isoformat()}

    def unload(self, key: str | None = None) -> None:
        with self._lock:
            if key:
                self._loaded.pop(key, None)
            else:
                self._loaded.clear()

    def checksum(self, path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None


registry = ModelRegistry()


def available_device() -> str:
    return detect_hardware().device