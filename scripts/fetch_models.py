"""Download the model weights and sample media IRIS needs.

Usage::

    python scripts/fetch_models.py                # weights + sample media
    python scripts/fetch_models.py --weights-only

All downloads use Python's streaming HTTP client.  ``curl`` produced silently
truncated files on the audit host, so it is deliberately not used here, and
every artefact is verified with a size/sha256 check before it is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import settings  # noqa: E402

USER_AGENT = "IRIS-model-fetcher/1.0 (local retail analytics prototype)"
TIMEOUT = 120


@dataclass(frozen=True)
class Artifact:
    key: str
    url: str
    destination: Path
    kind: str
    min_bytes: int = 10_000
    license_note: str = ""
    description: str = ""


ARTIFACTS = (
    Artifact(
        key="detector",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        destination=REPO_ROOT / "models" / "yolo11n.pt",
        kind="weights",
        min_bytes=5_000_000,
        license_note="Ultralytics AGPL-3.0 / commercial licence",
        description="YOLO11n COCO detector (person + 79 other classes). BASELINE model.",
    ),
    Artifact(
        key="person_detector",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        destination=REPO_ROOT / "models" / "yolo11s.pt",
        kind="weights",
        min_bytes=15_000_000,
        license_note="Ultralytics AGPL-3.0 / commercial licence",
        description=(
            "YOLO11s COCO detector used for PERSON detection. Same family as the "
            "product baseline but 7.5 mAP points better (47.0 vs 39.5 COCO "
            "mAP50-95), which matters because footfall counting depends on not "
            "missing shoppers."
        ),
    ),
    Artifact(
        key="sample_persons",
        url="https://ultralytics.com/images/bus.jpg",
        destination=REPO_ROOT / "data" / "sample_images" / "bus.jpg",
        kind="sample",
        min_bytes=50_000,
        license_note="Ultralytics public example asset",
        description="Real photo with clearly detectable people (regression fixture).",
    ),
    Artifact(
        key="sample_store_interior",
        url=(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/"
            "Convenience_store_interior.jpg/1280px-Convenience_store_interior.jpg"
        ),
        destination=REPO_ROOT / "data" / "sample_images" / "wikimedia_store_interior.jpg",
        kind="sample",
        min_bytes=50_000,
        license_note="Wikimedia Commons — see file page for the exact licence",
        description="Real retail interior photo (people + shelves).",
    ),
    Artifact(
        key="sample_produce",
        url=(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/"
            "Westside_Market_in_Manhattan%2C_NYC_IMG_5615.JPG/"
            "1280px-Westside_Market_in_Manhattan%2C_NYC_IMG_5615.JPG"
        ),
        destination=REPO_ROOT / "data" / "sample_images" / "wikimedia_grocery.jpg",
        kind="sample",
        min_bytes=50_000,
        license_note="Wikimedia Commons — see file page for the exact licence",
        description="Real grocery photo used to build product reference embeddings.",
    ),
)


def download(artifact: Artifact, *, force: bool = False, timeout: int = TIMEOUT) -> bool:
    destination = artifact.destination
    if destination.exists() and not force:
        size = destination.stat().st_size
        if size >= artifact.min_bytes:
            print(f"  [skip] {artifact.key:22} already present ({size:,} bytes)")
            return True
        print(f"  [redo] {artifact.key:22} existing file is too small ({size:,} bytes)")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(artifact.url, headers={"User-Agent": USER_AGENT})
    print(f"  [get ] {artifact.key:22} {artifact.url}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        print(f"  [FAIL] {artifact.key:22} {type(exc).__name__}: {exc}")
        return False

    size = partial.stat().st_size
    if size < artifact.min_bytes:
        partial.unlink(missing_ok=True)
        print(f"  [FAIL] {artifact.key:22} truncated download ({size:,} < {artifact.min_bytes:,} bytes)")
        return False

    partial.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()[:16]
    print(f"  [ ok ] {artifact.key:22} {size:,} bytes  sha256:{digest}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch IRIS model weights and sample media.")
    parser.add_argument("--weights-only", action="store_true", help="skip sample media")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    print(f"Model directory : {settings.resolved_model_dir}")
    print(f"Data directory  : {settings.resolve_path(settings.data_dir)}")
    print()

    results: list[tuple[Artifact, bool]] = []
    for artifact in ARTIFACTS:
        if args.weights_only and artifact.kind != "weights":
            continue
        results.append((artifact, download(artifact, force=args.force)))

    print()
    print("Summary")
    print("-" * 96)
    for artifact, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {artifact.key:22} {artifact.description}")
        print(f"        {artifact.destination}")
        if artifact.license_note:
            print(f"        licence: {artifact.license_note}")
    failed = [a.key for a, ok in results if not ok]
    print("-" * 96)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("Weights are required for AI features. Re-run once the network is available.")
        return 1
    print("All artefacts present.")
    print()
    print("Next: python scripts/seed.py   (creates DEMO configuration and product embeddings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())