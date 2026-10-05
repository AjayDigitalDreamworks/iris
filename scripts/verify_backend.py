"""Verify the IRIS backend: environment, imports, database, migrations, tests."""

from __future__ import annotations

import subprocess
import sys

from _verify_common import (
    BACKEND_DIR,
    REPO_ROOT,
    Report,
    backend_env,
    backend_python,
    in_backend_python,
)


def main() -> int:
    report = Report("IRIS — Backend verification")

    report.run(
        "Python interpreter available",
        lambda: (True, backend_python()),
    )
    report.run(
        "Core dependencies importable",
        lambda: in_backend_python(
            "import fastapi, sqlalchemy, pydantic, cv2, numpy, alembic; print('deps ok')"
        ),
    )
    report.run(
        "AI dependencies importable",
        lambda: in_backend_python(
            "import torch, ultralytics; print('ultralytics', ultralytics.__version__, "
            "'cuda', torch.cuda.is_available())"
        ),
    )
    report.run(
        "Application imports and route table builds",
        lambda: in_backend_python(
            "from app.main import app; "
            "paths=[p for p in app.openapi()['paths'] if p.startswith('/api/v1')]; "
            "print(len(paths), 'API paths'); "
            "assert len(paths) >= 50, 'expected the full v1 surface'"
        ),
    )
    report.run(
        "Model registry resolves weights",
        lambda: in_backend_python(
            "from app.vision.models_registry import registry; "
            "entries=registry.status_report(); "
            "missing=[e['key'] for e in entries if not e.get('weights_present')]; "
            "print('models:', len(entries), 'missing weights:', missing); "
            "assert entries, 'registry is empty'; "
            "assert not missing, 'models without weights: ' + str(missing)"
        ),
    )

    def run_tests() -> tuple[bool, str]:
        result = subprocess.run(
            [backend_python(), "-m", "pytest", "tests/", "-q"],
            capture_output=True,
            text=True,
            cwd=str(BACKEND_DIR),
            env=backend_env(),
            timeout=900,
        )
        tail = (result.stdout or "").strip().splitlines()[-1:] or [""]
        return result.returncode == 0, tail[0]

    report.run("Backend test suite", run_tests)

    def migrations() -> tuple[bool, str]:
        code = (
            "from app.db.base import Base; import app.models; from app.db.session import engine; "
            "n=len(Base.metadata.tables); print(n, 'ORM tables'); assert n >= 28"
        )
        return in_backend_python(code)

    report.run("ORM schema complete", migrations)

    report.add(
        "Repository layout",
        (REPO_ROOT / "backend" / "app" / "main.py").exists()
        and (REPO_ROOT / "scripts" / "seed.py").exists(),
        "backend/, scripts/, models/, data/, docs/",
    )

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())