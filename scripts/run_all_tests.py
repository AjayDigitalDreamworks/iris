"""Run every IRIS verification script and print one honest summary.

Usage:  python scripts/run_all_tests.py            # everything
        python scripts/run_all_tests.py --fast     # skip frontend build/tests
        python scripts/run_all_tests.py --only ai  # a single suite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _verify_common import REPO_ROOT, Report, backend_python  # noqa: E402

SUITES = {
    "backend": "verify_backend.py",
    "database": "verify_database.py",
    "ai": "verify_ai.py",
    "camera": "verify_camera.py",
    "anomaly": "verify_anomaly.py",
    "retail": "verify_retail.py",
    "dataset": "verify_dataset.py",
    "api": "verify_api.py",
    "frontend": "verify_frontend.py",
}



def main() -> int:
    parser = argparse.ArgumentParser(description="Run all IRIS verification suites.")
    parser.add_argument("--only", choices=sorted(SUITES), help="run a single suite")
    parser.add_argument("--fast", action="store_true", help="skip the slower frontend suite")
    args = parser.parse_args()

    selected = [args.only] if args.only else [
        name for name in SUITES if not (args.fast and name == "frontend")
    ]

    overall = Report("IRIS — verification summary")
    started = time.perf_counter()

    for name in selected:
        script = REPO_ROOT / "scripts" / SUITES[name]
        print(f"\n>>> running {SUITES[name]}")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT / "scripts"),
            timeout=1800,
        )
        output = (result.stdout or "") + (result.stderr or "")
        print(output.rstrip())

        if name == "backend" and "Backend test suite" not in output:
            overall.add(f"suite:{name}", False, "suite produced no recognisable report")
            continue
        overall.add(
            f"suite:{name}",
            result.returncode == 0,
            "all checks passed" if result.returncode == 0 else "one or more checks FAILED",
        )

    elapsed = time.perf_counter() - started
    print(f"\nTotal time: {elapsed:.1f}s")
    return overall.finish()


if __name__ == "__main__":
    sys.exit(main())