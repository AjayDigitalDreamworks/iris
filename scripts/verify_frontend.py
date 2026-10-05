"""Verify the frontend: toolchain, typecheck, unit tests, and a production build."""

from __future__ import annotations

import shutil
import subprocess
import sys

from _verify_common import FRONTEND_DIR, Report


def run(args: list[str], timeout: int = 900) -> tuple[bool, str]:
    npm = shutil.which("npm")
    if npm is None:
        return False, "npm not found on PATH"
    result = subprocess.run(
        [npm, *args],
        capture_output=True,
        text=True,
        cwd=str(FRONTEND_DIR),
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or "")).strip().splitlines()
        return False, (tail[-1] if tail else "failed")
    tail = ((result.stdout or "") + (result.stderr or "")).strip().splitlines()
    return True, (tail[-1] if tail else "ok")


def main() -> int:
    report = Report("IRIS — Frontend verification")

    report.add(
        "Frontend project present",
        (FRONTEND_DIR / "package.json").exists(),
        str(FRONTEND_DIR),
    )
    report.add(
        "node_modules installed",
        (FRONTEND_DIR / "node_modules").exists(),
        "run: npm install",
    )

    if not (FRONTEND_DIR / "node_modules").exists():
        report.add("Dependencies installed", False, "npm install has not been run", skipped=True)
        return report.finish()

    report.run("TypeScript typecheck", lambda: run(["exec", "--", "tsc", "--noEmit", "-p", "tsconfig.app.json"]))
    report.run("Component / unit tests", lambda: run(["exec", "--", "vitest", "run", "--reporter=dot"]))
    report.run("Production build", lambda: run(["run", "build"]))

    report.add(
        "Backend proxy configured",
        "proxy" in (FRONTEND_DIR / "vite.config.ts").read_text(encoding="utf-8"),
        "/api and /ws are proxied in dev",
    )
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())