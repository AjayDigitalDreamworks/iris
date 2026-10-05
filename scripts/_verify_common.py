"""Shared helpers for the IRIS verification scripts.

Every verify script reports the same shape so `run_all_tests.py` can aggregate
them without special-casing. A check either passes with real evidence or it
fails — nothing is ever reported as verified because it "should" work.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Windows consoles default to cp1252 and will crash on the box-drawing/arrow
# characters these reports use. Force UTF-8 so the output is readable everywhere.
for _stream in ("stdout", "stderr"):
    _handle = getattr(sys, _stream, None)
    if _handle is not None and hasattr(_handle, "reconfigure"):
        try:
            _handle.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - non-reconfigurable stream
            pass

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class Report:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", skipped: bool = False) -> None:
        self.checks.append(Check(name, passed, detail, skipped))

    def run(self, name: str, fn) -> None:
        """Execute ``fn``; any exception is a failure, not a crash."""
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - a failure is data, not a crash
            self.add(name, False, f"{type(exc).__name__}: {exc}")
            return
        elapsed = time.perf_counter() - started
        if isinstance(result, tuple):
            passed, detail = result
        else:
            passed, detail = bool(result), ""
        if not detail:
            detail = f"{elapsed * 1000:.0f} ms"
        self.add(name, passed, detail)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if not check.skipped)

    def render(self) -> str:
        lines = [f"\n{self.title}", "=" * len(self.title)]
        for check in self.checks:
            if check.skipped:
                mark = "-"
            elif check.passed:
                mark = "PASS"
            else:
                mark = "FAIL"
            detail = f"  {check.detail}" if check.detail else ""
            lines.append(f"  [{mark:>4}] {check.name}{detail}")
        failed = [c for c in self.checks if not c.passed and not c.skipped]
        skipped = [c for c in self.checks if c.skipped]
        lines.append(
            f"  -> {len(self.checks) - len(failed) - len(skipped)} passed, "
            f"{len(failed)} failed, {len(skipped)} skipped"
        )
        return "\n".join(lines)

    def finish(self) -> int:
        print(self.render())
        return 0 if self.passed else 1


def backend_python() -> str:
    """Path to the backend interpreter, preferring the project virtualenv."""
    for candidate in (BACKEND_DIR / ".venv" / "Scripts" / "python.exe",
                      BACKEND_DIR / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def in_backend_python(code: str) -> tuple[bool, str]:
    """Run a snippet inside the backend environment and capture its output."""
    import subprocess

    result = subprocess.run(
        [backend_python(), "-c", code],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_DIR),
        timeout=300,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def backend_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("IRIS_MODEL_DIR", str(REPO_ROOT / "models"))
    env.setdefault("IRIS_DATA_DIR", str(REPO_ROOT / "data"))
    return env