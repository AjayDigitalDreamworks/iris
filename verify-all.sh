#!/usr/bin/env bash
# IRIS — run every verification suite and exit non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")"
PY="backend/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="backend/.venv/bin/python"
exec "$PY" scripts/run_all_tests.py "$@"
