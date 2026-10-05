"""Verify the database: schema, indexes, constraints, and a full migration cycle."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _verify_common import BACKEND_DIR, Report, backend_python

SCHEMA_CHECK = """
import app.models  # noqa: F401  (registers every mapper)
from app.db.base import Base
tables = Base.metadata.tables
print(len(tables), "tables")
indexes = sum(len(t.indexes) for t in tables.values())
print(indexes, "indexes")
assert len(tables) >= 28, "expected the full IRIS schema"
"""


def main() -> int:
    report = Report("IRIS — Database verification")
    scratch = Path(tempfile.gettempdir()) / "iris_verify_migrations.db"

    def migration_cycle() -> tuple[bool, str]:
        if scratch.exists():
            scratch.unlink()
        url = f"sqlite:///{scratch.as_posix()}"
        steps: list[str] = []
        for args, label in (
            (["upgrade", "head"], "upgrade"),
            (["check"], "check"),
            (["downgrade", "base"], "downgrade"),
            (["upgrade", "head"], "re-upgrade"),
        ):
            result = subprocess.run(
                [backend_python(), "-m", "alembic", *args],
                capture_output=True,
                text=True,
                cwd=str(BACKEND_DIR),
                env={**__import__("os").environ, "IRIS_DATABASE_URL": url},
                timeout=600,
            )
            if result.returncode != 0:
                return False, f"{label} failed: {(result.stdout + result.stderr)[-400:]}"
            steps.append(label)

        import sqlite3

        connection = sqlite3.connect(scratch)
        count = connection.execute(
            "select count(*) from sqlite_master where type='table'"
        ).fetchone()[0]
        connection.close()
        assert count >= 28, f"expected >=28 tables after upgrade, found {count}"
        return True, f"{' > '.join(steps)}, {count} tables"

    report.run("ORM schema + indexes", lambda: _schema())
    report.run("Migration upgrade → check → downgrade → upgrade", migration_cycle)

    report.add(
        "Migrations directory present",
        (BACKEND_DIR / "migrations" / "env.py").exists(),
        "backend/migrations",
    )
    if scratch.exists():
        scratch.unlink()
    return report.finish()


def _schema() -> tuple[bool, str]:
    from _verify_common import in_backend_python

    return in_backend_python(SCHEMA_CHECK)


if __name__ == "__main__":
    sys.exit(main())