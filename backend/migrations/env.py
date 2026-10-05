"""Alembic migration environment for IRIS.

The database URL and target metadata come from the application itself so that
migrations and the running service can never disagree.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make ``app`` importable when alembic is invoked from the backend directory.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.base import Base  # noqa: E402
import app.models  # noqa: F401,E402  (registers every mapped class)
from app.db.session import DATABASE_URL  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_: str, obj: object, autogen_context) -> str | bool:
    """Render IRIS custom column types using portable stock SQLAlchemy types.

    Without this, autogenerate would emit ``app.db.base.JSONEncoded()`` into a
    migration file that does not import ``app``.
    """
    from sqlalchemy import JSON, DateTime

    from app.db.base import JSONEncoded, UTCDateTime

    if type_ == "type":
        if isinstance(obj, JSONEncoded):
            autogen_context.imports.add("import sqlalchemy as sa")
            return "sa.JSON()"
        if isinstance(obj, UTCDateTime):
            autogen_context.imports.add("import sqlalchemy as sa")
            return "sa.DateTime()"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_item=render_item,
        render_as_batch=DATABASE_URL.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_item=render_item,
            # SQLite cannot ALTER most things in place: batch mode rebuilds tables.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()