"""unique camera name per store

Camera names are how operators identify a camera in the dashboard and how the
seed/upsert paths find an existing row, so two cameras sharing a name inside one
store is a data-integrity problem.

Implemented as a unique index rather than a table constraint: SQLite enforces
unique indexes directly, whereas a constraint would force a full table rebuild
through ``batch_alter_table``. Any pre-existing duplicates are renamed first so
the migration cannot fail on a database that already contains them.

Revision ID: b7c1d2e3f4a5
Revises: 6a686da6a4bf
Create Date: 2026-10-03
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b7c1d2e3f4a5"
down_revision = "6a686da6a4bf"
branch_labels = None
depends_on = None

INDEX_NAME = "uq_cameras_store_name"


def _deduplicate() -> None:
    """Suffix any pre-existing duplicate names so the index can be created."""
    connection = op.get_bind()
    duplicates = connection.execute(
        sa.text(
            "SELECT store_id, name, COUNT(*) AS total FROM cameras "
            "GROUP BY store_id, name HAVING COUNT(*) > 1"
        )
    ).fetchall()
    for store_id, name, _total in duplicates:
        rows = connection.execute(
            sa.text(
                "SELECT id FROM cameras WHERE store_id = :store_id AND name = :name "
                "ORDER BY created_at, id"
            ),
            {"store_id": store_id, "name": name},
        ).fetchall()
        # Keep the oldest; rename the rest so each name stays addressable.
        for (camera_id,) in rows[1:]:
            new_name = f"{name} ({camera_id[:8]})"[:120]
            connection.execute(
                sa.text("UPDATE cameras SET name = :name WHERE id = :id"),
                {"name": new_name, "id": camera_id},
            )


def upgrade() -> None:
    _deduplicate()
    op.create_index(INDEX_NAME, "cameras", ["store_id", "name"], unique=True)


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="cameras")