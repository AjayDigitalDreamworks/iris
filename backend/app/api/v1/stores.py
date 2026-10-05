"""Read-only store listing.

The camera-registration form needs a ``store_id``. Deriving it from existing
cameras broke the moment the last camera was deleted, so the frontend asks for
the stores directly instead.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.models import Store
from app.schemas.common import ORMModel

router = APIRouter(prefix="/stores", tags=["stores"])


class StoreRead(ORMModel):
    id: str
    code: str
    name: str
    location: str | None = None
    timezone: str = "UTC"
    currency: str = "USD"
    is_active: bool = True


@router.get("", response_model=list[StoreRead], summary="List stores")
def list_stores(session: DbSession, _: CurrentUser) -> list[StoreRead]:
    return [
        StoreRead.model_validate(store)
        for store in session.query(Store).order_by(Store.created_at.asc()).all()
    ]
