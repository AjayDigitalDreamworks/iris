"""Shelf, shelf section and planogram configuration endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request

from app.api.deps import (
    AdminUser,
    CurrentUser,
    DbSession,
    ManagerUser,
    PageParams,
    commit_before_camera_work,
    write_audit,
)
from app.core.errors import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.models import (
    Camera,
    Detection,
    Planogram,
    PlanogramItem,
    Product,
    Shelf,
    ShelfSection,
    Store,
)
from app.schemas import (
    MessageResponse,
    Page,
    PlanogramCompliance,
    PlanogramComplianceRow,
    PlanogramCreate,
    PlanogramRead,
    ShelfCreate,
    ShelfRead,
    ShelfSectionCreate,
    ShelfSectionRead,
    ShelfUpdate,
)
from app.services.db_service import paginate
from app.vision.planogram import PlanogramChecker, planogram_rows_to_dicts

router = APIRouter(prefix="/shelves", tags=["shelves"])
planograms_router = APIRouter(prefix="/planograms", tags=["planograms"])
logger = get_logger("app")


# ── Shelves ──────────────────────────────────────────────────────────────────
@router.get("", response_model=Page[ShelfRead], summary="List shelves")
def list_shelves(
    session: DbSession,
    page: PageParams,
    _: CurrentUser,
    store_id: str | None = None,
    camera_id: str | None = None,
    is_active: bool | None = None,
) -> Page[ShelfRead]:
    query = session.query(Shelf)
    if store_id:
        query = query.filter(Shelf.store_id == store_id)
    if camera_id:
        query = query.filter(Shelf.camera_id == camera_id)
    if is_active is not None:
        query = query.filter(Shelf.is_active.is_(is_active))
    rows, total = paginate(query.order_by(Shelf.code.asc()), page.limit, page.offset)
    return Page.build([ShelfRead.model_validate(s) for s in rows], total, page.limit, page.offset)


@router.post("", response_model=ShelfRead, status_code=201, summary="Create a shelf (admin)")
def create_shelf(
    payload: ShelfCreate, request: Request, session: DbSession, principal: AdminUser
) -> ShelfRead:
    if session.get(Store, payload.store_id) is None if payload.store_id else False:
        raise NotFoundError(f"Store {payload.store_id!r} does not exist.")
    if payload.camera_id and session.get(Camera, payload.camera_id) is None:
        raise NotFoundError(f"Camera {payload.camera_id!r} does not exist.")
    existing = session.query(Shelf).filter(Shelf.code == payload.code).first()
    if existing is not None:
        raise ConflictError(f"Shelf code {payload.code!r} is already in use.")
    shelf = Shelf(**payload.model_dump())
    session.add(shelf)
    session.flush()
    write_audit(session, principal, "shelf.created", entity_type="shelf", entity_id=shelf.id,
                detail={"code": shelf.code, "capacity": shelf.capacity_units}, request=request)
    return ShelfRead.model_validate(shelf)


@router.get("/{shelf_id}", response_model=ShelfRead, summary="Shelf detail")
def get_shelf(shelf_id: str, session: DbSession, _: CurrentUser) -> ShelfRead:
    shelf = session.get(Shelf, shelf_id)
    if shelf is None:
        raise NotFoundError(f"Shelf {shelf_id!r} does not exist.")
    return ShelfRead.model_validate(shelf)


@router.patch("/{shelf_id}", response_model=ShelfRead, summary="Update a shelf (admin)")
def update_shelf(
    shelf_id: str, payload: ShelfUpdate, request: Request,
    session: DbSession, principal: AdminUser,
) -> ShelfRead:
    shelf = session.get(Shelf, shelf_id)
    if shelf is None:
        raise NotFoundError(f"Shelf {shelf_id!r} does not exist.")
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if value is not None:
            setattr(shelf, key, value)
    session.flush()
    write_audit(session, principal, "shelf.updated", entity_type="shelf", entity_id=shelf.id,
                detail={"fields": sorted(data)}, request=request)
    if payload.camera_id or payload.polygon or payload.zone_id:
        from app.services.camera_manager import camera_manager

        # Commit first: the reloaded pipeline writes to the DB immediately
        # and would otherwise block on this request's write transaction.
        if shelf.camera_id:
            commit_before_camera_work(session)
            camera_manager.restart(shelf.camera_id)
    return ShelfRead.model_validate(shelf)


@router.delete("/{shelf_id}", response_model=MessageResponse, summary="Delete a shelf (admin)")
def delete_shelf(
    shelf_id: str, request: Request, session: DbSession, principal: AdminUser
) -> MessageResponse:
    shelf = session.get(Shelf, shelf_id)
    if shelf is None:
        raise NotFoundError(f"Shelf {shelf_id!r} does not exist.")
    session.delete(shelf)
    write_audit(session, principal, "shelf.deleted", entity_type="shelf", entity_id=shelf_id,
                request=request)
    return MessageResponse(message=f"Shelf {shelf_id} deleted.")


# ── Sections ─────────────────────────────────────────────────────────────────
@router.get("/{shelf_id}/sections", response_model=list[ShelfSectionRead], summary="List sections")
def list_sections(shelf_id: str, session: DbSession, _: CurrentUser) -> list[ShelfSectionRead]:
    if session.get(Shelf, shelf_id) is None:
        raise NotFoundError(f"Shelf {shelf_id!r} does not exist.")
    return [
        ShelfSectionRead.model_validate(s)
        for s in session.query(ShelfSection)
        .filter(ShelfSection.shelf_id == shelf_id)
        .order_by(ShelfSection.position_index.asc())
        .all()
    ]


@router.post(
    "/{shelf_id}/sections", response_model=ShelfSectionRead, status_code=201,
    summary="Create a shelf section (admin)",
)
def create_section(
    shelf_id: str, payload: ShelfSectionCreate, request: Request,
    session: DbSession, principal: AdminUser,
) -> ShelfSectionRead:
    if session.get(Shelf, shelf_id) is None:
        raise NotFoundError(f"Shelf {shelf_id!r} does not exist.")
    existing = (
        session.query(ShelfSection)
        .filter(ShelfSection.shelf_id == shelf_id, ShelfSection.code == payload.code)
        .first()
    )
    if existing is not None:
        raise ConflictError(f"Section code {payload.code!r} already exists on this shelf.")
    section = ShelfSection(shelf_id=shelf_id, **payload.model_dump())
    session.add(section)
    session.flush()
    write_audit(session, principal, "shelf_section.created", entity_type="shelf_section",
                entity_id=section.id, request=request)
    return ShelfSectionRead.model_validate(section)


@router.delete(
    "/sections/{section_id}", response_model=MessageResponse, summary="Delete a section (admin)"
)
def delete_section(
    section_id: str, request: Request, session: DbSession, principal: AdminUser
) -> MessageResponse:
    section = session.get(ShelfSection, section_id)
    if section is None:
        raise NotFoundError(f"Section {section_id!r} does not exist.")
    session.delete(section)
    write_audit(session, principal, "shelf_section.deleted", entity_type="shelf_section",
                entity_id=section_id, request=request)
    return MessageResponse(message=f"Section {section_id} deleted.")


# ── Planograms ───────────────────────────────────────────────────────────────
@planograms_router.get("", response_model=list[PlanogramRead], summary="List planograms")
def list_planograms(
    session: DbSession, _: CurrentUser, shelf_id: str | None = None,
    status: str | None = None,
) -> list[PlanogramRead]:
    query = session.query(Planogram)
    if shelf_id:
        query = query.filter(Planogram.shelf_id == shelf_id)
    if status:
        query = query.filter(Planogram.status == status)
    return [
        PlanogramRead.model_validate(p)
        for p in query.order_by(Planogram.shelf_id.asc(), Planogram.version.desc()).all()
    ]


@planograms_router.post(
    "", response_model=PlanogramRead, status_code=201, summary="Create a planogram version (manager)"
)
def create_planogram(
    payload: PlanogramCreate, request: Request, session: DbSession, principal: ManagerUser
) -> PlanogramRead:
    shelf = session.get(Shelf, payload.shelf_id)
    if shelf is None:
        raise NotFoundError(f"Shelf {payload.shelf_id!r} does not exist.")
    version = payload.version or (
        (session.query(Planogram).filter(Planogram.shelf_id == shelf.id)
         .order_by(Planogram.version.desc()).first().version + 1)
        if session.query(Planogram).filter(Planogram.shelf_id == shelf.id).count()
        else 1
    )
    planogram = Planogram(
        shelf_id=shelf.id,
        name=payload.name,
        version=version,
        status="active" if payload.activate else "draft",
        reference_image_path=payload.reference_image_path,
        created_by=principal.username,
        activated_at=datetime.now(tz=timezone.utc) if payload.activate else None,
    )
    session.add(planogram)
    session.flush()
    for item in payload.items:
        product = session.get(Product, item.product_id) if item.product_id else None
        session.add(
            PlanogramItem(
                planogram_id=planogram.id,
                product_id=item.product_id,
                section_id=item.section_id,
                position_index=item.position_index,
                expected_facings=item.expected_facings,
                bbox=item.bbox,
                notes=item.notes,
            )
        )
        _ = product
    if payload.activate:
        # Only one active version per shelf.
        session.query(Planogram).filter(
            Planogram.shelf_id == shelf.id, Planogram.id != planogram.id,
            Planogram.status == "active",
        ).update({"status": "archived"}, synchronize_session=False)
    session.flush()
    write_audit(session, principal, "planogram.created", entity_type="planogram",
                entity_id=planogram.id, detail={"shelf_id": shelf.id, "version": version},
                request=request)
    planogram = session.get(Planogram, planogram.id)
    return PlanogramRead.model_validate(planogram)


@planograms_router.get("/{planogram_id}", response_model=PlanogramRead, summary="Planogram detail")
def get_planogram(planogram_id: str, session: DbSession, _: CurrentUser) -> PlanogramRead:
    planogram = session.get(Planogram, planogram_id)
    if planogram is None:
        raise NotFoundError(f"Planogram {planogram_id!r} does not exist.")
    return PlanogramRead.model_validate(planogram)


@planograms_router.delete("/{planogram_id}", response_model=MessageResponse, summary="Delete (admin)")
def delete_planogram(
    planogram_id: str, request: Request, session: DbSession, principal: AdminUser
) -> MessageResponse:
    planogram = session.get(Planogram, planogram_id)
    if planogram is None:
        raise NotFoundError(f"Planogram {planogram_id!r} does not exist.")
    session.delete(planogram)
    write_audit(session, principal, "planogram.deleted", entity_type="planogram",
                entity_id=planogram_id, request=request)
    return MessageResponse(message=f"Planogram {planogram_id} deleted.")


@planograms_router.post(
    "/{planogram_id}/check", response_model=PlanogramCompliance,
    summary="Compare the planogram against recent camera observations (manager)",
)
def check_planogram(
    planogram_id: str,
    session: DbSession,
    _: ManagerUser,
    lookback_minutes: int = 15,
) -> PlanogramCompliance:
    """Run a compliance check against detections stored in the lookback window."""
    from app.schemas import PlanogramComparisonRequest

    request = PlanogramComparisonRequest(lookback_minutes=lookback_minutes)
    return _run_compliance(session, planogram_id=planogram_id, request=request)


def _run_compliance(
    session, *, planogram_id: str | None = None, shelf_id: str | None = None,
    request: PlanogramComparisonRequest | None = None,
) -> PlanogramCompliance:
    request = request or PlanogramComparisonRequest()
    if planogram_id:
        planogram = session.get(Planogram, planogram_id)
    else:
        query = session.query(Planogram).filter(Planogram.status == "active")
        if shelf_id:
            query = query.filter(Planogram.shelf_id == shelf_id)
        planogram = query.first()

    if planogram is None:
        return PlanogramCompliance(
            planogram_id=None,
            shelf_id=shelf_id,
            version=None,
            compliance_pct=0.0,
            checked_at=datetime.now(tz=timezone.utc),
            rows=[],
            note=(
                "No planogram is configured for this shelf, so compliance cannot be "
                "computed. This is not a failure — configure a planogram to enable the check."
            ),
        )

    shelf = session.get(Shelf, planogram.shelf_id)
    since = datetime.now(tz=timezone.utc) - timedelta(minutes=request.lookback_minutes)
    detection_query = session.query(Detection).filter(Detection.frame_timestamp >= since)
    if shelf is not None:
        detection_query = detection_query.filter(Detection.shelf_id == shelf.id)
    if request.camera_id:
        detection_query = detection_query.filter(Detection.camera_id == request.camera_id)
    rows = detection_query.all()

    from app.vision.types import BBox, Detection as VisionDetection

    detections = [
        VisionDetection(
            bbox=BBox(r.x1, r.y1, r.x2, r.y2),
            class_id=r.class_id,
            class_name=r.class_name,
            confidence=r.confidence,
            product_id=r.product_id,
            sku=r.sku,
            similarity=r.similarity,
        )
        for r in rows
    ]

    sections = {
        s.id: s.code
        for s in session.query(ShelfSection).filter(ShelfSection.shelf_id == planogram.shelf_id).all()
    }
    products = {
        p.id: p
        for p in session.query(Product).filter(
            Product.id.in_([i.product_id for i in planogram.items if i.product_id]) or [None]
        ).all()
    } if planogram.items else {}
    planogram_dict = {
        "id": planogram.id,
        "shelf_id": planogram.shelf_id,
        "version": planogram.version,
        "items": [
            {
                "product_id": item.product_id,
                "sku": products[item.product_id].sku if item.product_id in products else None,
                "product_name": products[item.product_id].name if item.product_id in products else None,
                "section_code": sections.get(item.section_id) if item.section_id else None,
                "expected_facings": item.expected_facings,
            }
            for item in planogram.items
        ],
    }

    checker = PlanogramChecker()
    result = checker.check(planogram_dict, detections)
    if not rows:
        return PlanogramCompliance(
            planogram_id=planogram.id,
            shelf_id=planogram.shelf_id,
            shelf_code=shelf.code if shelf else None,
            version=planogram.version,
            compliance_pct=0.0,
            checked_at=datetime.now(tz=timezone.utc),
            rows=[],
            note=(
                f"No confident detections in the last {request.lookback_minutes} minute(s). "
                "Compliance was not evaluated rather than reported as zero."
            ),
        )

    return PlanogramCompliance(
        planogram_id=result.planogram_id,
        shelf_id=result.shelf_id,
        shelf_code=shelf.code if shelf else None,
        version=result.version,
        compliance_pct=result.compliance_pct,
        checked_at=datetime.now(tz=timezone.utc),
        rows=[
            PlanogramComplianceRow(**{k: v for k, v in payload.items() if k != "facing_difference"})
            for payload in planogram_rows_to_dicts(result.rows)
        ],
        note=result.note,
    )


@router.post("/{shelf_id}/planogram/check", response_model=PlanogramCompliance,
             summary="Check the shelf's active planogram (manager)")
def check_shelf_planogram(
    shelf_id: str, session: DbSession, _: ManagerUser, lookback_minutes: int = 15
) -> PlanogramCompliance:
    if session.get(Shelf, shelf_id) is None:
        raise NotFoundError(f"Shelf {shelf_id!r} does not exist.")
    return _run_compliance(
        session,
        shelf_id=shelf_id,
        request=PlanogramComparisonRequest(lookback_minutes=lookback_minutes),
    )