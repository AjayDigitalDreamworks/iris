"""Inventory service.

Owns the strict separation between:

* ``inventory_snapshots`` — camera-derived **estimates** (append-only history)
* ``products.expected_quantity`` — the **registered** figure (POS / ERP / manual)
* ``inventory_transactions`` — an audit trail of deliberate changes

A camera estimate never overwrites registered stock.  Manual adjustment is the
only path that does, and it always records who, when, why and the old value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import func

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.models import (
    InventorySnapshot,
    InventoryTransaction,
    Product,
    ProductSource,
    Shelf,
)
from app.services.db_service import paginate, run_in_session, settings_service
from app.vision.rules import StockState, classify_stock
from app.vision.stock import StockEstimate, compare_to_registered

logger = get_logger("app")


@dataclass
class InventoryView:
    """One SKU's live inventory position, assembled from estimate + registry."""

    sku: str
    product_id: str | None
    product_name: str | None
    shelf_id: str | None
    shelf_code: str | None
    camera_id: str | None
    observed_quantity: int
    estimated_quantity: float
    capacity_units: int
    stock_percentage: float
    occupancy_pct: float | None
    confidence: float
    observation_count: int
    method: str
    observed_at: datetime | None
    observation_age_seconds: float | None
    registered_quantity: int | None
    registered_source: str | None
    discrepancy: int | None
    state: StockState
    low_stock_threshold_pct: float
    out_of_stock_threshold_pct: float
    is_stale: bool


class InventoryService:
    """Reads and writes inventory estimates and registered stock."""

    def __init__(self) -> None:
        self.snapshots_written = 0

    # ── writing estimates ───────────────────────────────────────────────────
    def record_estimate(
        self,
        estimate: StockEstimate,
        *,
        store_id: str | None = None,
        job_id: str | None = None,
        source: str = "camera_estimate",
        notes: str | None = None,
        session_factory: Any = None,
    ) -> InventorySnapshot:
        """Append one camera estimate to the snapshot history."""

        def _write(session) -> InventorySnapshot:
            snapshot = InventorySnapshot(
                store_id=store_id,
                camera_id=estimate.camera_id,
                shelf_id=estimate.shelf_id,
                product_id=estimate.product_id,
                sku=estimate.sku,
                job_id=job_id,
                observed_quantity=estimate.observed_quantity,
                estimated_quantity=estimate.estimated_quantity,
                capacity_units=estimate.capacity_units,
                stock_percentage=estimate.stock_percentage,
                occupancy_pct=estimate.occupancy_pct,
                confidence=estimate.confidence,
                observation_count=estimate.observation_count,
                source=source,
                method=estimate.method,
                notes=notes or ("; ".join(estimate.notes)[:500] or None),
            )
            session.add(snapshot)
            session.flush()
            self.snapshots_written += 1
            return snapshot

        return run_in_session(_write, session_factory)

    def record_estimates(
        self,
        estimates: Sequence[StockEstimate],
        *,
        store_id: str | None = None,
        job_id: str | None = None,
        source: str = "camera_estimate",
        session_factory: Any = None,
    ) -> int:
        if not estimates:
            return 0
        written = 0

        def _write(session) -> int:
            nonlocal written
            for estimate in estimates:
                session.add(
                    InventorySnapshot(
                        store_id=store_id,
                        camera_id=estimate.camera_id,
                        shelf_id=estimate.shelf_id,
                        product_id=estimate.product_id,
                        sku=estimate.sku,
                        job_id=job_id,
                        observed_quantity=estimate.observed_quantity,
                        estimated_quantity=estimate.estimated_quantity,
                        capacity_units=estimate.capacity_units,
                        stock_percentage=estimate.stock_percentage,
                        occupancy_pct=estimate.occupancy_pct,
                        confidence=estimate.confidence,
                        observation_count=estimate.observation_count,
                        source=source,
                        method=estimate.method,
                        notes="; ".join(estimate.notes)[:500] or None,
                    )
                )
                written += 1
            session.flush()
            return written

        run_in_session(_write, session_factory)
        return written

    # ── reading ─────────────────────────────────────────────────────────────
    def latest_estimates(
        self,
        *,
        store_id: str | None = None,
        camera_id: str | None = None,
        shelf_id: str | None = None,
        within_minutes: int = 15,
        include_skus_without_data: bool = True,
        session_factory: Any = None,
    ) -> list[InventoryView]:
        """Return the newest estimate per SKU within a freshness window."""
        from app.models import Camera

        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=within_minutes)
        thresholds = settings_service.stock_thresholds()
        low_pct = settings_service.get_float("low_stock_threshold_pct", 20.0)
        out_pct = settings_service.get_float("out_of_stock_threshold_pct", 5.0)

        def _load(session) -> list[InventoryView]:
            latest: dict[str, InventorySnapshot] = {}
            query = session.query(InventorySnapshot).filter(
                InventorySnapshot.observed_at >= cutoff
            )
            if store_id:
                query = query.filter(InventorySnapshot.store_id == store_id)
            if camera_id:
                query = query.filter(InventorySnapshot.camera_id == camera_id)
            if shelf_id:
                query = query.filter(InventorySnapshot.shelf_id == shelf_id)
            for row in query.order_by(InventorySnapshot.observed_at.asc()).all():
                latest[row.sku] = row  # ascending order ⇒ newest wins

            products = {
                p.sku: p
                for p in session.query(Product)
                .filter(Product.is_active.is_(True))
                .filter(Product.store_id == store_id if store_id else Product.store_id.isnot(None))
                .all()
            }
            if store_id:
                products.update({p.sku: p for p in session.query(Product)
                                 .filter(Product.store_id.is_(None)).all()})
            shelves = {s.id: s for s in session.query(Shelf).all()}
            cameras = {c.id: c for c in session.query(Camera).all()}
            now = datetime.now(tz=timezone.utc)

            views: list[InventoryView] = []
            skus = set(products) | set(latest)
            for sku in sorted(skus):
                snapshot = latest.get(sku)
                product = products.get(sku)
                if snapshot is None and not include_skus_without_data:
                    continue
                capacity = 0
                if product is not None and product.shelf_id:
                    shelf = shelves.get(product.shelf_id)
                    capacity = shelf.capacity_units if shelf else 0
                if snapshot is not None and snapshot.capacity_units:
                    capacity = snapshot.capacity_units

                estimated = snapshot.estimated_quantity if snapshot else 0.0
                confidence = snapshot.confidence if snapshot else 0.0
                pct = snapshot.stock_percentage if snapshot else -1.0
                state = (
                    classify_stock(estimated, capacity, confidence, thresholds)
                    if capacity > 0
                    else StockState.UNKNOWN
                )
                age = (
                    (now - snapshot.observed_at).total_seconds() if snapshot else None
                )
                comparison = compare_to_registered(
                    estimated, product.expected_quantity if product else None
                )
                views.append(
                    InventoryView(
                        sku=sku,
                        product_id=(snapshot.product_id if snapshot else None)
                        or (product.id if product else None),
                        product_name=product.name if product else None,
                        shelf_id=(snapshot.shelf_id if snapshot else None)
                        or (product.shelf_id if product else None),
                        shelf_code=(
                            shelves.get(snapshot.shelf_id).code
                            if snapshot and snapshot.shelf_id and snapshot.shelf_id in shelves
                            else (shelves.get(product.shelf_id).code if product and product.shelf_id in shelves else None)
                        ),
                        camera_id=(snapshot.camera_id if snapshot else None),
                        observed_quantity=snapshot.observed_quantity if snapshot else 0,
                        estimated_quantity=estimated,
                        capacity_units=capacity,
                        stock_percentage=pct,
                        occupancy_pct=snapshot.occupancy_pct if snapshot else None,
                        confidence=confidence,
                        observation_count=snapshot.observation_count if snapshot else 0,
                        method=snapshot.method if snapshot else "no_observation",
                        observed_at=snapshot.observed_at if snapshot else None,
                        observation_age_seconds=round(age, 1) if age is not None else None,
                        registered_quantity=product.expected_quantity if product else None,
                        registered_source=product.quantity_source if product else None,
                        discrepancy=comparison["discrepancy"],
                        state=state,
                        low_stock_threshold_pct=low_pct,
                        out_of_stock_threshold_pct=out_pct,
                        is_stale=bool(age is not None and age > 900),
                    )
                )
            return views

        return run_in_session(_load, session_factory)

    def history(
        self,
        sku: str,
        *,
        hours: int = 24,
        limit: int = 500,
        session_factory: Any = None,
    ) -> list[InventorySnapshot]:
        since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)

        def _load(session) -> list[InventorySnapshot]:
            return list(
                session.query(InventorySnapshot)
                .filter(InventorySnapshot.sku == sku, InventorySnapshot.observed_at >= since)
                .order_by(InventorySnapshot.observed_at.asc())
                .limit(limit)
                .all()
            )

        return run_in_session(_load, session_factory)

    def sku_detail(self, sku: str, *, session_factory: Any = None) -> dict[str, Any]:
        def _load(session) -> dict[str, Any]:
            product = session.query(Product).filter(Product.sku == sku).first()
            if product is None:
                raise NotFoundError(f"SKU {sku!r} is not in the catalog.")
            shelf = session.get(Shelf, product.shelf_id) if product.shelf_id else None
            snapshots = (
                session.query(InventorySnapshot)
                .filter(InventorySnapshot.sku == sku)
                .order_by(InventorySnapshot.observed_at.desc())
                .limit(200)
                .all()
            )
            transactions = (
                session.query(InventoryTransaction)
                .filter(InventoryTransaction.sku == sku)
                .order_by(InventoryTransaction.created_at.desc())
                .limit(100)
                .all()
            )
            comparison = compare_to_registered(
                snapshots[0].estimated_quantity if snapshots else 0.0,
                product.expected_quantity,
            )
            return {
                "product": product,
                "shelf": shelf,
                "snapshots": list(reversed(snapshots)),
                "transactions": transactions,
                "comparison": comparison,
            }

        return run_in_session(_load, session_factory)

    # ── manual correction ───────────────────────────────────────────────────
    def adjust(
        self,
        sku: str,
        new_quantity: int,
        *,
        reason: str,
        username: str,
        reference: str | None = None,
        create_transaction: bool = True,
        session_factory: Any = None,
    ) -> dict[str, Any]:
        """Set the REGISTERED quantity for a SKU, with a full audit trail."""

        def _write(session) -> dict[str, Any]:
            product = session.query(Product).filter(Product.sku == sku).first()
            if product is None:
                raise NotFoundError(f"SKU {sku!r} is not in the catalog.")
            previous = product.expected_quantity
            if previous == new_quantity:
                raise ConflictError("The registered quantity is already that value.")
            change = new_quantity - (previous or 0)
            product.expected_quantity = new_quantity
            product.last_counted_at = datetime.now(tz=timezone.utc)
            product.last_counted_by = username
            product.quantity_source = ProductSource.MANUAL.value
            product.updated_at = datetime.now(tz=timezone.utc)

            transaction = None
            if create_transaction:
                transaction = InventoryTransaction(
                    store_id=product.store_id,
                    product_id=product.id,
                    sku=sku,
                    shelf_id=product.shelf_id,
                    change=change,
                    resulting_quantity=new_quantity,
                    source=ProductSource.MANUAL.value,
                    reason=reason,
                    reference=reference,
                    performed_by=username,
                )
                session.add(transaction)
                session.flush()
            return {
                "sku": sku,
                "previous_quantity": previous,
                "new_quantity": new_quantity,
                "change": change,
                "performed_by": username,
                "transaction_id": transaction.id if transaction else None,
            }

        result = run_in_session(_write, session_factory)
        logger.info(
            "Inventory adjusted manually",
            extra={"sku": sku, "change": result["change"], "by": username},
        )
        return result

    def transactions(
        self, *, sku: str | None = None, limit: int = 100, offset: int = 0, session_factory: Any = None
    ) -> tuple[list[InventoryTransaction], int]:
        def _load(session) -> tuple[list[InventoryTransaction], int]:
            query = session.query(InventoryTransaction)
            if sku:
                query = query.filter(InventoryTransaction.sku == sku)
            rows, total = paginate(
                query.order_by(InventoryTransaction.created_at.desc()), limit, offset
            )
            return rows, total

        return run_in_session(_load, session_factory)

    # ── summary ─────────────────────────────────────────────────────────────
    def summary(
        self, *, store_id: str | None = None, within_minutes: int = 15, session_factory: Any = None
    ) -> dict[str, Any]:
        views = self.latest_estimates(
            store_id=store_id, within_minutes=within_minutes, session_factory=session_factory
        )
        counts = {state: 0 for state in StockState}
        confidences: list[float] = []
        percentages: list[float] = []
        observed = 0
        stale = 0
        for view in views:
            counts[view.state] += 1
            if view.observed_at is not None:
                observed += 1
            if view.is_stale:
                stale += 1
            if view.confidence:
                confidences.append(view.confidence)
            if view.stock_percentage >= 0:
                percentages.append(view.stock_percentage)
        return {
            "total_skus": len(views),
            "observed_skus": observed,
            "low_stock_count": counts[StockState.LOW_STOCK],
            "out_of_stock_count": counts[StockState.OUT_OF_STOCK],
            "healthy_count": counts[StockState.OK],
            "unknown_count": counts[StockState.UNKNOWN],
            "stale_skus": stale,
            "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "mean_stock_percentage": round(sum(percentages) / len(percentages), 2)
            if percentages
            else 0.0,
        }


inventory_service = InventoryService()


def validate_adjustment(new_quantity: int) -> int:
    if new_quantity < 0:
        raise ValidationError("Registered quantity cannot be negative.")
    return new_quantity


def count_transactions(session: Any) -> int:
    return int(session.query(func.count(InventoryTransaction.id)).scalar() or 0)


_ = SessionLocal  # re-exported for convenience in scripts