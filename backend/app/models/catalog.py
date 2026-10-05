"""Product catalog, shelves, shelf sections and planogram ORM models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import JSONEncoded, Base, UTCDateTime, pk_column, utcnow


class ProductSource(str, Enum):
    """Where a product's authoritative quantity comes from."""

    CAMERA_ESTIMATE = "camera_estimate"  # visual estimate only
    MANUAL = "manual"                    # human count
    POS = "pos"                          # point of sale
    ERP = "erp"                          # enterprise resource planning
    IMPORT = "import"


class ProductCategory(Base):
    """A product grouping used for planogram sections and reporting."""

    __tablename__ = "product_categories"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    code: Mapped[str | None] = mapped_column(String(64), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("product_categories.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)

    products: Mapped[list["Product"]] = relationship(back_populates="category")

    __table_args__ = (UniqueConstraint("store_id", "name", name="uq_category_store_name"),)


class Product(Base):
    """A SKU in the catalog.

    ``reference_image_path`` points at an operator-supplied photo used for
    embedding-based recognition.  ``expected_quantity`` is the *registered*
    figure (from POS/ERP/manual count) and is deliberately kept separate from
    ``estimated_quantity`` produced by the camera.
    """

    __tablename__ = "products"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True
    )
    category_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("product_categories.id", ondelete="SET NULL"), index=True)

    sku: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(120))
    barcode: Mapped[str | None] = mapped_column(String(64), index=True)
    unit_size: Mapped[str | None] = mapped_column(String(64))
    price: Mapped[float | None] = mapped_column(Float)

    # Registered / authoritative inventory inputs
    expected_quantity: Mapped[int | None] = mapped_column(Integer)
    min_threshold: Mapped[int | None] = mapped_column(Integer)
    quantity_source: Mapped[str] = mapped_column(String(24), default=ProductSource.MANUAL.value, nullable=False)
    last_counted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_counted_by: Mapped[str | None] = mapped_column(String(80))

    # Recognition support
    reference_image_path: Mapped[str | None] = mapped_column(Text)
    shelf_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelves.id", ondelete="SET NULL"), index=True)
    facing_capacity: Mapped[int | None] = mapped_column(Integer)
    detector_class_name: Mapped[str | None] = mapped_column(String(80), index=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    category: Mapped[ProductCategory | None] = relationship(back_populates="products")
    shelf: Mapped["Shelf | None"] = relationship(back_populates="products")  # noqa: F821
    embeddings: Mapped[list["ProductEmbedding"]] = relationship(
        back_populates="product", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("store_id", "sku", name="uq_product_store_sku"),
        Index("ix_products_store_active", "store_id", "is_active"),
        Index("ix_products_name", "name"),
    )


class ProductEmbedding(Base):
    """A cached visual embedding for one reference image of a product."""

    __tablename__ = "product_embeddings"

    id: Mapped[str] = pk_column()
    product_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(JSONEncoded, nullable=False)
    source_image: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)

    product: Mapped[Product] = relationship(back_populates="embeddings")

    __table_args__ = (UniqueConstraint("product_id", "model_id", name="uq_embedding_product_model"),)


class Shelf(Base):
    """A configured shelf/rack region with a known capacity."""

    __tablename__ = "shelves"

    id: Mapped[str] = pk_column()
    store_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("stores.id", ondelete="CASCADE"), index=True
    )
    camera_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    zone_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("camera_zones.id", ondelete="SET NULL"), index=True
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    aisle: Mapped[str | None] = mapped_column(String(80))
    level: Mapped[int | None] = mapped_column(Integer)
    capacity_units: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Optional explicit polygon; otherwise the shelf zone polygon is used.
    polygon: Mapped[list[list[float]] | None] = mapped_column(JSONEncoded)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    products: Mapped[list[Product]] = relationship(back_populates="shelf")
    sections: Mapped[list["ShelfSection"]] = relationship(
        back_populates="shelf", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (UniqueConstraint("store_id", "code", name="uq_shelf_store_code"),)


class ShelfSection(Base):
    """A horizontal band of a shelf used for facing counts and planograms."""

    __tablename__ = "shelf_sections"

    id: Mapped[str] = pk_column()
    shelf_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("shelves.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    position_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    capacity_units: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Normalised [x1, y1, x2, y2] within the parent shelf polygon.
    bbox: Mapped[list[float] | None] = mapped_column(JSONEncoded)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    shelf: Mapped[Shelf] = relationship(back_populates="sections")

    __table_args__ = (UniqueConstraint("shelf_id", "code", name="uq_section_shelf_code"),)


class Planogram(Base):
    """The expected layout for a shelf, versioned."""

    __tablename__ = "planograms"

    id: Mapped[str] = pk_column()
    shelf_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("shelves.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="active", nullable=False, index=True)
    reference_image_path: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(80))
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False)

    items: Mapped[list["PlanogramItem"]] = relationship(
        back_populates="planogram", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("shelf_id", "version", name="uq_planogram_shelf_version"),
        Index("ix_planograms_shelf_status", "shelf_id", "status"),
    )


class PlanogramItem(Base):
    """One expected placement: SKU, section, order and facing count."""

    __tablename__ = "planogram_items"

    id: Mapped[str] = pk_column()
    planogram_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("planograms.id", ondelete="CASCADE"), nullable=False, index=True
    )
    product_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("products.id", ondelete="SET NULL"), index=True)
    section_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("shelf_sections.id", ondelete="SET NULL"), index=True)
    position_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    expected_facings: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # Normalised box inside the shelf polygon where this item is expected.
    bbox: Mapped[list[float] | None] = mapped_column(JSONEncoded)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, nullable=False)

    planogram: Mapped[Planogram] = relationship(back_populates="items")

    __table_args__ = (Index("ix_planogram_items_planogram", "planogram_id", "position_index"),)