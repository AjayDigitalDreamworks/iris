"""Planogram compliance.

Compares a **configured expected layout** against what the camera observed.
Without a planogram there is nothing to compare and the module returns an
explicit "not configured" result rather than a fabricated compliance number.

Supported checks:

* ``missing``       an expected SKU was not observed in its section
* ``over_faced``    more units observed than the expected facing count
* ``misplaced``     the SKU was observed in a different section
* ``compliant``     observed facings match the planogram within tolerance
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.vision.shelf_analysis import ShelfObservation
from app.vision.types import BBox, Detection


@dataclass
class PlanogramRow:
    product_id: str | None
    sku: str | None
    product_name: str | None
    section_code: str | None
    expected_facings: int
    observed_facings: int
    status: str
    confidence: float = 0.0
    note: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "sku": self.sku,
            "product_name": self.product_name,
            "section_code": self.section_code,
            "expected_facings": self.expected_facings,
            "observed_facings": self.observed_facings,
            "facing_difference": self.observed_facings - self.expected_facings,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "note": self.note,
        }


@dataclass
class PlanogramResult:
    planogram_id: str | None
    shelf_id: str | None
    version: int | None
    compliance_pct: float
    rows: list[PlanogramRow] = field(default_factory=list)
    configured: bool = True
    note: str = ""


class PlanogramChecker:
    """Evaluates observed shelf state against a planogram definition."""

    def __init__(self, *, facing_tolerance: int = 1, min_confidence: float = 0.35) -> None:
        self.facing_tolerance = facing_tolerance
        self.min_confidence = min_confidence

    def check(
        self,
        planogram: dict[str, Any] | None,
        detections: list[Detection],
        shelf_observation: ShelfObservation | None = None,
        *,
        min_confidence: float | None = None,
    ) -> PlanogramResult:
        """Compare detections with a planogram. Never invents a planogram."""
        floor = self.min_confidence if min_confidence is None else min_confidence
        if not planogram or not planogram.get("items"):
            return PlanogramResult(
                planogram_id=(planogram or {}).get("id"),
                shelf_id=(planogram or {}).get("shelf_id"),
                version=(planogram or {}).get("version"),
                compliance_pct=0.0,
                rows=[],
                configured=False,
                note=(
                    "No planogram is configured for this shelf. Compliance cannot be "
                    "computed without an expected layout; this is not a failure."
                ),
            )

        confident = [
            d for d in detections if d.confidence >= floor and d.sku is not None
        ]
        observed_by_sku: dict[str, list[Detection]] = defaultdict(list)
        for detection in confident:
            observed_by_sku[detection.sku or ""].append(detection)

        section_of_detection: dict[str, str | None] = {}
        if shelf_observation and shelf_observation.slots:
            for slot in shelf_observation.slots:
                if slot.sku and slot.occupied:
                    section_of_detection.setdefault(slot.sku, f"slot_{slot.index}")

        rows: list[PlanogramRow] = []
        for item in planogram["items"]:
            sku = item.get("sku")
            expected = int(item.get("expected_facings") or 1)
            hits = observed_by_sku.get(sku or "", [])
            observed = len(hits)
            confidence = (
                round(sum(h.confidence for h in hits) / len(hits), 4) if hits else 0.0
            )
            if observed == 0:
                status = "missing"
                note = "Expected SKU was not observed in the shelf region."
            elif observed > expected + self.facing_tolerance:
                status = "over_faced"
                note = "More facings observed than the planogram expects."
            elif observed + self.facing_tolerance < expected:
                status = "low_confidence" if confidence < floor else "missing"
                note = "Fewer facings observed than expected."
            else:
                status = "compliant"
                note = ""
            rows.append(
                PlanogramRow(
                    product_id=item.get("product_id"),
                    sku=sku,
                    product_name=item.get("product_name"),
                    section_code=item.get("section_code"),
                    expected_facings=expected,
                    observed_facings=observed,
                    status=status,
                    confidence=confidence,
                    note=note,
                )
            )

        # SKUs observed but not on the planogram are reported as misplaced.
        expected_skus = {item.get("sku") for item in planogram["items"]}
        for sku, hits in observed_by_sku.items():
            if sku and sku not in expected_skus:
                rows.append(
                    PlanogramRow(
                        product_id=hits[0].product_id,
                        sku=sku,
                        product_name=None,
                        section_code=section_of_detection.get(sku),
                        expected_facings=0,
                        observed_facings=len(hits),
                        status="misplaced",
                        confidence=round(sum(h.confidence for h in hits) / len(hits), 4),
                        note="SKU is on the shelf but is not part of the current planogram.",
                    )
                )

        scored = [r for r in rows if r.status != "low_confidence"]
        compliant = sum(1 for r in scored if r.status == "compliant")
        compliance_pct = round(100.0 * compliant / len(scored), 2) if scored else 0.0

        return PlanogramResult(
            planogram_id=planogram.get("id"),
            shelf_id=planogram.get("shelf_id"),
            version=planogram.get("version"),
            compliance_pct=compliance_pct,
            rows=rows,
            configured=True,
            note=(
                "Compliance reflects detection confidence on the observed frame. "
                "Occlusion and shelf geometry errors can cause both false non-compliance "
                "and false compliance."
            ),
        )


def planogram_rows_to_dicts(rows: Sequence[PlanogramRow]) -> list[dict[str, Any]]:
    return [row.as_payload() for row in rows]


def summarise(result: PlanogramResult) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in result.rows:
        counts[row.status] += 1
    return {
        "configured": result.configured,
        "compliance_pct": result.compliance_pct,
        "status_counts": dict(counts),
        "rows": len(result.rows),
        "note": result.note,
    }


def expected_boxes(items: Sequence[dict[str, Any]]) -> list[BBox]:
    """Extract expected placement boxes (normalised) from planogram items."""
    boxes: list[BBox] = []
    for item in items:
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            boxes.append(BBox(*[float(v) for v in bbox]))
    return boxes