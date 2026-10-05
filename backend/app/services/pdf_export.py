"""PDF rendering for report exports.

Produces a real, paginated PDF from the same row dictionaries the CSV exporter
uses, so the two formats can never drift apart.

The footer carries the honesty note that applies to every camera-derived number
in IRIS: these are estimates of *visible* shelf stock, and retail accuracy is
unvalidated until measured against store ground truth. A report that leaves the
building should not be able to imply more than the system actually knows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger

logger = get_logger("app")

#: The note printed on every exported report.
ESTIMATE_DISCLAIMER = (
    "Camera-derived figures are estimates of VISIBLE shelf stock and are not "
    "authoritative inventory. They never overwrite POS/ERP/manual counts. "
    "Retail accuracy is unvalidated until measured against store ground truth."
)

#: Column order and width budget. Reports are read on screen as often as on
#: paper, so the table is kept to a readable number of columns and any extra
#: key is dropped rather than shrunk into illegibility.
MAX_COLUMNS = 9

#: Hard cap on rows rendered into one PDF. A 7-day detections export can be
#: hundreds of thousands of rows; flowing that many laid-out cells takes minutes
#: and produces an unusable file. Rows beyond this are omitted, and the PDF says
#: so explicitly - a truncated report must never look like a complete one.
MAX_PDF_ROWS = 2000


def _flatten(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    """Union of keys, in first-seen order, capped at MAX_COLUMNS."""
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen and len(seen) < MAX_COLUMNS:
                seen.append(key)
    return seen


def render_report_pdf(
    path: str,
    *,
    report_type: str,
    rows: list[dict[str, Any]],
    store_name: str | None = None,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
) -> str:
    """Write ``rows`` to ``path`` as a PDF and return the path.

    An empty result set still produces a valid PDF that says so, rather than a
    zero-byte file that looks like a failed export.
    """
    # Imported lazily so the CSV/JSON path never pays for a PDF dependency.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    generated_at = datetime.now(tz=timezone.utc)
    # A table of numbers reads better across the page than down it.
    pagesize = landscape(A4) if rows and len(_columns(rows)) > 5 else A4
    document = SimpleDocTemplate(
        path,
        pagesize=pagesize,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=f"IRIS {report_type} report",
        author="IRIS — Intelligent Retail Insight System",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "IRISTitle", parent=styles["Title"], fontSize=16, spaceAfter=2
    )
    meta_style = ParagraphStyle(
        "IRISMeta", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#444444")
    )
    note_style = ParagraphStyle(
        "IRISNote", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#7a3b00")
    )

    story: list[Any] = [
        Paragraph(f"IRIS — {report_type.replace('_', ' ').title()} report", title_style)
    ]
    meta_bits = [f"Generated: {generated_at.isoformat(timespec='seconds')}", f"Rows: {len(rows)}"]
    if store_name:
        meta_bits.append(f"Store: {store_name}")
    if range_start or range_end:
        start = range_start.isoformat(timespec="seconds") if range_start else "start"
        end = range_end.isoformat(timespec="seconds") if range_end else "now"
        meta_bits.append(f"Range: {start} → {end}")
    story.append(Paragraph(" · ".join(meta_bits), meta_style))
    story.append(Spacer(1, 6 * mm))

    if not rows:
        story.append(
            Paragraph(
                "No data in this range. This is an empty result, not a failed export.",
                styles["Normal"],
            )
        )
    else:
        columns = _columns(rows)
        header = [Paragraph(f"<b>{c}</b>", meta_style) for c in columns]
        table_data = [header]
        # Plain strings, not one Paragraph per cell: Paragraph is a mini
        # markup engine and building 9 x 20,000 of them takes minutes. Strings
        # are what reportlab's fast path is for.
        for row in rows[:MAX_PDF_ROWS]:
            table_data.append([_flatten(row.get(c)) for c in columns])
        omitted = len(rows) - MAX_PDF_ROWS
        if omitted > 0:
            table_data.append([f"... {omitted} further row(s) omitted from this PDF "
                               f"(see the CSV export for the full set)"] + [""] * (len(columns) - 1))
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#94a3b8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    # Zebra striping, so long tables stay readable.
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        story.append(table)

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(f"<b>Note.</b> {ESTIMATE_DISCLAIMER}", note_style))

    def _footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(14 * mm, 9 * mm, "IRIS — local-first retail intelligence")
        canvas.drawRightString(pagesize[0] - 14 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    logger.info(
        "PDF report written",
        extra={"report_type": report_type, "rows": len(rows), "pagesize": str(pagesize)},
    )
    _ = PageBreak
    return path
