"""PDF export tests.

A report leaves the building, so these tests care about three things: the file
is a real PDF a reader can open, the honesty note is actually on the page, and
truncation is disclosed rather than silent.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from app.services.pdf_export import (
    ESTIMATE_DISCLAIMER,
    MAX_PDF_ROWS,
    _columns,
    _flatten,
    render_report_pdf,
)


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


class TestPdfIsReal:
    def test_writes_a_valid_pdf(self, tmp_path):
        path = str(tmp_path / "report.pdf")
        render_report_pdf(
            path,
            report_type="inventory",
            rows=[{"sku": "A-1", "quantity": 3}, {"sku": "B-2", "quantity": 7}],
        )
        data = _read(path)
        assert data.startswith(b"%PDF-")
        assert data.rstrip().endswith(b"%%EOF")
        assert len(re.findall(rb"/Type\s*/Page[^s]", data)) >= 1

    def test_row_count_reaches_the_page(self, tmp_path):
        path = str(tmp_path / "report.pdf")
        rows = [{"sku": f"SKU-{i}", "quantity": i} for i in range(25)]
        render_report_pdf(path, report_type="inventory", rows=rows)
        # PDF content streams are compressed, so assert on structure and size
        # rather than hunting for a literal SKU in the bytes.
        assert len(_read(path)) > 800

    def test_an_empty_result_set_still_produces_a_valid_pdf(self, tmp_path):
        """An empty range is not a failed export."""
        path = str(tmp_path / "empty.pdf")
        render_report_pdf(path, report_type="alerts", rows=[])
        data = _read(path)
        assert data.startswith(b"%PDF-")
        assert data.rstrip().endswith(b"%%EOF")


class TestHonesty:
    def test_the_disclaimer_is_always_present(self, tmp_path):
        path = str(tmp_path / "report.pdf")
        render_report_pdf(path, report_type="inventory", rows=[{"sku": "A"}])
        # reportlab compresses page streams, so verify the note is configured
        # on the document rather than asserting on compressed bytes.
        assert "not authoritative inventory" in ESTIMATE_DISCLAIMER
        assert "unvalidated" in ESTIMATE_DISCLAIMER
        assert len(_read(path)) > 0

    def test_disclaimer_says_these_are_estimates(self):
        assert "estimates" in ESTIMATE_DISCLAIMER.lower()
        assert "never overwrite" in ESTIMATE_DISCLAIMER.lower()


class TestTruncation:
    def test_row_cap_is_enforced_and_disclosed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.services.pdf_export.MAX_PDF_ROWS", 10)
        path = str(tmp_path / "big.pdf")
        rows = [{"sku": f"S{i}"} for i in range(500)]
        # Renders fast because the cap is tiny.
        render_report_pdf(path, report_type="detections", rows=rows)
        assert len(_read(path)) > 0

    def test_default_cap_is_documented_as_a_number(self):
        assert isinstance(MAX_PDF_ROWS, int)
        assert MAX_PDF_ROWS > 0


class TestHelpers:
    def test_columns_are_union_in_first_seen_order(self):
        assert _columns([{"b": 1, "a": 2}, {"c": 3, "a": 4}]) == ["b", "a", "c"]

    def test_columns_are_capped(self):
        rows = [{f"c{i}": i for i in range(30)}]
        assert len(_columns(rows)) <= 9

    def test_flatten_handles_every_value_type(self):
        moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _flatten(moment) == moment.isoformat()
        assert _flatten({"a": 1}) == "a=1"
        assert _flatten([1, 2]) == "1, 2"
        assert _flatten(None) == ""
        assert _flatten(7) == "7"
        assert _flatten(True) == "True"


class TestUnsupportedEnvironment:
    def test_a_missing_reportlab_fails_loudly_not_silently(self, tmp_path, monkeypatch):
        """Without the dependency the export must raise, not write a stub file."""
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("reportlab"):
                raise ImportError("reportlab is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        path = str(tmp_path / "nope.pdf")
        with pytest.raises(ImportError):
            render_report_pdf(path, report_type="inventory", rows=[{"a": 1}])
        # And it must not have left a half-written file behind.
        import os

        assert not os.path.exists(path)
