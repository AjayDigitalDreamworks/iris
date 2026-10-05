"""Unit tests for the deterministic business rules engine (README §5)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.vision.rules import (
    AlertThresholds,
    QueueStatus,
    QueueThresholds,
    Severity,
    StockState,
    StockThresholds,
    build_dedup_key,
    camera_is_frozen,
    camera_is_stale,
    classify_stock,
    describe_rule,
    estimate_wait_seconds,
    evidence_allowed,
    in_cooldown,
    is_stable_observation,
    queue_status,
    should_create_low_stock_alert,
    should_resolve_alert,
    stock_percentage,
    uptime_ratio_24h,
)


class TestStockPercentage:
    def test_percentage_is_ratio_of_capacity(self):
        assert stock_percentage(24, 48) == pytest.approx(50.0)
        assert stock_percentage(0, 48) == 0.0
        assert stock_percentage(48, 48) == pytest.approx(100.0)

    def test_percentage_is_clamped(self):
        assert stock_percentage(100, 48) == 100.0
        assert stock_percentage(-5, 48) == 0.0

    def test_unknown_capacity_is_reported_as_unknown_not_zero(self):
        # Reporting 0% when capacity is unknown would be a fabricated number.
        assert stock_percentage(10, 0) == -1.0
        assert stock_percentage(10, -5) == -1.0


class TestClassifyStock:
    thresholds = StockThresholds(low_stock_pct=20.0, out_of_stock_pct=5.0, min_confidence=0.8)

    @pytest.mark.parametrize(
        ("quantity", "capacity", "confidence", "expected"),
        [
            (48, 48, 0.95, StockState.OK),
            (10, 48, 0.95, StockState.OK),               # 20.83% is above 20%
            (9, 48, 0.95, StockState.LOW_STOCK),         # 18.75%
            (2, 48, 0.95, StockState.OUT_OF_STOCK),      # 4.2% <= 5
            (24, 0, 0.95, StockState.UNKNOWN),           # capacity unknown
            (24, 48, 0.1, StockState.UNKNOWN),           # very low confidence
        ],
    )
    def test_states(self, quantity, capacity, confidence, expected):
        assert classify_stock(quantity, capacity, confidence, self.thresholds) is expected

    def test_low_confidence_never_claims_out_of_stock(self):
        """A wrong 'the shelf is empty' claim is operationally expensive."""
        result = classify_stock(0, 48, 0.45, self.thresholds)
        assert result is StockState.LOW_STOCK
        assert result is not StockState.OUT_OF_STOCK

    def test_thresholds_are_configurable(self):
        strict = StockThresholds(low_stock_pct=90.0, out_of_stock_pct=70.0)
        assert classify_stock(30, 48, 0.95, strict) is StockState.OUT_OF_STOCK
        assert classify_stock(30, 48, 0.95, self.thresholds) is StockState.OK


class TestLowStockRule:
    """The reference rule from README §5.3."""

    def test_fires_when_all_conditions_met(self):
        assert should_create_low_stock_alert(8, 10, 0.91, 3) is True

    def test_requires_confidence(self):
        assert should_create_low_stock_alert(8, 10, 0.5, 5) is False

    def test_requires_persistence(self):
        assert should_create_low_stock_alert(8, 10, 0.95, 2) is False

    def test_requires_below_threshold(self):
        assert should_create_low_stock_alert(12, 10, 0.95, 5) is False

    def test_thresholds_are_overridable(self):
        assert should_create_low_stock_alert(12, 10, 0.95, 5, required_observations=5) is False
        assert should_create_low_stock_alert(12, 15, 0.95, 5) is True


class TestStability:
    def test_flat_series_is_stable(self):
        assert is_stable_observation(10, [10, 10, 10]) is True

    def test_wild_series_is_not_stable(self):
        assert is_stable_observation(10, [1, 40, 3]) is False

    def test_too_few_samples_is_not_stable(self):
        assert is_stable_observation(10, [], min_samples=2) is False


class TestQueueRules:
    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError):
            QueueThresholds(busy=9, critical=2)

    @pytest.mark.parametrize(
        ("length", "expected"),
        [(0, QueueStatus.NORMAL), (3, QueueStatus.NORMAL), (4, QueueStatus.BUSY),
         (7, QueueStatus.CRITICAL), (20, QueueStatus.CRITICAL)],
    )
    def test_status_transitions(self, length, expected):
        assert queue_status(length, QueueThresholds(busy=4, critical=7)) is expected

    def test_thresholds_are_configurable(self):
        relaxed = QueueThresholds(busy=10, critical=20)
        assert queue_status(12, relaxed) is QueueStatus.BUSY
        assert queue_status(12, QueueThresholds(busy=4, critical=7)) is QueueStatus.CRITICAL

    def test_wait_is_zero_for_empty_queue(self):
        assert estimate_wait_seconds(0, QueueThresholds()) == 0.0

    def test_wait_increases_with_length(self):
        thresholds = QueueThresholds(service_time_seconds=120.0)
        short = estimate_wait_seconds(2, thresholds)
        long = estimate_wait_seconds(8, thresholds)
        assert 0 < short < long

    def test_wait_is_bounded_for_an_overloaded_system(self):
        """An unstable system must return a finite, bounded sentinel, never inf."""
        thresholds = QueueThresholds(busy=1, critical=2, service_time_seconds=60.0)
        value = estimate_wait_seconds(500, thresholds, servers=1)
        assert value == pytest.approx(60.0 * 500)


class TestDeduplication:
    def test_key_is_stable(self):
        args = dict(camera_id="cam1", shelf_id="s1", sku="SKU-1", zone_id=None)
        assert build_dedup_key("low", **args) == build_dedup_key("low", **args)

    def test_key_differs_per_subject(self):
        base = dict(camera_id="cam1", shelf_id="s1", sku="SKU-1", zone_id=None)
        assert build_dedup_key("low", **base) != build_dedup_key("low", **{**base, "sku": "SKU-2"})
        assert build_dedup_key("low", **base) != build_dedup_key("out", **base)

    def test_cooldown_blocks_immediate_resend(self):
        now = datetime.now(tz=timezone.utc)
        assert in_cooldown(now, 300, now=now) is True
        assert in_cooldown(now - timedelta(seconds=400), 300, now=now) is False

    def test_never_observed_is_not_in_cooldown(self):
        assert in_cooldown(None, 300) is False


class TestResolution:
    def test_requires_sustained_recovery(self):
        history = [StockState.OK]
        assert should_resolve_alert(StockState.OK, history) is False

    def test_resolves_after_repeated_recovery(self):
        history = [StockState.OK, StockState.OK]
        assert should_resolve_alert(StockState.OK, history) is True

    def test_does_not_resolve_while_low(self):
        history = [StockState.LOW_STOCK, StockState.OK]
        assert should_resolve_alert(StockState.LOW_STOCK, history) is False

    def test_unknown_state_blocks_auto_resolution(self):
        history = [StockState.UNKNOWN, StockState.UNKNOWN]
        assert should_resolve_alert(StockState.OK, history) is False


class TestCameraHealthRules:
    def test_never_produced_a_frame_is_stale(self):
        assert camera_is_stale(None, 60.0) is True

    def test_recent_frame_is_fresh(self):
        assert camera_is_stale(5.0, 60.0) is False

    def test_long_gap_is_stale(self):
        assert camera_is_stale(120.0, 60.0) is True

    def test_frozen_detection(self):
        assert camera_is_frozen(30.0, 15.0) is True
        assert camera_is_frozen(5.0, 15.0) is False
        assert camera_is_frozen(None, 15.0) is False

    def test_uptime_ratio(self):
        assert uptime_ratio_24h(90, 100) == pytest.approx(0.9)
        assert uptime_ratio_24h(0, 0) is None


class TestEvidencePrivacy:
    def test_evidence_can_be_disabled(self):
        assert evidence_allowed(False) is False

    def test_demo_frames_never_store_evidence(self):
        assert evidence_allowed(True, is_demo=True) is False

    def test_enabled_allows_real_frames(self):
        assert evidence_allowed(True, is_demo=False) is True


class TestSeverity:
    def test_severity_matches_state(self):
        assert severity_is("out_of_stock", Severity.CRITICAL)
        assert severity_is("low_stock", Severity.WARNING)
        assert severity_is("ok", Severity.INFO)


def severity_is(state: str, expected: Severity) -> bool:
    from app.vision.rules import severity_for_stock_state

    return severity_for_stock_state(StockState(state)) is expected


class TestAlertThresholds:
    def test_defaults_are_sane(self):
        thresholds = AlertThresholds()
        assert thresholds.required_observations >= 1
        assert 0.0 <= thresholds.min_confidence <= 1.0
        assert thresholds.cooldown_seconds >= 0


class TestDescribeRule:
    def test_description_includes_parameters(self):
        text = describe_rule("low_stock", {"threshold": 10, "confidence": 0.9})
        assert "low_stock" in text and "threshold=10" in text