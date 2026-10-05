"""Deterministic business rules.

This module is intentionally **pure**: no database, no network, no model. Every
threshold arrives as an argument (defaults come from the runtime settings
table), so the same rule can be unit tested, explained to an operator, and
changed without redeploying.  README §5 requires business logic to be separate
from neural-network inference; this file is that boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class StockState(str, Enum):
    OK = "ok"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    UNKNOWN = "unknown"


class QueueStatus(str, Enum):
    NORMAL = "normal"
    BUSY = "busy"
    CRITICAL = "critical"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class StockThresholds:
    """Configurable inventory thresholds (percentages of shelf capacity)."""

    low_stock_pct: float = 20.0
    out_of_stock_pct: float = 5.0
    min_confidence: float = 0.80
    required_observations: int = 3


@dataclass(frozen=True)
class QueueThresholds:
    busy: int = 4
    critical: int = 7
    service_time_seconds: float = 150.0

    def __post_init__(self) -> None:
        if self.critical < self.busy:
            raise ValueError("critical threshold must be >= busy threshold")


@dataclass(frozen=True)
class AlertThresholds:
    cooldown_seconds: int = 300
    min_confidence: float = 0.80
    required_observations: int = 3


# ── Inventory rules ──────────────────────────────────────────────────────────
def stock_percentage(estimated_quantity: float, capacity_units: int) -> float:
    """Visible stock as a percentage of configured capacity.

    ``capacity_units <= 0`` means "capacity unknown", which is a real and common
    situation.  We then report the percentage as unknown (``-1.0``) rather than
    inventing a denominator.
    """
    if capacity_units is None or capacity_units <= 0:
        return -1.0
    return max(0.0, min(100.0, (float(estimated_quantity) / float(capacity_units)) * 100.0))


def classify_stock(
    estimated_quantity: float,
    capacity_units: int,
    confidence: float,
    thresholds: StockThresholds,
) -> StockState:
    """Classify a stock estimate into a business state.

    A low-confidence estimate never triggers ``OUT_OF_STOCK``: a wrong "the shelf
    is empty" claim is operationally expensive.
    """
    pct = stock_percentage(estimated_quantity, capacity_units)
    if pct < 0:
        return StockState.UNKNOWN
    if confidence < thresholds.min_confidence * 0.5:
        return StockState.UNKNOWN
    if pct <= thresholds.out_of_stock_pct:
        return StockState.OUT_OF_STOCK if confidence >= thresholds.min_confidence else StockState.LOW_STOCK
    if pct <= thresholds.low_stock_pct:
        return StockState.LOW_STOCK
    return StockState.OK


def should_create_low_stock_alert(
    estimated_quantity: int,
    threshold: int,
    confidence: float,
    stable_observations: int,
    min_confidence: float = 0.80,
    required_observations: int = 3,
) -> bool:
    """The reference rule from README §5.3.

    Returns True only when the estimate is below the threshold, confident
    enough, and has been stable across enough consecutive observations.
    """
    return (
        estimated_quantity <= threshold
        and confidence >= min_confidence
        and stable_observations >= required_observations
    )


def is_stable_observation(
    new_quantity: float,
    history: list[float],
    *,
    tolerance_ratio: float = 0.20,
    min_samples: int = 2,
) -> bool:
    """True when the recent history is tight around ``new_quantity``.

    ``tolerance_ratio`` is the maximum relative spread (max-min)/mean allowed.
    """
    samples = [float(v) for v in ([*history[-9:], new_quantity])]
    if len(samples) < min_samples:
        return False
    mean = sum(samples) / len(samples)
    if mean <= 0:
        return max(samples) - min(samples) <= 0.5
    return (max(samples) - min(samples)) / mean <= tolerance_ratio


def severity_for_stock_state(state: StockState, discrepancy_ratio: float | None = None) -> Severity:
    if state == StockState.OUT_OF_STOCK:
        return Severity.CRITICAL
    if state == StockState.LOW_STOCK:
        return Severity.WARNING
    if state == StockState.UNKNOWN:
        return Severity.INFO
    if discrepancy_ratio is not None and discrepancy_ratio >= 0.5:
        return Severity.WARNING
    return Severity.INFO


# ── Queue rules ──────────────────────────────────────────────────────────────
def queue_status(queue_length: int, thresholds: QueueThresholds) -> QueueStatus:
    if queue_length >= thresholds.critical:
        return QueueStatus.CRITICAL
    if queue_length >= thresholds.busy:
        return QueueStatus.BUSY
    return QueueStatus.NORMAL


def estimate_wait_seconds(
    queue_length: int, thresholds: QueueThresholds, *, servers: int = 1
) -> float:
    """ESTIMATE of waiting time from what was actually observed.

    A textbook M/M/c Erlang-C wait needs an arrival rate lambda, which a single
    camera does not measure. Inventing one (the earlier implementation assumed
    lambda == mu) makes the number meaningless, so the estimate here is derived
    only from measured inputs:

    * ``queue_length``  — people counted waiting right now,
    * ``servers``       — service stations configured for the zone,
    * ``service_time_seconds`` — configured mean service time.

    Model: a person arriving now is behind at most ``queue_length`` people, and
    ``servers`` stations serve them in parallel, so the wait is
    ``queue_length * mean_service_time / servers``. It is bounded above by
    ``queue_length * mean_service_time`` (the fully serial case) so the value
    stays finite under overload.

    This is a planning-grade ESTIMATE under stated assumptions — it ignores
    service-time variance, station downtime and arrivals during the wait. It is
    never a promise or a SLA figure.
    """
    servers = max(1, servers)
    if queue_length <= 0:
        return 0.0
    service_time = max(1.0, thresholds.service_time_seconds)
    parallel = service_time * queue_length / servers
    serial_cap = service_time * queue_length
    return float(max(0.0, min(parallel, serial_cap)))





# ── Alert deduplication / cooldown ────────────────────────────────────────────
def build_dedup_key(
    alert_type: str, *, camera_id: str | None, shelf_id: str | None, sku: str | None, zone_id: str | None
) -> str:
    """Stable key identifying 'the same alert' across consecutive frames."""
    parts = [alert_type, camera_id or "-", shelf_id or "-", zone_id or "-", sku or "-"]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:48]


def in_cooldown(
    last_observed_at: datetime | None, cooldown_seconds: int, *, now: datetime | None = None
) -> bool:
    """True when the previous alert for this key is still inside its cooldown."""
    if last_observed_at is None:
        return False
    reference = now or datetime.now(tz=timezone.utc)
    if last_observed_at.tzinfo is None:
        last_observed_at = last_observed_at.replace(tzinfo=timezone.utc)
    return reference - last_observed_at < timedelta(seconds=max(0, cooldown_seconds))


def should_resolve_alert(
    current_state: StockState, state_history: list[StockState], *, recover_observations: int = 2
) -> bool:
    """Auto-resolve once the condition has genuinely cleared for N observations.

    A single recovered frame is not enough — restocking and occlusion both cause
    transient recoveries. ``UNKNOWN`` explicitly does **not** count as recovery:
    "we could not tell" is not "the problem is gone".
    """
    if current_state is not StockState.OK:
        return False
    if len(state_history) < recover_observations:
        return False
    return all(state is StockState.OK for state in state_history[-recover_observations:])


# ── Camera health rules ──────────────────────────────────────────────────────
def camera_is_stale(seconds_since_last_frame: float | None, stale_seconds: float) -> bool:
    if seconds_since_last_frame is None:
        return True  # never produced a frame
    return seconds_since_last_frame > stale_seconds


def camera_is_frozen(frame_signature_age: float | None, frozen_seconds: float) -> bool:
    if frame_signature_age is None:
        return False
    return frame_signature_age > frozen_seconds


def uptime_ratio_24h(samples_online: int, samples_total: int) -> float | None:
    if samples_total <= 0:
        return None
    return round(samples_online / samples_total, 4)


# ── Evidence / privacy ───────────────────────────────────────────────────────
def evidence_allowed(evidence_enabled: bool, *, is_demo: bool = False) -> bool:
    """Evidence capture can be disabled globally; demo frames never store media."""
    return bool(evidence_enabled) and not is_demo


def describe_rule(name: str, params: dict[str, Any]) -> str:
    """Human-readable rule description used in alert payloads."""
    rendered = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{name}({rendered})"