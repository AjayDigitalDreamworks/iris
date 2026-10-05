"""Alert service — rule evaluation, deduplication, lifecycle, evidence.

Lifecycle (README §5.2)::

    CANDIDATE -> SUPPRESSED          (low confidence / transient)
    CANDIDATE -> ACTIVE              (rule + persistence checks passed)
    ACTIVE     -> ACKNOWLEDGED       (staff acknowledges)
    ACTIVE/ACK -> RESOLVED           (condition cleared or authorised manual resolve)

A threshold that is violated on 200 consecutive frames must produce **one**
alert that is refreshed, not 200 alerts.  That is enforced by the
``dedup_key`` + ``cooldown_seconds`` pair plus a persistence check.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.models import ACTIVE_STATUSES, Alert, AlertSeverity, AlertStatus, AlertType
from app.services.db_service import run_in_session, settings_service
from app.vision.rules import (
    AlertThresholds,
    StockState,
    build_dedup_key,
    in_cooldown,
    should_create_low_stock_alert,
    should_resolve_alert,
)

logger = get_logger("app")


@dataclass
class RuleOutcome:
    """The result of evaluating one business rule for one subject."""

    action: str  # 'raise' | 'refresh' | 'suppress' | 'resolve' | 'none'
    alert_type: str | None = None
    severity: str = AlertSeverity.INFO.value
    title: str = ""
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    dedup_key: str = ""
    reason: str = ""


@dataclass
class ObservationState:
    """Per-subject observation memory used for persistence checks."""

    consecutive: int = 0
    history: list[Any] = field(default_factory=list)

    def push(self, value: Any) -> None:
        self.consecutive += 1
        self.history.append(value)
        if len(self.history) > 20:
            del self.history[:-20]


class AlertService:
    """Owns alert creation, deduplication and state transitions."""

    def __init__(self) -> None:
        self._observations: dict[str, ObservationState] = defaultdict(ObservationState)
        self.created = 0
        self.suppressed = 0
        self.refreshed = 0
        self.auto_resolved = 0

    def reset(self) -> None:
        self._observations.clear()

    # ── rule evaluation (pure) ──────────────────────────────────────────────
    def evaluate_stock(
        self,
        *,
        sku: str,
        estimated_quantity: float,
        capacity_units: int,
        confidence: float,
        stock_percentage: float,
        state: StockState,
        camera_id: str | None,
        shelf_id: str | None,
        registered_quantity: int | None = None,
        thresholds: AlertThresholds | None = None,
        min_confidence: float | None = None,
    ) -> RuleOutcome:
        """Decide whether a stock condition should raise or refresh an alert."""
        thresholds = thresholds or settings_service.alert_thresholds()
        min_conf = (
            min_confidence
            if min_confidence is not None
            else settings_service.get_float("alert_min_confidence", 0.8)
        )
        key = build_dedup_key(
            f"stock:{state.value}", camera_id=camera_id, shelf_id=shelf_id, sku=sku, zone_id=None
        )
        observations = self._observations[key]
        observations.push(state)

        if state in (StockState.OK, StockState.UNKNOWN):
            if should_resolve_alert(state, observations.history):
                observations.consecutive = 0
                return RuleOutcome(
                    action="resolve",
                    dedup_key=key,
                    title=f"{sku} recovered",
                    message=f"Estimated visible stock for {sku} returned to a healthy level.",
                    reason="condition_cleared",
                    payload={"sku": sku, "state": state.value},
                )
            return RuleOutcome(action="none", dedup_key=key, reason="state_ok")

        alert_type = (
            AlertType.OUT_OF_STOCK.value
            if state == StockState.OUT_OF_STOCK
            else AlertType.LOW_STOCK.value
        )
        threshold_pct = (
            settings_service.get_float("out_of_stock_threshold_pct", 5.0)
            if state == StockState.OUT_OF_STOCK
            else settings_service.get_float("low_stock_threshold_pct", 20.0)
        )

        if not should_create_low_stock_alert(
            int(round(estimated_quantity)),
            max(0, int(round(capacity_units * threshold_pct / 100.0))) if capacity_units > 0 else 0,
            confidence,
            observations.consecutive,
            min_confidence=min_conf,
            required_observations=thresholds.required_observations,
        ):
            self.suppressed += 1
            return RuleOutcome(
                action="suppress",
                alert_type=alert_type,
                dedup_key=key,
                reason=(
                    f"persistence/confidence not met "
                    f"(observations={observations.consecutive}/{thresholds.required_observations}, "
                    f"confidence={confidence:.2f}/{min_conf:.2f})"
                ),
                payload={"observations": observations.consecutive, "confidence": confidence},
            )

        severity = (
            AlertSeverity.CRITICAL.value
            if state == StockState.OUT_OF_STOCK
            else AlertSeverity.WARNING.value
        )
        registered_note = (
            f" Registered (POS/ERP/manual) quantity: {registered_quantity}."
            if registered_quantity is not None
            else ""
        )
        return RuleOutcome(
            action="raise",
            alert_type=alert_type,
            severity=severity,
            title=f"{sku} {'out of stock' if state == StockState.OUT_OF_STOCK else 'low stock'}",
            message=(
                f"Estimated visible stock {estimated_quantity:.1f} unit(s) "
                f"({stock_percentage:.1f}% of {capacity_units} capacity, threshold {threshold_pct:.0f}%), "
                f"confidence {confidence:.2f}.{registered_note} "
                "This is a camera estimate of VISIBLE shelf stock only."
            ),
            dedup_key=key,
            payload={
                "sku": sku,
                "estimated_quantity": round(estimated_quantity, 3),
                "capacity_units": capacity_units,
                "stock_percentage": round(stock_percentage, 3),
                "threshold_pct": threshold_pct,
                "confidence": confidence,
                "observations": observations.consecutive,
                "registered_quantity": registered_quantity,
                "state": state.value,
            },
        )

    #: Consecutive normal readings required before a queue alert auto-resolves.
    #: One quiet frame is not proof the rush is over — people mill around.
    queue_resolve_observations: int = 2

    def evaluate_queue(
        self,
        *,
        zone_id: str,
        zone_name: str,
        queue_length: int,
        status: str,
        estimated_wait_seconds: float | None,
        camera_id: str | None,
        busy_threshold: int,
        critical_threshold: int,
        cooldown_seconds: int | None = None,
    ) -> list[RuleOutcome]:
        """Evaluate one queue measurement; returns raise/resolve outcomes.

        A list because one reading can carry several transitions at once: a
        queue that goes straight back to normal must resolve BOTH the busy and
        the critical alert for that zone, and a critical reading supersedes an
        active busy alert. ``apply`` no-ops a resolve when nothing is active.
        """
        busy_key = build_dedup_key(
            AlertType.QUEUE_BUSY.value, camera_id=camera_id, shelf_id=None, sku=None, zone_id=zone_id
        )
        critical_key = build_dedup_key(
            AlertType.QUEUE_CRITICAL.value, camera_id=camera_id, shelf_id=None, sku=None, zone_id=zone_id
        )

        if status == "normal":
            streak = self._observations[f"queue-normal:{zone_id}"]
            streak.push("normal")
            if streak.consecutive < self.queue_resolve_observations:
                return [RuleOutcome(action="none", reason="queue_normal_awaiting_persistence")]
            # Genuinely recovered: forget the violation memory and resolve
            # whatever queue alert is still active for this zone.
            self._observations.pop(busy_key, None)
            self._observations.pop(critical_key, None)
            note = (
                f"{zone_name} queue is back to normal ({queue_length} people, "
                f"busy ≥ {busy_threshold}). Alerts resolved automatically."
            )
            payload = {
                "zone_id": zone_id,
                "queue_length": queue_length,
                "status": status,
                "busy_threshold": busy_threshold,
                "critical_threshold": critical_threshold,
            }
            return [
                RuleOutcome(
                    action="resolve",
                    dedup_key=key,
                    title=f"Queue recovered at {zone_name}",
                    message=note,
                    reason="queue_recovered",
                    payload=payload,
                )
                for key in (critical_key, busy_key)
            ]

        # busy or critical: recovery streak is broken.
        self._observations.pop(f"queue-normal:{zone_id}", None)
        cooldown = (
            cooldown_seconds
            if cooldown_seconds is not None
            else settings_service.get_int("queue_alert_cooldown_seconds", 180)
        )
        is_critical = status == "critical"
        alert_type = AlertType.QUEUE_CRITICAL.value if is_critical else AlertType.QUEUE_BUSY.value
        key = critical_key if is_critical else busy_key
        self._observations[key].push(status)

        wait_note = (
            f"Estimated wait ≈ {estimated_wait_seconds:.0f}s. "
            "The wait is a queueing-model estimate, not a guarantee."
            if estimated_wait_seconds is not None
            else ""
        )
        if is_critical:
            title = f"OPEN ANOTHER COUNTER — queue critical at {zone_name}"
            message = (
                f"{queue_length} people are waiting at {zone_name} "
                f"(critical ≥ {critical_threshold}). Open another checkout counter "
                f"to split the queue. {wait_note}"
            ).strip()
        else:
            title = f"Queue busy at {zone_name}"
            message = (
                f"{queue_length} people in the {zone_name} queue "
                f"(busy ≥ {busy_threshold}, critical at ≥ {critical_threshold}). "
                f"{wait_note}"
            ).strip()

        outcomes = [
            RuleOutcome(
                action="raise",
                alert_type=alert_type,
                severity=(
                    AlertSeverity.CRITICAL.value if is_critical else AlertSeverity.WARNING.value
                ),
                title=title,
                message=message,
                dedup_key=key,
                payload={
                    "zone_id": zone_id,
                    "queue_length": queue_length,
                    "status": status,
                    "estimated_wait_seconds": estimated_wait_seconds,
                    "busy_threshold": busy_threshold,
                    "critical_threshold": critical_threshold,
                    "cooldown_seconds": cooldown,
                },
            )
        ]
        if is_critical:
            # A critical alert supersedes the busy one for the same zone:
            # keeping both active double-reports one problem.
            outcomes.append(
                RuleOutcome(
                    action="resolve",
                    dedup_key=busy_key,
                    title=f"Queue busy superseded at {zone_name}",
                    message=f"{zone_name} escalated to critical; the busy alert is superseded.",
                    reason="superseded_by_critical",
                    payload={"zone_id": zone_id, "status": status},
                )
            )
        else:
            # Downgrade (critical eased back to busy): the critical alert is
            # superseded by the new busy one, so the operator is not told to
            # open another counter while the queue is merely busy.
            outcomes.append(
                RuleOutcome(
                    action="resolve",
                    dedup_key=critical_key,
                    title=f"Queue critical eased at {zone_name}",
                    message=f"{zone_name} eased below the critical threshold; the critical alert is superseded by the busy alert.",
                    reason="superseded_by_busy",
                    payload={"zone_id": zone_id, "status": status},
                )
            )
        return outcomes

    def evaluate_camera(
        self,
        *,
        camera_id: str,
        camera_name: str,
        offline: bool,
        recovered: bool = False,
        is_frozen: bool = False,
        seconds_since_last_frame: float | None = None,
    ) -> RuleOutcome:
        if offline:
            return RuleOutcome(
                action="raise",
                alert_type=AlertType.CAMERA_OFFLINE.value,
                severity=AlertSeverity.CRITICAL.value,
                title=f"Camera offline: {camera_name}",
                message=(
                    f"No frame received for {seconds_since_last_frame:.0f}s."
                    if seconds_since_last_frame is not None
                    else "The camera has never produced a frame."
                ),
                dedup_key=build_dedup_key(
                    AlertType.CAMERA_OFFLINE.value, camera_id=camera_id,
                    shelf_id=None, sku=None, zone_id=None,
                ),
                payload={"camera_id": camera_id, "seconds_since_last_frame": seconds_since_last_frame},
            )
        if is_frozen:
            return RuleOutcome(
                action="raise",
                alert_type=AlertType.CAMERA_FROZEN.value,
                severity=AlertSeverity.WARNING.value,
                title=f"Camera stream looks frozen: {camera_name}",
                message="The frame signature has not changed for an extended period.",
                dedup_key=build_dedup_key(
                    AlertType.CAMERA_FROZEN.value, camera_id=camera_id,
                    shelf_id=None, sku=None, zone_id=None,
                ),
                payload={"camera_id": camera_id},
            )
        if recovered:
            return RuleOutcome(
                action="raise",
                alert_type=AlertType.CAMERA_RECOVERED.value,
                severity=AlertSeverity.INFO.value,
                title=f"Camera recovered: {camera_name}",
                message="The camera is producing frames again.",
                dedup_key=build_dedup_key(
                    AlertType.CAMERA_RECOVERED.value, camera_id=camera_id,
                    shelf_id=None, sku=None, zone_id=None,
                ),
                payload={"camera_id": camera_id},
            )
        return RuleOutcome(action="none")

    # ── persistence ─────────────────────────────────────────────────────────
    def apply(self, outcome: RuleOutcome, *, camera_id: str | None, shelf_id: str | None,
              zone_id: str | None = None, sku: str | None = None, product_id: str | None = None,
              store_id: str | None = None, evidence_path: str | None = None) -> Alert | None:
        """Create, refresh, resolve or record a suppression for an outcome."""
        if outcome.action in {"none", "suppress"}:
            if outcome.action == "suppress":
                logger.info(
                    "Alert suppressed",
                    extra={"alert_type": outcome.alert_type, "reason": outcome.reason,
                           "sku": sku, "camera_id": camera_id},
                )
            return None

        def _write(session) -> Alert | None:
            # Newest row for the key: after resolve/re-raise cycles several rows
            # share a dedup_key, and the current generation is the latest one.
            existing = (
                session.query(Alert)
                .filter(Alert.dedup_key == outcome.dedup_key)
                .order_by(Alert.created_at.desc())
                .first()
            )
            now = datetime.now(tz=timezone.utc)

            if outcome.action == "resolve":
                if existing is not None and existing.status in ACTIVE_STATUSES:
                    existing.status = AlertStatus.RESOLVED.value
                    existing.resolved_at = now
                    existing.resolution_source = "condition_resolved"
                    existing.resolution_note = outcome.message
                    existing.resolved_by = "system"
                    self.auto_resolved += 1
                    return existing
                return None

            cooldown = settings_service.get_int(
                "alert_cooldown_seconds", 300, session
            )
            if (
                existing is not None
                and existing.status in ACTIVE_STATUSES
                and not in_cooldown(existing.last_observed_at, cooldown, now=now)
            ):
                # Still inside cooldown: refresh the existing alert silently.
                existing.last_observed_at = now
                existing.occurrence_count += 1
                existing.confidence = float(outcome.payload.get("confidence", existing.confidence))
                existing.payload = {**existing.payload, **outcome.payload}
                if evidence_path:
                    existing.evidence_path = evidence_path
                self.refreshed += 1
                return existing

            if existing is not None and existing.status in ACTIVE_STATUSES:
                existing.last_observed_at = now
                existing.occurrence_count += 1
                existing.payload = {**existing.payload, **outcome.payload}
                if evidence_path:
                    existing.evidence_path = evidence_path
                self.refreshed += 1
                return existing

            # Post-resolve cooldown: an operator resolving (or the system
            # superseding) an alert does not mean the condition is gone. While
            # the condition persists, re-raising would rebuild the alert every
            # frame — the alert-storm failure mode. Keep the condition quiet
            # until the cooldown expires; after it does, a still-present
            # condition legitimately re-raises.
            if (
                existing is not None
                and existing.status == AlertStatus.RESOLVED.value
                and existing.last_observed_at is not None
                and in_cooldown(
                    existing.last_observed_at,
                    int(outcome.payload.get("cooldown_seconds") or cooldown),
                    now=now,
                )
            ):
                existing.last_observed_at = now
                self.suppressed += 1
                return None

            alert = Alert(
                store_id=store_id,
                camera_id=camera_id,
                shelf_id=shelf_id,
                zone_id=zone_id,
                product_id=product_id,
                sku=sku,
                alert_type=outcome.alert_type or AlertType.LOW_STOCK.value,
                severity=outcome.severity,
                status=AlertStatus.ACTIVE.value,
                title=outcome.title,
                message=outcome.message,
                payload=outcome.payload,
                evidence_path=evidence_path,
                confidence=float(outcome.payload.get("confidence", 0.0)),
                observation_count=int(outcome.payload.get("observations", 1) or 1),
                dedup_key=outcome.dedup_key,
                created_at=now,
                last_observed_at=now,
            )
            session.add(alert)
            session.flush()
            self.created += 1
            return alert

        return run_in_session(_write)

    # ── operator actions ────────────────────────────────────────────────────
    def acknowledge(self, alert_id: str, username: str, note: str | None = None) -> Alert:
        def _write(session) -> Alert:
            alert = session.get(Alert, alert_id)
            if alert is None:
                from app.core.errors import NotFoundError

                raise NotFoundError(f"Alert {alert_id} does not exist.")
            if alert.status == AlertStatus.RESOLVED.value:
                from app.core.errors import ConflictError

                raise ConflictError("A resolved alert cannot be acknowledged.")
            alert.status = AlertStatus.ACKNOWLEDGED.value
            alert.acknowledged_at = datetime.now(tz=timezone.utc)
            alert.acknowledged_by = username
            if note:
                alert.payload = {**(alert.payload or {}), "acknowledgement_note": note}
            session.flush()
            return alert

        return run_in_session(_write)

    def resolve(
        self, alert_id: str, username: str, note: str, resolution_source: str = "manual"
    ) -> Alert:
        def _write(session) -> Alert:
            alert = session.get(Alert, alert_id)
            if alert is None:
                from app.core.errors import NotFoundError

                raise NotFoundError(f"Alert {alert_id} does not exist.")
            now = datetime.now(tz=timezone.utc)
            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_at = now
            alert.resolved_by = username
            alert.resolution_note = note
            alert.resolution_source = resolution_source
            if alert.acknowledged_at is None:
                alert.acknowledged_at = now
                alert.acknowledged_by = username
            # Clear the persistence memory so the next violation raises fresh.
            self._observations.pop(alert.dedup_key, None)
            session.flush()
            return alert

        return run_in_session(_write)

    def stats(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "refreshed": self.refreshed,
            "suppressed": self.suppressed,
            "auto_resolved": self.auto_resolved,
            "tracked_conditions": len(self._observations),
        }


alert_service = AlertService()