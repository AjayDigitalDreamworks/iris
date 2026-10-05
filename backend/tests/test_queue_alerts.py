"""Queue alert lifecycle: raise → critical supersedes busy → auto-resolve.

Exercises the alert-service contract the pipeline depends on:

* a busy reading raises one WARNING alert (deduped, refreshed on repeats);
* a critical reading raises the "OPEN ANOTHER COUNTER" CRITICAL alert and
  emits a resolve outcome that supersedes the still-active busy alert;
* recovery needs TWO consecutive normal readings, then the queue alert for
  the zone is resolved automatically by the system.

Uses a committed store + camera + queue zone because ``alerts`` has real
foreign keys into all three.
"""

from __future__ import annotations

import pytest

from app.models import (
    ACTIVE_STATUSES,
    Alert,
    AlertStatus,
    AlertType,
    Camera,
    CameraRole,
    CameraStatus,
    CameraZone,
    Store,
    ZoneType,
)


@pytest.fixture()
def queue_camera(db_session):
    """A committed store + checkout camera + queue zone, removed afterwards."""
    store = Store(code="TEST-QUEUE-STORE", name="Queue test store", timezone="UTC")
    db_session.add(store)
    db_session.flush()
    camera = Camera(
        store_id=store.id,
        name="Queue Test Camera",
        role=CameraRole.CHECKOUT.value,
        source_type="synthetic",
        capture_fps=2.0,
        is_enabled=False,
        status=CameraStatus.OFFLINE.value,
        notes="queue alert test fixture",
    )
    db_session.add(camera)
    db_session.flush()
    zone = CameraZone(
        camera_id=camera.id,
        name="Front checkout queue",
        zone_type=ZoneType.QUEUE.value,
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        is_active=True,
    )
    db_session.add(zone)
    db_session.commit()
    try:
        yield {"camera": camera.id, "zone": zone.id, "store": store.id}
    finally:
        db_session.query(Alert).filter(Alert.camera_id == camera.id).delete()
        db_session.commit()
        db_session.delete(zone)
        db_session.delete(camera)
        db_session.delete(store)
        db_session.commit()


class QueueScenario:
    """Evaluate + apply through the real service, against the real DB."""

    def __init__(self, ids: dict) -> None:
        from app.services.alert_service import AlertService
        from app.vision.rules import build_dedup_key

        self.service = AlertService()
        self.camera_id = ids["camera"]
        self.zone_id = ids["zone"]
        self.store_id = ids["store"]
        self.busy_key = build_dedup_key(
            AlertType.QUEUE_BUSY.value,
            camera_id=self.camera_id,
            shelf_id=None,
            sku=None,
            zone_id=self.zone_id,
        )
        self.critical_key = build_dedup_key(
            AlertType.QUEUE_CRITICAL.value,
            camera_id=self.camera_id,
            shelf_id=None,
            sku=None,
            zone_id=self.zone_id,
        )

    def evaluate(self, status: str, queue_length: int):
        return self.service.evaluate_queue(
            zone_id=self.zone_id,
            zone_name="Front checkout queue",
            queue_length=queue_length,
            status=status,
            estimated_wait_seconds=120.0,
            camera_id=self.camera_id,
            busy_threshold=4,
            critical_threshold=7,
        )

    def apply(self, outcome):
        return self.service.apply(
            outcome,
            camera_id=self.camera_id,
            shelf_id=None,
            zone_id=self.zone_id,
            store_id=self.store_id,
        )

    def raise_and_apply(self, status: str, queue_length: int):
        """Apply every actionable outcome, return the DB rows touched."""
        rows = []
        for outcome in self.evaluate(status, queue_length):
            if outcome.action not in {"raise", "resolve"}:
                continue
            applied = self.apply(outcome)
            if applied is not None:
                rows.append(applied)
        return rows

    def alert_for(self, dedup_key: str, db_session) -> Alert | None:
        return (
            db_session.query(Alert).filter(Alert.dedup_key == dedup_key).one_or_none()
        )


@pytest.fixture()
def scenario(queue_camera):
    return QueueScenario(queue_camera)


class TestQueueRaise:
    def test_busy_reading_raises_a_warning_alert(self, db_session, scenario):
        outcomes = scenario.evaluate("busy", queue_length=5)

        assert [o.action for o in outcomes] == ["raise"]
        outcome = outcomes[0]
        assert outcome.alert_type == AlertType.QUEUE_BUSY.value
        assert outcome.severity == "warning"
        assert outcome.dedup_key == scenario.busy_key

        created = scenario.apply(outcome)
        assert created is not None
        row = scenario.alert_for(scenario.busy_key, db_session)
        assert row is not None
        assert row.status == AlertStatus.ACTIVE.value
        assert row.zone_id == scenario.zone_id
        assert row.camera_id == scenario.camera_id

    def test_repeated_critical_readings_refresh_one_alert(self, db_session, scenario):
        """One long rush = one alert refreshed, not an alert per frame."""
        scenario.raise_and_apply("critical", queue_length=8)
        scenario.raise_and_apply("critical", queue_length=9)

        rows = (
            db_session.query(Alert)
            .filter(Alert.dedup_key == scenario.critical_key)
            .all()
        )
        assert len(rows) == 1
        assert rows[0].occurrence_count == 2
        assert rows[0].status == AlertStatus.ACTIVE.value


class TestQueueCriticalSupersedesBusy:
    def test_critical_raises_open_another_counter_and_resolves_busy(
        self, db_session, scenario
    ):
        scenario.raise_and_apply("busy", queue_length=5)

        outcomes = scenario.evaluate("critical", queue_length=8)
        assert [o.action for o in outcomes] == ["raise", "resolve"]
        raise_outcome, supersede = outcomes

        assert raise_outcome.dedup_key == scenario.critical_key
        assert raise_outcome.severity == "critical"
        assert "OPEN ANOTHER COUNTER" in raise_outcome.title
        assert "another checkout counter" in raise_outcome.message.lower()
        assert supersede.dedup_key == scenario.busy_key
        assert supersede.reason == "superseded_by_critical"

        scenario.apply(raise_outcome)
        scenario.apply(supersede)

        busy_row = scenario.alert_for(scenario.busy_key, db_session)
        critical_row = scenario.alert_for(scenario.critical_key, db_session)
        assert busy_row is not None
        assert busy_row.status == AlertStatus.RESOLVED.value
        assert critical_row is not None
        assert critical_row.status == AlertStatus.ACTIVE.value


class TestQueueAutoResolve:
    def test_single_normal_reading_does_not_resolve_yet(self, db_session, scenario):
        scenario.raise_and_apply("critical", queue_length=8)

        outcomes = scenario.evaluate("normal", queue_length=2)
        assert [o.action for o in outcomes] == ["none"]

        row = scenario.alert_for(scenario.critical_key, db_session)
        assert row is not None and row.status in ACTIVE_STATUSES

    def test_two_normal_readings_resolve_the_active_alert(self, db_session, scenario):
        scenario.raise_and_apply("critical", queue_length=8)

        first = scenario.evaluate("normal", queue_length=2)
        assert [o.action for o in first] == ["none"]

        # Second consecutive normal reading: the active alert must be handed
        # back for resolve (the superseded busy key resolves as a no-op).
        second = scenario.evaluate("normal", queue_length=1)
        assert all(o.action == "resolve" for o in second)
        assert scenario.critical_key in {o.dedup_key for o in second}

        resolved = scenario.raise_and_apply("normal", queue_length=1)
        assert resolved, "the still-active critical alert should be resolved"
        for alert in resolved:
            assert alert.status == AlertStatus.RESOLVED.value
            assert alert.resolved_by == "system"
            assert alert.resolution_source == "condition_resolved"

        row = scenario.alert_for(scenario.critical_key, db_session)
        assert row is not None
        assert row.status == AlertStatus.RESOLVED.value

    def test_normal_then_busy_breaks_the_recovery_streak(self, db_session, scenario):
        """normal → busy → normal must not resolve on the lone normal."""
        scenario.raise_and_apply("critical", queue_length=8)

        scenario.evaluate("normal", queue_length=2)  # streak = 1
        busy = scenario.evaluate("busy", queue_length=5)  # streak reset
        assert [o.action for o in busy] == ["raise"]

        outcomes = scenario.evaluate("normal", queue_length=3)  # streak = 1
        assert [o.action for o in outcomes] == ["none"]

        row = scenario.alert_for(scenario.critical_key, db_session)
        assert row is not None and row.status in ACTIVE_STATUSES
