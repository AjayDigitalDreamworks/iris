"""Inference scheduler tests (README §3, Pipeline A step 5).

The scheduler decides which AI tasks run on which frame. These tests pin the
rates from the README's sampling table, because a scheduler that silently runs
everything on every frame defeats its own purpose while still looking healthy.
"""

from __future__ import annotations

import pytest

from app.vision.scheduler import (
    DETECTION,
    FOOTFALL,
    PERSON_TRACKING,
    PLANOGRAM,
    QUEUE,
    RECOGNITION,
    ROLE_PROFILES,
    SHELF_ANALYSIS,
    InferenceScheduler,
)


def run_frames(scheduler: InferenceScheduler, task: str, count: int) -> list[bool]:
    """Advance `count` delivered frames and record whether `task` ran."""
    decisions = []
    for _ in range(count):
        scheduler.begin_frame()
        decisions.append(scheduler.allow(task))
    return decisions


class TestRoleProfiles:
    def test_entrance_camera_follows_people_fast(self):
        """README table: Entrance — person detection and tracking at 10 FPS."""
        scheduler = InferenceScheduler(role="entrance", capture_fps=10.0)
        assert scheduler.target_for(PERSON_TRACKING) == 10.0
        assert scheduler.target_for(FOOTFALL) == 10.0

    def test_shelf_camera_runs_product_work_slowly(self):
        """README table: Shelf aisle — detection 3-5 FPS, segmentation 1 FPS."""
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        assert scheduler.target_for(DETECTION) == 4.0
        assert scheduler.target_for(SHELF_ANALYSIS) == 1.0
        assert scheduler.target_for(RECOGNITION) == 1.0
        # Planogram is periodic by design, so it is well below 1 FPS.
        assert scheduler.target_for(PLANOGRAM) < 1.0

    def test_checkout_camera_watches_the_queue(self):
        """README table: Checkout — queue monitoring at 5 FPS."""
        scheduler = InferenceScheduler(role="checkout", capture_fps=8.0)
        assert scheduler.target_for(QUEUE) == 5.0

    def test_unknown_role_falls_back_to_the_default_profile(self):
        scheduler = InferenceScheduler(role="nonsense", capture_fps=5.0)
        assert scheduler.role == "other"
        assert scheduler.target_for(DETECTION) > 0

    def test_a_task_can_never_exceed_the_delivery_rate(self):
        """Asking for 10 FPS on a camera delivering 3 FPS must yield 3 FPS."""
        scheduler = InferenceScheduler(role="entrance", capture_fps=3.0)
        assert scheduler.target_for(PERSON_TRACKING) == 3.0


class TestGating:
    def test_first_frame_always_runs(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        assert run_frames(scheduler, SHELF_ANALYSIS, 1) == [True]

    def test_a_one_fps_task_runs_once_per_five_delivered_frames(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        decisions = run_frames(scheduler, SHELF_ANALYSIS, 10)
        assert decisions == [
            True, False, False, False, False,
            True, False, False, False, False,
        ]

    def test_detection_runs_every_frame_when_the_rate_matches(self):
        scheduler = InferenceScheduler(role="entrance", capture_fps=10.0)
        assert all(run_frames(scheduler, DETECTION, 6))

    def test_a_half_rate_task_runs_every_other_frame(self):
        scheduler = InferenceScheduler(role="checkout", capture_fps=10.0)
        decisions = run_frames(scheduler, QUEUE, 4)
        assert decisions == [True, False, True, False]

    def test_a_task_never_running_never_runs(self):
        scheduler = InferenceScheduler(role="entrance", capture_fps=10.0)
        # Queue is not part of the entrance profile at all.
        assert not any(run_frames(scheduler, QUEUE, 20))

    def test_an_unknown_task_never_runs(self):
        """A typo in a task name must not silently create a new hot path."""
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        assert not any(run_frames(scheduler, "detecton", 10))

    def test_tasks_are_independent(self):
        """Gating shelf analysis must not gate detection."""
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        shelf_runs = []
        for _ in range(5):
            scheduler.begin_frame()
            shelf_runs.append(scheduler.allow(SHELF_ANALYSIS))
            assert scheduler.allow(DETECTION) is True
        # Detection ran on all 5 frames; shelf analysis ran once.
        assert shelf_runs.count(True) == 1


class TestRuntimeOverrides:
    def test_set_target_changes_the_rate(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=10.0)
        scheduler.set_target(SHELF_ANALYSIS, 10.0)
        assert scheduler.target_for(SHELF_ANALYSIS) == 10.0
        assert all(run_frames(scheduler, SHELF_ANALYSIS, 5))

    def test_set_target_cannot_exceed_the_delivery_rate(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=4.0)
        scheduler.set_target(SHELF_ANALYSIS, 30.0)
        assert scheduler.target_for(SHELF_ANALYSIS) == 4.0

    def test_set_target_to_zero_disables_a_task(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        scheduler.set_target(SHELF_ANALYSIS, 0.0)
        assert not any(run_frames(scheduler, SHELF_ANALYSIS, 10))

    def test_set_target_can_add_a_new_task(self):
        scheduler = InferenceScheduler(role="entrance", capture_fps=10.0)
        scheduler.set_target(QUEUE, 5.0)
        decisions = run_frames(scheduler, QUEUE, 4)
        assert decisions == [True, False, True, False]


class TestStats:
    def test_stats_report_targets_and_achieved_rates(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        run_frames(scheduler, SHELF_ANALYSIS, 20)
        stats = scheduler.stats()
        assert stats["role"] == "shelf"
        assert stats["frames_seen"] == 20
        task = stats["tasks"][SHELF_ANALYSIS]
        assert task["target_fps"] == 1.0
        assert task["frames_run"] == 4  # frames 1, 6, 11, 16
        assert task["achieved_fps"] == pytest.approx(1.0, abs=0.05)

    def test_stats_are_empty_before_any_frame(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        stats = scheduler.stats()
        assert stats["frames_seen"] == 0
        assert all(t["frames_run"] == 0 for t in stats["tasks"].values())

    def test_reset_clears_counters(self):
        scheduler = InferenceScheduler(role="shelf", capture_fps=5.0)
        run_frames(scheduler, SHELF_ANALYSIS, 10)
        scheduler.reset()
        stats = scheduler.stats()
        assert stats["frames_seen"] == 0
        assert stats["tasks"][SHELF_ANALYSIS]["frames_run"] == 0
        # And gating starts fresh, so the next frame runs again.
        assert run_frames(scheduler, SHELF_ANALYSIS, 1) == [True]


class TestProfileSanity:
    @pytest.mark.parametrize("role", sorted(ROLE_PROFILES))
    def test_no_profile_asks_for_more_than_the_readme_allows(self, role):
        """The README's sampling table caps detection at 10 FPS."""
        for task, fps in ROLE_PROFILES[role].items():
            assert fps <= 10.0, f"{role}/{task} targets {fps} FPS"

    @pytest.mark.parametrize("role", sorted(ROLE_PROFILES))
    def test_every_profile_can_detect(self, role):
        """Without detection nothing downstream has anything to work on."""
        assert ROLE_PROFILES[role].get(DETECTION, 0) > 0
