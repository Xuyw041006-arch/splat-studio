"""Regression checks for observed training ETA without running a trainer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.training_progress import TrainingProgress, fresh_training_progress


def training_line(step, total=1000, rate="100.00it/s"):
    suffix = f", {rate}" if rate else ""
    return f"Training progress: 30%|███       | {step}/{total} [00:03<00:07{suffix}]"


def stable_progress():
    tracker = TrainingProgress()
    for at, step in [(100, 100), (101, 200), (102, 300)]:
        progress = tracker.update(training_line(step), now=at)
    assert progress["eta_seconds"] == pytest.approx(7)
    return tracker, progress


def test_tqdm_requires_three_observations_and_two_seconds():
    tracker = TrainingProgress()
    assert tracker.update(training_line(100), now=100)["eta_seconds"] is None
    assert tracker.update(training_line(200), now=101)["eta_seconds"] is None
    result = tracker.update(training_line(300), now=102)
    assert result["eta_seconds"] == pytest.approx(7)
    assert result["iterations_per_second"] == pytest.approx(100)
    assert result["eta_basis"] == "tqdm_observed_rate"
    assert result["scope"] == "training_loop_only"

    tracker = TrainingProgress()
    for at, step in [(100, 100), (100.2, 200), (100.4, 300)]:
        assert tracker.update(training_line(step), now=at)["eta_seconds"] is None


def test_speed_change_suppresses_previous_eta_until_rates_stabilize():
    tracker, _ = stable_progress()
    changed = tracker.update(training_line(400, rate="40.00it/s"), now=103)
    assert changed["eta_seconds"] is None
    assert changed["iterations_per_second"] is None
    assert changed["eta_basis"] == "collecting"
    for at, step in [(104, 440), (105, 480), (106, 520)]:
        assert tracker.update(training_line(step, rate="40.00it/s"), now=at)["eta_seconds"] is None
    settled = tracker.update(training_line(560, rate="40.00it/s"), now=107)
    assert settled["eta_seconds"] == pytest.approx(11)


def test_missing_tqdm_rate_uses_observed_iteration_intervals():
    tracker = TrainingProgress()
    for at, step in [(10, 10), (12, 50), (14, 90)]:
        assert tracker.update(training_line(step, rate=""), now=at)["eta_seconds"] is None
    result = tracker.update(training_line(130, rate=""), now=16)
    assert result["iterations_per_second"] == pytest.approx(20)
    assert result["eta_seconds"] == pytest.approx(43.5)
    assert result["eta_basis"] == "observed_iteration_intervals"


def test_repeated_step_does_not_add_samples_or_refresh_age():
    tracker = TrainingProgress()
    tracker.update(training_line(100), now=100)
    for at in (101, 102, 103):
        result = tracker.update(training_line(100, rate="999.0it/s"), now=at)
        assert result["eta_seconds"] is None
        assert result["updated_at"] == 100
    assert tracker.update(training_line(200), now=104)["eta_seconds"] is None

    tracker, progress = stable_progress()
    duplicate = tracker.update(training_line(300), now=147.001)
    assert duplicate["updated_at"] == progress["updated_at"] == 102
    fresh = fresh_training_progress(duplicate, "running", now=147.001)
    assert fresh["phase"] == "stale"
    assert fresh["eta_seconds"] is None
    assert fresh["iterations_per_second"] is None


def test_freshness_expires_without_new_messages_and_does_not_mutate_source():
    _, progress = stable_progress()
    assert fresh_training_progress(progress, "running", now=147)["eta_seconds"] == pytest.approx(7)
    stale = fresh_training_progress(progress, "running", now=147.001)
    assert stale["phase"] == "stale"
    assert stale["eta_seconds"] is None
    assert progress["phase"] == "training"
    assert progress["eta_seconds"] == pytest.approx(7)
    assert fresh_training_progress(progress, "running", now=101)["eta_seconds"] is None


def test_first_advancing_update_after_long_pause_requires_new_calibration():
    tracker, _ = stable_progress()
    resumed = tracker.update(training_line(400), now=148)
    assert resumed["eta_seconds"] is None
    assert resumed["eta_basis"] == "collecting"
    assert tracker.update(training_line(500), now=149)["eta_seconds"] is None
    assert tracker.update(training_line(600), now=150)["eta_seconds"] == pytest.approx(4)


@pytest.mark.parametrize("message", [
    "[ITER 300] Saving Gaussians",
    "[ITER 300] Saving Checkpoint",
    "[ITER 300] Evaluating test: L1 0.1 PSNR 20",
    "Reading Training Cameras",
    "Number of points at initialisation : 40000",
])
def test_nontraining_phases_clear_progress_and_rate_history(message):
    tracker, _ = stable_progress()
    assert tracker.update(message, now=103) is None
    assert fresh_training_progress(tracker.latest, "running", now=103) is None
    assert tracker.update(training_line(400), now=104)["eta_seconds"] is None
    assert tracker.update(training_line(500), now=105)["eta_seconds"] is None
    assert tracker.update(training_line(600), now=106)["eta_seconds"] == pytest.approx(4)


@pytest.mark.parametrize("step,total", [(1000, 1000), (1001, 1000), (0, 0)])
def test_completed_or_invalid_totals_clear_eta(step, total):
    tracker, _ = stable_progress()
    assert tracker.update(training_line(step, total), now=103) is None
    assert tracker.latest is None


@pytest.mark.parametrize("message", [
    "Loading cameras: 300/1000 [00:03<00:07, 100.00it/s]",
    "Rendering progress: 300/1000 [00:03<00:07, 100.00it/s]",
    "Saving 300/1000 Gaussians",
    "[ITER 300] Evaluating 300/1000, 100it/s",
    "scene_300/1000.png",
])
def test_unrelated_counters_cannot_start_training_eta(message):
    tracker = TrainingProgress()
    for at in (100, 101, 102):
        assert tracker.update(message, now=at) is None


def test_seconds_per_iteration_is_converted_to_iterations_per_second():
    tracker = TrainingProgress()
    for at, step in [(100, 1), (102, 2), (104, 3)]:
        result = tracker.update(training_line(step, total=10, rate="2.00s/it"), now=at)
    assert result["iterations_per_second"] == pytest.approx(0.5)
    assert result["eta_seconds"] == pytest.approx(14)


def test_alternate_optimizing_splats_label_is_supported():
    tracker = TrainingProgress()
    for at, step in [(100, 100), (101, 200), (102, 300)]:
        result = tracker.update(training_line(step).replace("Training progress:", "Optimizing splats"), now=at)
    assert result["eta_seconds"] == pytest.approx(7)


@pytest.mark.parametrize("step,total", [(200, 1000), (400, 2000)])
def test_counter_restart_or_changed_total_requires_recalibration(step, total):
    tracker, _ = stable_progress()
    assert tracker.update(training_line(step, total), now=103)["eta_seconds"] is None


@pytest.mark.parametrize("status", ["queued", "completed", "failed", "cancelled"])
def test_nonrunning_jobs_have_no_training_eta(status):
    _, progress = stable_progress()
    assert fresh_training_progress(progress, status, now=103) is None

