import csv
import math

import pytest

from post_release_pose_evaluation import PoseSample, PostReleasePoseCsv, post_release_row


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _sample(gt_x: float) -> PoseSample:
    return PoseSample(
        tcp_base=(1.0, 2.0, 3.0, *IDENTITY),
        tcp_to_book=(0.1, 0.0, 0.0, *IDENTITY),
        gt_book_base=(gt_x, 2.0, 3.0, *IDENTITY),
        slot_from_base_quaternion=(
            math.sqrt(0.5),
            0.0,
            0.0,
            math.sqrt(0.5),
        ),
    )


def test_pose_error_is_reported_in_slot_frame():
    release = _sample(1.08)
    row = post_release_row(
        episode_index=3,
        env_id=0,
        release_step=42,
        release=release,
        settled_gt_book_base=(1.11, 2.0, 3.0, *IDENTITY),
    )

    assert row["estimate_error_slot_dx_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert row["estimate_error_slot_dy_m"] == pytest.approx(0.02)
    assert row["estimate_error_slot_dz_m"] == pytest.approx(0.0)
    assert row["estimate_orientation_error_rad"] == pytest.approx(0.0)
    assert row["true_settling_displacement_m"] == pytest.approx(0.03)


def test_tracker_captures_release_once_and_settles_before_push(tmp_path):
    output = tmp_path / "post_release.csv"
    tracker = PostReleasePoseCsv(output, num_envs=1)
    release_sample = _sample(1.08)
    settled_sample = _sample(1.11)
    samples = iter((release_sample, settled_sample))

    assert tracker.observe(
        env_id=0, release_step=-1, push_start_step=-1, sample=lambda: release_sample
    ) is None
    assert tracker.observe(
        env_id=0, release_step=42, push_start_step=-1, sample=lambda: next(samples)
    ) is None
    row = tracker.observe(
        env_id=0, release_step=42, push_start_step=136, sample=lambda: next(samples)
    )
    assert row is not None
    assert tracker.observe(
        env_id=0, release_step=42, push_start_step=136, sample=lambda: settled_sample
    ) is None
    tracker.close()

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["release_step"] == "42"
    assert float(rows[0]["gt_book_release_base_x"]) == pytest.approx(1.08)
    assert float(rows[0]["gt_book_settled_base_x"]) == pytest.approx(1.11)


def test_episode_reset_discards_release_without_settle(tmp_path):
    tracker = PostReleasePoseCsv(tmp_path / "post_release.csv", num_envs=1)
    assert tracker.observe(
        env_id=0, release_step=9, push_start_step=-1, sample=lambda: _sample(1.08)
    ) is None
    tracker.episode_done(0)
    tracker.close()
    assert tracker.row_count == 0
