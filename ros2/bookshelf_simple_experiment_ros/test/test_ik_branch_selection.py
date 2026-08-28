from types import SimpleNamespace

import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectoryPoint

from bookshelf_simple_experiment_ros.ik_branch_selection import (
    diverse_seeds,
    is_duplicate,
    joint_limit_margin,
    select_candidate,
    trajectory_joint_path_length,
    wrapped_joint_delta,
)


def test_wrap_aware_deduplication_recognizes_equivalent_xarm_joints():
    candidate = np.asarray([0.1, -0.2, 0.3])
    equivalent = candidate + np.asarray([2.0 * np.pi, 0.0, -2.0 * np.pi])

    assert is_duplicate(equivalent, [candidate], tolerance_rad=1.0e-3)
    assert np.max(np.abs(wrapped_joint_delta(equivalent, candidate))) < 1.0e-12


def test_diverse_seeds_are_deterministic_and_keep_current_seed_first():
    current = np.zeros(7)
    lower = np.full(7, -2.0)
    upper = np.full(7, 2.0)

    first = diverse_seeds(current, lower, upper, count=6, random_seed=7)
    second = diverse_seeds(current, lower, upper, count=6, random_seed=7)

    assert len(first) == 6
    np.testing.assert_allclose(first[0], current)
    np.testing.assert_allclose(first, second)


def test_joint_limit_margin_rejects_out_of_bounds_values():
    assert joint_limit_margin(
        [0.0, 0.9], [-1.0, -1.0], [1.0, 1.0]
    ) == pytest.approx(0.1)
    assert joint_limit_margin([0.0, 1.1], [-1.0, -1.0], [1.0, 1.0]) < 0.0


def test_transition_cost_uses_short_revolute_distance():
    trajectory = SimpleNamespace(joint_trajectory=SimpleNamespace(
        joint_names=["joint1", "joint2"],
        points=[
            JointTrajectoryPoint(positions=[3.1, 0.0]),
            JointTrajectoryPoint(positions=[-3.1, 0.3]),
        ],
    ))

    cost = trajectory_joint_path_length(trajectory, ["joint1", "joint2"])

    assert 0.30 < cost < 0.32


def test_known_geometry_prefers_candidate5_family_over_old_singular_branch():
    # Regression values measured for the 2026-08-27 accepted slot geometry.
    old_branch = {
        "candidate_id": 1,
        "joints": [1.1537, 1.6860, 4.9030, 1.4665, 3.5450, 0.6665, 4.4174],
        "max_condition": 29.63,
        "transition_cost": 0.1,
        "plan": object(),
    }
    candidate5_family = {
        "candidate_id": 5,
        "joints": [
            0.142503, 0.767091, -0.019342, 1.377317,
            -3.100515, 0.983269, 3.154346,
        ],
        "max_condition": 21.003,
        "transition_cost": 3.0,
        "plan": object(),
    }

    selected = select_candidate(
        [old_branch, candidate5_family], similar_condition_band=1.0
    )

    assert selected is candidate5_family


def test_transition_cost_breaks_tie_for_similarly_nonsingular_branches():
    candidates = [
        {"candidate_id": 1, "max_condition": 20.0,
         "transition_cost": 4.0, "plan": object()},
        {"candidate_id": 2, "max_condition": 20.7,
         "transition_cost": 1.0, "plan": object()},
        {"candidate_id": 3, "max_condition": 22.0,
         "transition_cost": 0.1, "plan": object()},
    ]

    selected = select_candidate(candidates, similar_condition_band=1.0)

    assert selected["candidate_id"] == 2
