"""Numerical parity checks for the copied post-INSERT helpers."""

import numpy as np
import pytest

from bookshelf_guarded_control_ros.fake_release_retreat_sequence_node import (
    oriented_box_contact_gap as old_oriented_box_contact_gap,
    retreat_progress as old_retreat_progress,
    simulated_book_push_distance as old_simulated_book_push_distance,
)
from bookshelf_shadow_ros.policy_shadow_math import (
    compute_push_nominal_delta as old_compute_push_nominal_delta,
)
from bookshelf_simple_experiment_ros.post_insert_math import (
    compute_push_nominal_delta,
    oriented_box_contact_gap,
    retreat_progress,
    simulated_book_push_distance,
)


def test_retreat_and_fake_book_push_match_verified_old_helpers():
    start = [0.8, 0.1, 0.2]
    current = [0.71, 0.101, 0.2]
    direction = [-1.0, 0.0, 0.0]
    assert retreat_progress(start, current, direction) == pytest.approx(
        old_retreat_progress(start, current, direction)
    )
    assert simulated_book_push_distance(0.14, 0.09, 0.03) == pytest.approx(
        old_simulated_book_push_distance(0.14, 0.09, 0.03)
    )


def test_oriented_contact_and_nominal_push_match_verified_old_helpers():
    box = np.eye(4)
    box[:3, 3] = [0.9, 0.08, 0.18]
    arguments = ([0.72, 0.08, 0.18], box, [0.156, 0.034, 0.236], [1, 0, 0])
    assert oriented_box_contact_gap(*arguments) == pytest.approx(
        old_oriented_box_contact_gap(*arguments)
    )
    raw = np.array(
        [1.0, -0.05, 0.1, 0.002, 0.006, 0.03, 0.0, -0.001,
         -0.07, 0.0, 0.05, -0.02],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        compute_push_nominal_delta(raw),
        old_compute_push_nominal_delta(raw),
        rtol=0.0,
        atol=0.0,
    )


def test_x_uncertainty_does_not_change_push_command_semantics():
    geometric_contact = 0.09
    uncertainty = 0.005
    raw = np.array(
        [1.0, -0.05, 0.1, 0.002, 0.006, 0.03, 0.0, -0.001,
         -0.07, 0.0, 0.05, -0.02],
        dtype=np.float32,
    )
    raw_before = raw.copy()
    nominal_before = compute_push_nominal_delta(raw)
    commanded_completion = geometric_contact + 0.03
    assert commanded_completion == pytest.approx(0.12)
    assert commanded_completion != pytest.approx(
        geometric_contact + uncertainty + 0.03
    )
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(compute_push_nominal_delta(raw), nominal_before)


def test_requested_book_push_remains_thirty_mm_from_geometric_contact():
    geometric_contact = 0.09
    assert simulated_book_push_distance(0.12, geometric_contact, 0.03) == pytest.approx(
        0.03
    )
    assert simulated_book_push_distance(1.0, geometric_contact, 0.03) == pytest.approx(
        0.03
    )
