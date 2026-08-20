import math

import numpy as np
import pytest

from bookshelf_guarded_control_ros.policy_servo_simulation_math import (
    initial_eef_from_slot_book,
    integrate_base_frame_twist,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import (
    euler_xyz_to_matrix,
    make_transform,
)


def test_integrates_base_frame_linear_and_angular_velocity():
    initial = make_transform([0.40, 0.10, 0.20])
    twist = np.array([0.02, -0.01, 0.0, 0.0, 0.0, 0.10])

    result = integrate_base_frame_twist(initial, twist, 0.5)

    np.testing.assert_allclose(result[:3, 3], [0.41, 0.095, 0.20])
    np.testing.assert_allclose(
        result[:3, :3],
        euler_xyz_to_matrix(0.0, 0.0, 0.05),
        atol=1.0e-12,
    )


def test_zero_duration_does_not_change_transform():
    initial = make_transform([0.40, 0.10, 0.20], [0.1, 0.2, 0.3, 0.9])
    result = integrate_base_frame_twist(
        initial,
        np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6]),
        0.0,
    )
    np.testing.assert_allclose(result, initial)


def test_initial_eef_reconstructs_requested_book_pose():
    transform_base_slot = make_transform([0.85, 0.08, 0.17])
    transform_slot_book = make_transform([-0.10, 0.0, 0.006])
    transform_eef_book = make_transform(
        [0.006, 0.004, 0.181],
        [0.717, 0.013, 0.696, 0.032],
    )

    transform_base_eef = initial_eef_from_slot_book(
        transform_base_slot,
        transform_slot_book,
        transform_eef_book,
    )

    np.testing.assert_allclose(
        transform_base_eef @ transform_eef_book,
        transform_base_slot @ transform_slot_book,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("duration", [-1.0, math.inf, math.nan])
def test_rejects_invalid_integration_duration(duration):
    with pytest.raises(ValueError, match="duration_s"):
        integrate_base_frame_twist(np.eye(4), np.zeros(6), duration)
