import math

import numpy as np

from bookshelf_guarded_control_ros.direct_policy_servo_math import (
    bounded_error_twist,
    eef_target_from_tcp_target,
    matrix_to_axis_angle_vector,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import (
    euler_xyz_to_matrix,
    make_transform,
)


def test_axis_angle_identity_is_zero():
    np.testing.assert_allclose(matrix_to_axis_angle_vector(np.eye(3)), np.zeros(3))


def test_tcp_target_is_converted_back_to_eef_target():
    target_base_tcp = make_transform([0.50, 0.10, 0.20])
    transform_eef_tcp = make_transform([0.08, 0.0, 0.0])

    target_base_eef = eef_target_from_tcp_target(
        target_base_tcp, transform_eef_tcp
    )

    np.testing.assert_allclose(target_base_eef[:3, 3], [0.42, 0.10, 0.20])


def test_error_twist_uses_base_frame_and_caps_linear_and_angular_speed():
    current = make_transform([0.40, 0.0, 0.20])
    target = make_transform([0.50, 0.0, 0.20])
    target[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, 0.20)

    twist = bounded_error_twist(
        current,
        target,
        duration_s=1.0,
        maximum_linear_speed_m_s=0.02,
        maximum_angular_speed_rad_s=0.05,
        translation_tolerance_m=0.0001,
        rotation_tolerance_rad=0.001,
    )

    np.testing.assert_allclose(twist[:3], [0.02, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(twist[3:], [0.0, 0.0, 0.05], atol=1.0e-12)


def test_error_twist_is_zero_inside_pose_tolerance():
    current = make_transform([0.40, 0.0, 0.20])
    target = make_transform([0.4001, 0.0, 0.20])
    target[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, math.radians(0.1))

    twist = bounded_error_twist(
        current,
        target,
        duration_s=0.2,
        maximum_linear_speed_m_s=0.02,
        maximum_angular_speed_rad_s=0.05,
        translation_tolerance_m=0.0005,
        rotation_tolerance_rad=math.radians(0.25),
    )

    np.testing.assert_allclose(twist, np.zeros(6))
