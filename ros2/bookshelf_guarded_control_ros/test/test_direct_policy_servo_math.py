import math

import numpy as np

from bookshelf_guarded_control_ros.direct_policy_servo_math import (
    interpolate_transform,
    matrix_to_axis_angle_vector,
    transform_to_xarm_axis_angle_pose,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import (
    euler_xyz_to_matrix,
    make_transform,
)


def test_xarm_pose_uses_millimetres_and_axis_angle():
    transform = make_transform([0.5, -0.1, 0.25])
    transform[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, math.pi / 2.0)

    pose = transform_to_xarm_axis_angle_pose(transform)

    np.testing.assert_allclose(pose[:3], [500.0, -100.0, 250.0])
    np.testing.assert_allclose(pose[3:], [0.0, 0.0, math.pi / 2.0])


def test_axis_angle_identity_is_zero():
    np.testing.assert_allclose(matrix_to_axis_angle_vector(np.eye(3)), np.zeros(3))


def test_interpolation_reaches_target_without_changing_endpoints():
    start = make_transform([0.4, 0.0, 0.2])
    target = make_transform([0.41, 0.02, 0.23])
    target[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, math.pi / 2.0)

    np.testing.assert_allclose(interpolate_transform(start, target, 0.0), start)
    np.testing.assert_allclose(interpolate_transform(start, target, 1.0), target)
    halfway = interpolate_transform(start, target, 0.5)
    np.testing.assert_allclose(halfway[:3, 3], [0.405, 0.01, 0.215])
    np.testing.assert_allclose(
        halfway[:3, :3], euler_xyz_to_matrix(0.0, 0.0, math.pi / 4.0)
    )

