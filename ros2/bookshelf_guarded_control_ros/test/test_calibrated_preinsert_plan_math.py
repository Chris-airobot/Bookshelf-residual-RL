import math

import numpy as np

from bookshelf_guarded_control_ros.calibrated_preinsert_plan_math import (
    PreinsertTargetLimits,
    preinsert_target_error,
    preinsert_target_metrics,
    target_identifier,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import (
    euler_xyz_to_matrix,
    make_transform,
)


def test_preserved_orientation_global_translation_is_accepted():
    current = make_transform([0.50, 0.05, 0.20])
    target = make_transform([0.82, 0.08, 0.24])

    metrics = preinsert_target_metrics(current, target)

    assert metrics["translation_m"] > 0.30
    assert metrics["rotation_rad"] == 0.0
    assert preinsert_target_error(current, target) is None


def test_large_rotation_and_workspace_violation_are_rejected():
    current = make_transform([0.50, 0.0, 0.20])
    rotated = make_transform([0.60, 0.0, 0.20])
    rotated[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, math.radians(10.0))
    assert "rotation exceeds" in preinsert_target_error(current, rotated)

    outside = make_transform([1.10, 0.0, 0.20])
    assert "outside workspace" in preinsert_target_error(
        current,
        outside,
        limits=PreinsertTargetLimits(maximum_translation_m=2.0),
    )


def test_target_identifier_is_stable_and_pose_sensitive():
    target = make_transform([0.82, 0.08, 0.24])
    same = np.array(target, copy=True)
    changed = np.array(target, copy=True)
    changed[0, 3] += 0.001

    assert target_identifier(target) == target_identifier(same)
    assert target_identifier(target) != target_identifier(changed)
