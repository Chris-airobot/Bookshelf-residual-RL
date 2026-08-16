from pathlib import Path

import numpy as np
import pytest

from bookshelf_guarded_control_ros.held_book_pose_check import (
    compare_transforms,
    load_configured_transform,
    mean_transform,
    transform_spread,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import make_transform


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bookshelf_guarded_control_ros" / "held_book_pose_check_node.py"
SCENE_CONFIG = ROOT / "config" / "bookshelf_scene_physical.yaml"


def test_repository_scene_transform_is_loadable():
    transform = load_configured_transform(SCENE_CONFIG)

    assert transform.shape == (4, 4)
    assert transform[:3, 3] == pytest.approx(
        [0.0081243577, -0.0101568565, -0.0477412824]
    )


def test_transform_comparison_reports_translation_and_axis_error():
    configured = make_transform([0.0, 0.0, 0.0])
    live = make_transform(
        [0.003, 0.004, 0.0],
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
    )

    result = compare_transforms(configured, live)

    assert result.translation_error_m == pytest.approx(0.005)
    assert result.rotation_error_deg == pytest.approx(90.0)


def test_stable_transform_estimate_preserves_cluster_center():
    samples = [
        make_transform([0.100 + offset, -0.020, 0.030])
        for offset in (-0.001, 0.0, 0.001)
    ]

    center = mean_transform(samples)
    spread = transform_spread(samples, center)

    assert center[:3, 3] == pytest.approx([0.100, -0.020, 0.030])
    assert spread.translation_error_m == pytest.approx(0.001)
    assert spread.rotation_error_deg == pytest.approx(0.0)


def test_check_node_has_no_motion_or_scene_update_interface():
    source = NODE.read_text(encoding="utf-8")
    forbidden = (
        "ActionClient",
        "ApplyPlanningScene",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "create_client",
        "create_service",
    )
    for token in forbidden:
        assert token not in source

    assert "active_configuration_modified" in source
    assert '"hardware_commanded": False' in source
