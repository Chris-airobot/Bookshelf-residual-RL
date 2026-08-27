import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from visualization_msgs.msg import Marker

from bookshelf_simple_experiment_ros.fake_policy_start_node import (
    FakePolicyStartNode,
    REVIEWED_PREINSERT_JOINT_POSITIONS,
)
from bookshelf_simple_experiment_ros.policy_tool_math import make_transform
from bookshelf_simple_experiment_ros.simple_policy_control_node import (
    SimplePolicyControlNode,
    build_policy_visualization_markers,
    visualization_hold_deadline_ns,
)


def test_policy_snapshot_contains_all_required_visual_elements():
    identity = np.eye(4)
    markers = build_policy_visualization_markers(
        "link_base",
        identity,
        make_transform([0.1, 0.0, 0.0]),
        make_transform([0.2, 0.0, 0.0]),
        make_transform([0.3, 0.0, 0.0]),
        make_transform([0.4, 0.0, 0.0]),
        make_transform([0.5, 0.0, 0.0]),
        slot_depth_m=0.20,
        slot_width_m=0.038,
        book_size=(0.156, 0.034, 0.236),
    ).markers
    assert [marker.text for marker in markers] == [
        "saved_slot",
        "current_book",
        "current_tcp",
        "current_policy_tool",
        "target_tcp",
        "target_policy_tool",
    ]
    assert markers[0].type == Marker.CUBE
    assert markers[1].type == Marker.CUBE
    assert all(marker.type == Marker.SPHERE for marker in markers[2:])
    assert len({marker.id for marker in markers}) == len(markers)


def test_shadow_visualization_hold_is_bounded_or_explicitly_indefinite():
    assert visualization_hold_deadline_ns(1_000, 60.0) == 60_000_001_000
    assert visualization_hold_deadline_ns(1_000, 0.0) is None
    with pytest.raises(ValueError):
        visualization_hold_deadline_ns(1_000, -1.0)


def test_fake_policy_start_keeps_reviewed_preinsert_arm_pose_exactly():
    assert REVIEWED_PREINSERT_JOINT_POSITIONS == [
        1.2342693425054612,
        1.5322427671441177,
        4.904658882462919,
        1.302429752118059,
        3.302595179623167,
        0.6839448116011184,
        4.4791192150828865,
    ]


def test_fake_policy_initializer_finishes_via_process_loop_flag():
    harness = SimpleNamespace(
        initialization_complete=False,
        gripper_pending=True,
        _result_succeeded=lambda _future, _label: True,
        get_logger=lambda: SimpleNamespace(info=Mock()),
    )
    FakePolicyStartNode._gripper_result(harness, Mock())
    assert harness.initialization_complete is True


def test_one_step_waits_when_joint_state_precedes_tf_buffer():
    harness = SimpleNamespace(
        eef_frame="link_eef",
        tcp_frame="link_tcp",
        _live_input_error=lambda: None,
        _lookup=Mock(side_effect=RuntimeError("TF buffer not primed")),
        _fail=Mock(),
    )
    SimplePolicyControlNode._try_calculate(harness)
    harness._fail.assert_not_called()
