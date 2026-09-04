from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import yaml

from bookshelf_simple_experiment_ros.policy_observation_math import ObservationScales
from bookshelf_simple_experiment_ros.preinsert_node import (
    SimplePreinsertNode,
    _frozen_slot_document,
)
from bookshelf_simple_experiment_ros.simple_policy_control_node import (
    ReviewedPolicyGeometry,
    SimplePolicyControlNode,
)


def test_scan_plan_clears_only_per_trial_preinsert_state():
    harness = SimpleNamespace(
        frozen_slot=object(),
        latest_target=object(),
        latest_target_ns=1,
        planned_trajectory=object(),
        planned_kind="return_loading",
        planned_type="direct_joint",
        executing_kind="return_loading",
        executing_type="direct_joint",
        pending={"kind": "old"},
        branch_search=object(),
        direct_check=object(),
        diagnostics_printed=True,
        latest_slot_candidate="persistent live detector candidate",
        scan_positions="persistent scan pose",
        loading_positions="persistent loading pose",
    )
    SimplePreinsertNode._clear_trial_state_for_scan(harness)
    assert harness.frozen_slot is None
    assert harness.planned_trajectory is None
    assert harness.pending is None
    assert harness.latest_slot_candidate == "persistent live detector candidate"
    assert harness.scan_positions == "persistent scan pose"
    assert harness.loading_positions == "persistent loading pose"


def test_second_policy_start_rearms_completed_rollout():
    harness = SimpleNamespace(
        started=True,
        phase="holding_visualization",
        shadow_full_sequence=False,
        _reset_completed_episode=Mock(),
        _activate_frozen_policy_slot=Mock(),
        _publish_status=Mock(),
        per_grasp_eef_book=np.eye(4),
    )
    response = SimpleNamespace(success=None, message="")
    SimplePolicyControlNode._start_policy_callback(harness, None, response)
    assert response.success is True
    harness._reset_completed_episode.assert_called_once_with()
    assert harness.started is True
    assert harness.phase == "waiting_for_live_state"


def test_policy_restart_rejected_while_episode_is_active():
    harness = SimpleNamespace(started=True, phase="continuous_rollout")
    response = SimpleNamespace(success=None, message="")
    SimplePolicyControlNode._start_policy_callback(harness, None, response)
    assert response.success is False


def test_policy_snapshots_accepted_frozen_slot_instead_of_live_slot(tmp_path):
    accepted = np.eye(4)
    accepted[:3, 3] = [0.8677716788, 0.0694443763, 0.1786139044]
    path = tmp_path / "accepted_slot.yaml"
    path.write_text(
        yaml.safe_dump(_frozen_slot_document("link_base", accepted, 0.041, 0.9)),
        encoding="utf-8",
    )
    stale_live_slot = np.eye(4)
    stale_live_slot[:3, 3] = [0.8554391825, 0.0841262575, 0.1709225333]
    geometry = ReviewedPolicyGeometry(
        transform_base_slot=stale_live_slot.copy(),
        transform_eef_book=np.eye(4),
        transform_eef_tcp=np.eye(4),
        transform_eef_policy_tool=np.eye(4),
        transform_tcp_policy_tool=np.eye(4),
        book_size=(0.156, 0.034, 0.236),
        slot_depth_m=0.2,
        slot_width_m=0.034,
        observation_scales=ObservationScales(),
        gripper_open_joint_position=0.0,
        gripper_closed_joint_position=0.85,
    )
    harness = SimpleNamespace(
        base_frame="link_base",
        geometry=geometry,
        retreat_direction=np.array([-1.0, 0.0, 0.0]),
        policy_slot_source="approved_config",
        get_parameter=lambda name: SimpleNamespace(value={
            "frozen_slot_config": str(path),
        }[name]),
        get_logger=lambda: SimpleNamespace(warning=Mock()),
    )

    SimplePolicyControlNode._activate_frozen_policy_slot(harness)

    np.testing.assert_allclose(harness.geometry.transform_base_slot, accepted)
    assert not np.allclose(harness.geometry.transform_base_slot, stale_live_slot)
    assert harness.geometry.slot_width_m == 0.041
    assert harness.policy_slot_source == f"frozen_accepted:{path.resolve()}"
