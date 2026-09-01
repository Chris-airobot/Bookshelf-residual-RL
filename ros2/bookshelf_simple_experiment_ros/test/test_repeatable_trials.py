from types import SimpleNamespace
from unittest.mock import Mock

from bookshelf_simple_experiment_ros.preinsert_node import SimplePreinsertNode
from bookshelf_simple_experiment_ros.simple_policy_control_node import SimplePolicyControlNode


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
        _publish_status=Mock(),
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
