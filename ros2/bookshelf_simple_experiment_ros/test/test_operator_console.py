from types import SimpleNamespace
from unittest.mock import Mock

from bookshelf_simple_experiment_ros.operator_console_node import (
    OperatorWorkflow,
    SERVICES,
    log_service_response,
)


def _plan_and_execute(workflow, key, action, kind, destination):
    assert workflow.command(key) == (action, None)
    workflow.service_result(action, True)
    workflow.preinsert_status("planning", kind)
    workflow.preinsert_status("awaiting_execute_confirmation", kind)
    assert workflow.command("e") == ("execute", None)
    workflow.service_result("execute", True)
    followup = workflow.preinsert_status("done", kind)
    assert workflow.state == destination
    return followup


def _complete_cycle(workflow):
    _plan_and_execute(workflow, "g", "plan_scan", "scan", workflow.SCAN)
    assert workflow.command("s") == ("accept_slot", None)
    workflow.service_result("accept_slot", True)
    _plan_and_execute(
        workflow, "l", "plan_loading", "loading", workflow.LOADING_HOLD
    )
    for key, action, status, destination in (
        ("o", "open", "open", workflow.WAITING_FOR_BOOK),
        ("c", "close", "close", workflow.BOOK_HELD),
    ):
        assert workflow.command(key) == (action, None)
        workflow.service_result(action, True)
        workflow.operator_action_status(status, True)
        assert workflow.state == destination
    _plan_and_execute(
        workflow, "p", "plan_preinsert", "preinsert", workflow.PREINSERT_READY
    )
    assert workflow.command("i") == ("start_policy", None)
    workflow.service_result("start_policy", True)
    workflow.policy_status("episode_complete")
    followup = _plan_and_execute(
        workflow,
        "h",
        "plan_return",
        "return_loading",
        workflow.OPENING_AFTER_RETURN,
    )
    assert followup == "finish_return"
    workflow.pending = followup
    workflow.service_result(followup, True)
    workflow.operator_action_status("ready", True)
    assert workflow.state == workflow.READY_FOR_NEXT_BOOK


def test_two_complete_cycles_accept_second_slot_freeze():
    workflow = OperatorWorkflow()
    _complete_cycle(workflow)
    _plan_and_execute(workflow, "g", "plan_scan", "scan", workflow.SCAN)
    assert workflow.command("s") == ("accept_slot", None)
    workflow.service_result("accept_slot", True)
    assert workflow.state == workflow.SLOT_ACCEPTED
    # Finish the second cycle too, proving all subsequent gates are re-usable.
    _plan_and_execute(
        workflow, "l", "plan_loading", "loading", workflow.LOADING_HOLD
    )


def test_arm_keys_plan_and_only_e_executes():
    cases = (
        (OperatorWorkflow.START, "g", "plan_scan", "scan"),
        (OperatorWorkflow.SLOT_ACCEPTED, "l", "plan_loading", "loading"),
        (OperatorWorkflow.BOOK_HELD, "p", "plan_preinsert", "preinsert"),
        (
            OperatorWorkflow.PUSH_COMPLETE_WAITING_RETURN,
            "h",
            "plan_return",
            "return_loading",
        ),
    )
    for state, key, action, kind in cases:
        workflow = OperatorWorkflow()
        workflow.state = state
        assert workflow.command(key) == (action, None)
        assert workflow.command("e")[0] is None
        workflow.service_result(action, True)
        workflow.preinsert_status("planning", kind)
        workflow.preinsert_status("awaiting_execute_confirmation", kind)
        assert workflow.command("e") == ("execute", None)


def test_e_rejected_without_plan_and_new_target_replaces_old_plan():
    workflow = OperatorWorkflow()
    assert workflow.command("e")[0] is None
    workflow.state = workflow.READY_FOR_NEXT_BOOK
    workflow.pending_plan_kind = "return_loading"
    assert workflow.command("g") == ("plan_scan", None)
    assert workflow.pending_plan_kind == "scan"


def test_failed_direct_verification_returns_to_retry_state_without_ready_plan():
    workflow = OperatorWorkflow()
    assert workflow.command("g") == ("plan_scan", None)
    workflow.service_result("plan_scan", True)
    workflow.preinsert_status("verifying_direct_trajectory", "scan")
    workflow.preinsert_status("failed", "scan")
    assert workflow.state == workflow.START
    assert workflow.pending_plan_kind is None
    assert workflow.command("g") == ("plan_scan", None)


def test_h_rejected_before_native_episode_complete():
    workflow = OperatorWorkflow()
    for state in (workflow.START, workflow.POLICY_RUNNING, workflow.PREINSERT_READY):
        workflow.state = state
        assert workflow.command("h")[0] is None


def test_console_logger_uses_fixed_severity_call_sites():
    logger = SimpleNamespace(info=Mock(), error=Mock())
    log_service_response(logger, True, "ok")
    log_service_response(logger, False, "bad")
    logger.info.assert_called_once_with("ok")
    logger.error.assert_called_once_with("bad")


def test_service_mapping_has_one_generic_reviewed_execute_service():
    assert SERVICES["plan_scan"] == "/bookshelf_simple/plan_scan"
    assert SERVICES["plan_loading"] == "/bookshelf_simple/plan_loading"
    assert SERVICES["plan_preinsert"] == "/bookshelf_simple/plan_preinsert"
    assert SERVICES["plan_return"] == "/bookshelf_simple/plan_return_loading"
    assert SERVICES["execute"] == "/bookshelf_simple/execute_preinsert"
    assert SERVICES["finish_return"] == "/bookshelf_simple/finish_return"
