from pathlib import Path

from bookshelf_simple_experiment_ros.operator_console_node import (
    OperatorWorkflow,
    SERVICES,
)


PACKAGE = Path(__file__).resolve().parents[1]


def test_execute_is_locked_until_successful_plan_status():
    workflow = OperatorWorkflow()

    action, message = workflow.command("e")

    assert action is None
    assert "locked" in message


def test_reviewed_operator_sequence_is_state_aware():
    workflow = OperatorWorkflow()

    assert workflow.command("s") == ("accept_slot", None)
    workflow.service_result("accept_slot", True)
    assert workflow.state == workflow.SLOT_ACCEPTED

    assert workflow.command("p") == ("plan", None)
    assert workflow.state == workflow.PLANNING
    workflow.service_result("plan", True)
    assert workflow.command("e")[0] is None

    workflow.status("planning_ik_branches", slot_frozen=True)
    workflow.status("awaiting_execute_confirmation", slot_frozen=True)
    assert workflow.state == workflow.PLAN_READY
    assert workflow.command("e") == ("execute", None)
    assert workflow.state == workflow.PLAN_READY
    workflow.status("executing", slot_frozen=True)
    assert workflow.state == workflow.PLAN_READY
    workflow.service_result("execute", True)
    assert workflow.state == workflow.EXECUTING
    workflow.status("done", slot_frozen=True)
    assert workflow.state == workflow.COMPLETE


def test_pending_service_prevents_double_dispatch():
    workflow = OperatorWorkflow()

    assert workflow.command("s") == ("accept_slot", None)
    action, message = workflow.command("s")

    assert action is None
    assert "pending" in message


def test_plan_never_enables_execute_without_plan_ready_status():
    workflow = OperatorWorkflow()
    workflow.status("slot_frozen", slot_frozen=True)
    workflow.command("p")
    workflow.service_result("plan", True)

    assert workflow.plan_ready is False
    assert workflow.command("e")[0] is None


def test_s_p_without_e_never_enters_execution_states():
    workflow = OperatorWorkflow()
    workflow.command("s")
    workflow.service_result("accept_slot", True)
    workflow.command("p")
    workflow.service_result("plan", True)

    observed = [workflow.state]
    for phase in ("executing", "planning_ik_branches",
                  "awaiting_execute_confirmation", "done"):
        workflow.status(phase, slot_frozen=True)
        observed.append(workflow.state)

    assert observed == [
        workflow.PLANNING,
        workflow.PLANNING,
        workflow.PLANNING,
        workflow.PLAN_READY,
        workflow.PLAN_READY,
    ]
    assert workflow.execution_request_accepted is False


def test_stale_status_cannot_unlock_or_create_execution_state():
    workflow = OperatorWorkflow()

    for phase in ("awaiting_execute_confirmation", "executing", "done"):
        workflow.status(phase, slot_frozen=False)
    assert workflow.state == workflow.SCAN
    assert workflow.command("e")[0] is None

    workflow.command("s")
    workflow.service_result("accept_slot", True)
    workflow.command("p")
    workflow.status("awaiting_execute_confirmation", slot_frozen=True)
    assert workflow.state == workflow.PLANNING
    assert workflow.command("e")[0] is None
    workflow.service_result("plan", True)
    workflow.status("awaiting_execute_confirmation", slot_frozen=True)
    assert workflow.state == workflow.PLANNING


def test_new_plan_invalidates_previous_execute_authorization():
    workflow = OperatorWorkflow()
    workflow.command("s")
    workflow.service_result("accept_slot", True)
    workflow.command("p")
    workflow.service_result("plan", True)
    workflow.status("planning_ik_branches", slot_frozen=True)
    workflow.status("awaiting_execute_confirmation", slot_frozen=True)
    assert workflow.plan_ready

    workflow.command("p")

    assert workflow.state == workflow.PLANNING
    assert workflow.plan_ready is False
    assert workflow.execution_request_accepted is False
    workflow.status("done", slot_frozen=True)
    assert workflow.state == workflow.PLANNING


def test_service_mapping_is_strictly_plan_then_execute():
    assert SERVICES["plan"] == "/bookshelf_simple/plan_preinsert"
    assert SERVICES["execute"] == "/bookshelf_simple/execute_preinsert"
    assert SERVICES["accept_slot"] == "/bookshelf_simple/accept_slot"
    assert "/bookshelf_simple/plan_and_execute_preinsert" not in SERVICES.values()


def test_no_reset_is_invented():
    action, message = OperatorWorkflow().command("r")

    assert action is None
    assert "No safe reset service" in message


def test_console_uses_tty_thread_queue_and_async_service_calls():
    source = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "operator_console_node.py"
    ).read_text(encoding="utf-8")

    assert 'os.open("/dev/tty"' in source
    assert "threading.Thread" in source
    assert "queue.Queue()" in source
    assert ".call_async(" in source
    assert "input(" not in source


def test_top_level_launch_reuses_existing_bringups_and_one_rviz_owner():
    launch = (
        PACKAGE / "launch" / "real_experiment_operator.launch.py"
    ).read_text(encoding="utf-8")

    assert "physical_hardware_bringup.launch.py" in launch
    assert "real_preinsert_workflow.launch.py" in launch
    assert '"robot_ip", default_value="192.168.1.209"' in launch
    assert '"show_rviz": "false"' in launch
    assert '"show_rviz": LaunchConfiguration("show_rviz")' in launch
    assert 'executable="real_experiment_operator"' in launch
