from pathlib import Path

from bookshelf_simple_experiment_ros.operator_console_node import OperatorWorkflow


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

    workflow.status("awaiting_execute_confirmation", slot_frozen=True)
    assert workflow.state == workflow.PLAN_READY
    assert workflow.command("e") == ("execute", None)
    assert workflow.state == workflow.EXECUTING


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
