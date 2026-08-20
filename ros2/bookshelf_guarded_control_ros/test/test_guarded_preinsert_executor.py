from pathlib import Path


ROOT = Path(__file__).parents[1]
NODE = ROOT / "bookshelf_guarded_control_ros" / "guarded_preinsert_executor_node.py"
CONFIG = ROOT / "config" / "guarded_preinsert_executor.yaml"
LAUNCH = ROOT / "launch" / "guarded_preinsert_execute_once.launch.py"
SETUP = ROOT / "setup.py"
PLANNER = ROOT / "bookshelf_guarded_control_ros" / "calibrated_preinsert_plan_only_node.py"
RUNBOOK = ROOT.parents[1] / "REAL_ROBOT_EXPERIMENT_COMMANDS_2026-08-16.md"
LOGGER = ROOT.parent / "bookshelf_shadow_ros" / "launch" / "experiment_logging.launch.py"


def test_committed_global_executor_is_fail_closed():
    source = CONFIG.read_text(encoding="utf-8")
    for value in (
        "dry_run: true",
        "allow_execution: false",
        "planning_scene_complete: false",
        "human_trajectory_review_complete: false",
        "target_transform_physically_validated: false",
        "approval_token: DISABLED",
        "expected_scene_config_sha256: DISABLED",
        "expected_target_transform_status: DISABLED",
        "maximum_joint_state_age_s: 0.50",
        "maximum_scene_status_age_s: 1.0",
    ):
        assert value in source


def test_executor_is_one_shot_token_gated_and_has_no_gripper():
    source = NODE.read_text(encoding="utf-8")
    required = (
        "OneShotExecutionGuard",
        "self._startup_gates_open()",
        "self.execution_guard.try_consume()",
        "maximum_named_joint_difference",
        "trajectory_sha256",
        "TRAJECTORY_FINGERPRINT_KIND",
        "expected_scene_config_sha256",
        "expected_target_transform_status",
        "latest_joint_state_ns",
        "latest_scene_status_ns",
        "ExecuteTrajectory",
        '"gripper_command_interface": False',
    )
    for token in required:
        assert token in source
    for forbidden in ("GripperCommand", "GetMotionPlan", "GetPositionIK"):
        assert forbidden not in source


def test_execution_client_is_created_only_after_startup_gates():
    source = NODE.read_text(encoding="utf-8")
    gate = source.index("if self._startup_gates_open():")
    client = source.index("ActionClient(", gate)
    subscriptions = source.index("self.create_subscription(", client)
    assert gate < client < subscriptions


def test_planner_binds_trajectory_to_report_and_launch_installs_executor():
    planner = PLANNER.read_text(encoding="utf-8")
    launch = LAUNCH.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")
    assert '"trajectory_sha256"' in planner
    assert '"trajectory_fingerprint_kind"' in planner
    assert "canonical_ros_message_sha256" in planner
    assert "canonical_ros_message_sha256" in NODE.read_text(encoding="utf-8")
    assert "serialize_message" not in planner
    assert "serialize_message" not in NODE.read_text(encoding="utf-8")
    assert 'executable="guarded_preinsert_executor"' in launch
    assert "committed" in launch.lower()
    assert '"guarded_preinsert_executor = "' in setup


def test_supervised_runbook_keeps_execution_explicit_and_logged():
    source = RUNBOOK.read_text(encoding="utf-8")
    for token in (
        "physical_experiment_observation_bringup.launch.py",
        "calibrated_preinsert_spine_mount_candidate_plan_only.launch.py",
        "guarded_preinsert_execute_once.launch.py",
        "/bookshelf_preinsert/approve_once",
        "/bookshelf_preinsert/execution_report",
        "supervised multi-step insertion",
    ):
        assert token in source


def test_automatic_logger_records_approval_and_execution_evidence():
    source = LOGGER.read_text(encoding="utf-8")
    for topic in (
        "/bookshelf_preinsert/approve_once",
        "/bookshelf_preinsert/execution_report",
        "/bookshelf_guarded/approve_once",
        "/bookshelf_guarded/execution_report",
        "/bookshelf_scene/held_book_pose_check_passed",
        "/bookshelf_scene/held_book_pose_check_status",
        "/bookshelf_scene/live_held_book_pose_tcp",
        "/bookshelf_scene/configured_held_book_pose_tcp",
        "/bookshelf_policy/book_boxes",
    ):
        assert topic in source
