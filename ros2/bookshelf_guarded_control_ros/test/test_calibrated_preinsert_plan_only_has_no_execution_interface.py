from pathlib import Path


ROOT = Path(__file__).parents[1]
NODE = (
    ROOT
    / "bookshelf_guarded_control_ros"
    / "calibrated_preinsert_plan_only_node.py"
)
LAUNCH = ROOT / "launch" / "calibrated_preinsert_plan_only.launch.py"
CANDIDATE_LAUNCH = (
    ROOT
    / "launch"
    / "calibrated_preinsert_spine_mount_candidate_plan_only.launch.py"
)
CONFIG = ROOT / "config" / "calibrated_preinsert_plan_only.yaml"
SETUP = ROOT / "setup.py"


def test_preinsert_planner_has_planning_but_no_execution_interface():
    source = NODE.read_text(encoding="utf-8")
    assert "GetMotionPlan" in source
    assert "GetPositionIK" in source
    assert "self.create_client(" in source
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "GripperCommand",
        "approval_topic",
    )
    for token in forbidden:
        assert token not in source


def test_preinsert_planner_requires_global_scene_and_never_reports_execution_ready():
    source = NODE.read_text(encoding="utf-8")
    assert "global_scene_status_error(self.latest_scene_status)" in source
    assert '"execution_ready": False' in source
    assert '"execution_authorized": False' in source
    assert '"hardware_commanded": False' in source


def test_trajectory_sanity_precedes_valid_plan_publication():
    source = NODE.read_text(encoding="utf-8")
    sanity = source.index("trajectory_report, error = self._trajectory_sanity(response)")
    valid = source.index('"valid": True', sanity)
    publish = source.index("self.plan_valid_publisher.publish(Bool(data=True))", valid)
    assert sanity < valid < publish


def test_invalid_joint_branch_request_is_reported_fail_closed():
    source = NODE.read_text(encoding="utf-8")
    assert "named_joint_target_branch_report(" in source
    assert "near-current IK branch validation is disabled" in source
    assert '"goal_joint_branch_constraint"' in source
    assert '"seeded_collision_aware_ik_then_joint_goal_plan"' in source


def test_launch_combines_target_calculation_and_plan_only_node():
    source = LAUNCH.read_text(encoding="utf-8")
    assert '"target_config"' in source
    assert '"scene_config"' in source
    assert '"bookshelf_scene_manager.launch.py"' in source
    assert '"preserve_current_tcp"' in source
    assert 'executable="calibrated_preinsert_plan_only"' in source
    assert "guarded_policy_tool_executor" not in source


def test_candidate_launch_layers_candidate_target_scene_and_plan_only_node():
    source = CANDIDATE_LAUNCH.read_text(encoding="utf-8")
    required = (
        '"target_config"',
        '"candidate_config"',
        '"scene_config"',
        '"bookshelf_scene_manager.launch.py"',
        'executable="calibrated_preinsert_target"',
        '"target_orientation_mode": "preserve_current_tcp"',
        'executable="calibrated_preinsert_plan_only"',
        "UNAPPROVED SPINE-MOUNT CANDIDATE PLAN-ONLY",
        "Execution remains unauthorized",
    )
    for token in required:
        assert token in source


def test_candidate_plan_only_launch_has_no_execution_interface():
    source = CANDIDATE_LAUNCH.read_text(encoding="utf-8")
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "guarded_policy_tool_executor",
        "guarded_policy_tool_single_step",
    )
    for token in forbidden:
        assert token not in source


def test_default_configuration_is_bounded_and_global():
    source = CONFIG.read_text(encoding="utf-8")
    required = (
        "velocity_scaling: 0.05",
        "acceleration_scaling: 0.05",
        "maximum_target_translation_m: 0.75",
        "maximum_target_rotation_deg: 5.0",
        "require_trajectory_sanity: true",
        "require_near_current_goal_joints: true",
        "maximum_goal_joint_delta_rad: 1.5",
        "ik_service: /compute_ik",
        "ik_avoid_collisions: true",
        "joint_goal_tolerance_rad: 0.001",
        "guarded_policy_tool_executor",
    )
    for value in required:
        assert value in source


def test_console_entry_point_is_installed():
    source = SETUP.read_text(encoding="utf-8")
    assert '"calibrated_preinsert_plan_only = "' in source
