from pathlib import Path


def test_plan_checker_source_has_no_execution_client():
    package = Path(__file__).parents[1] / "bookshelf_guarded_control_ros"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            package / "policy_tool_plan_checker_node.py",
            package / "policy_tool_planner_base.py",
        )
    )
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
    )
    for token in forbidden:
        assert token not in source


def test_plan_checker_reports_frame_chain_on_workspace_rejection():
    source = (
        Path(__file__).parents[1]
        / "bookshelf_guarded_control_ros"
        / "policy_tool_planner_base.py"
    ).read_text(encoding="utf-8")
    required_diagnostics = (
        '"runtime_source_file"',
        '"slot_pose_base"',
        '"current_tcp_base"',
        '"current_policy_tool_base"',
        '"current_policy_tool_slot"',
        '"target_policy_tool_slot"',
        '"target_policy_tool_base"',
        '"target_tcp_base"',
        '"tcp_policy_tool"',
        "self._publish_invalid(error, report=report)",
    )
    for token in required_diagnostics:
        assert token in source


def test_moveit_plan_is_sanity_checked_before_becoming_executable():
    source = (
        Path(__file__).parents[1]
        / "bookshelf_guarded_control_ros"
        / "policy_tool_planner_base.py"
    ).read_text(encoding="utf-8")
    validation = source.index(
        "trajectory_report, trajectory_error = self._trajectory_sanity(response)"
    )
    snapshot = source.index("self.latest_plan = PlanSnapshot(")
    assert validation < snapshot
    assert 'report["trajectory_sanity"] = trajectory_report' in source
    assert "if trajectory_error:" in source


def test_committed_executor_configuration_is_non_executable():
    config = (
        Path(__file__).parents[1]
        / "config"
        / "guarded_policy_tool_executor.yaml"
    ).read_text(encoding="utf-8")
    required_closed_gates = (
        "allow_unverified_policy_tool: false",
        "planning_scene_complete: false",
        "dry_run: true",
        "allow_execution: false",
        "approval_token: DISABLED",
        "require_trajectory_sanity: true",
    )
    for gate in required_closed_gates:
        assert gate in config
