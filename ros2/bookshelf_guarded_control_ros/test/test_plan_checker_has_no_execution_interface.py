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
    )
    for gate in required_closed_gates:
        assert gate in config
