from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bookshelf_guarded_control_ros" / "bookshelf_scene_manager_node.py"
LAUNCH = ROOT / "launch" / "bookshelf_scene_manager.launch.py"
CONFIG = ROOT / "config" / "bookshelf_scene_physical.yaml"


def test_scene_manager_has_no_motion_or_execution_client():
    source = NODE.read_text(encoding="utf-8")
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "GripperCommand",
        "trajectory_msgs",
        "control_msgs",
    )
    for token in forbidden:
        assert token not in source


def test_launch_exposes_one_physical_configuration_argument():
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument(\n                "scene_config"' in source
    for hardware_argument in (
        "shelf_box_size_xyz",
        "table_box_size_xyz",
        "held_book_size_xyz",
    ):
        assert hardware_argument not in source


def test_repository_physical_scene_config_is_fail_closed():
    source = CONFIG.read_text(encoding="utf-8")
    assert "hardware_measurements_confirmed: false" in source
    assert "allow_local_insertion: false" in source


def test_scene_manager_shutdown_is_idempotent_after_sigint():
    source = NODE.read_text(encoding="utf-8")
    assert "if rclpy.ok():\n            rclpy.shutdown()" in source
