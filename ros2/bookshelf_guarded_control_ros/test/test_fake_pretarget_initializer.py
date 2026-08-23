"""Contracts for the simulation-only fake xArm initializer."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = (
    ROOT
    / "bookshelf_guarded_control_ros"
    / "fake_pretarget_initializer_node.py"
)


def test_fake_initializer_enables_only_after_confirmation():
    """Require joint confirmation before simulated control is enabled."""
    source = NODE.read_text(encoding="utf-8")
    required = (
        "FollowJointTrajectory",
        '"/xarm7_traj_controller/follow_joint_trajectory"',
        '"/bookshelf_sim/pretarget_ready"',
        "maximum_error <= tolerance",
        'self._publish(True, "fake xArm initialized at reviewed pre-target")',
        '"simulation_only": True',
        '"hardware_commanded": False',
    )
    for token in required:
        assert token in source


def test_fake_initializer_has_no_physical_xarm_api_or_gripper_interface():
    """Keep the initializer isolated from physical xArm interfaces."""
    source = NODE.read_text(encoding="utf-8")
    for token in (
        "xarm_msgs",
        "/xarm/set_mode",
        "/xarm/set_state",
        "xarm_gripper",
    ):
        assert token not in source
