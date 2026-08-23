"""Static ownership checks for the combined physical experiment launch."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "xarm7_policy_physical_experiment.launch.py"


def test_combined_launch_owns_one_hardware_and_policy_stack():
    source = LAUNCH.read_text(encoding="utf-8")
    assert '"physical_hardware_bringup.launch.py"' in source
    assert '"physical_policy_deployment.launch.py"' in source
    assert 'executable="physical_episode_coordinator"' in source
    assert source.count('"physical_hardware_bringup.launch.py"') == 1
    assert source.count('"physical_policy_deployment.launch.py"') == 1


def test_combined_launch_defaults_to_nonmoving_calculate_mode():
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument("operation", default_value="calculate")' in source
    assert 'DeclareLaunchArgument("start_immediately", default_value="false")' in source
    assert '"physical_release_boundary_confirmed", default_value="false"' in source
    assert "control requires physical_release_boundary_confirmed:=true" in source
    assert "validate_episode_operation(" in source


def test_combined_launch_uses_live_control_handoff_and_marker_depth_parameters():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"require_control_enable": "true"',
        '"yield_when_control_disabled": "true"',
        '"control_enable_topic"',
        '"target_book_frame"',
        '"push_target_trailing_depth_m", default_value="-0.012"',
        '"minimum_book_leading_penetration_m", default_value="0.08"',
        '"physical_release_tcp_x_limit_m", default_value="-0.006"',
        '"gripper_action"',
        "physical_episode_geometry_overrides(",
    )
    for token in required:
        assert token in source
