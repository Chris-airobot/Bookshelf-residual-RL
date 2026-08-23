from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_geometry_diagnostic_captures_synchronized_release_without_changing_control():
    diagnostic = read(ROOT / "scripts/sb3/release_geometry_diagnostic.py")
    for expected in (
        "(mode_before == 0) & (mode_after == 1)",
        '"trailing_edge_depth_from_mouth_m"',
        '"leading_edge_penetration_from_mouth_m"',
        '"physical_frames"',
        '"virtual_policy_tool"',
        '"physical_gripper_to_shelf"',
        '"usd_collision_api_local_bounds_transformed_by_live_body_pose"',
        '"usd_authored_body_bounds_fallback_no_collision_api"',
        '"[RELEASE_GEOMETRY_PROGRESS] "',
        '"panda6_to_xarm7_zero_extra_rotation_environment_clamp"',
        "environment_action_dim = int(env.action_space.shape[0])",
        "_adapt_action(policy_action, environment_action_dim)",
        '"raw_observation_12d"',
        '"book_delta_xyz_mm"',
        '"tcp_delta_xyz_mm"',
    ):
        assert expected in diagnostic


def test_release_geometry_comparison_reports_depth_frames_and_clearance():
    comparison = read(ROOT / "scripts/compare_release_geometry.py")
    for expected in (
        "===== RELEASE DEPTH =====",
        "===== FRAME OFFSETS =====",
        "===== PHYSICAL GRIPPER ENVELOPES =====",
        "===== BODY OPENING MARGINS =====",
    ):
        assert expected in comparison
