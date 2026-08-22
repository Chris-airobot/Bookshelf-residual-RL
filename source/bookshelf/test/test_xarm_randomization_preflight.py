import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = ROOT / "scripts/xarm_randomization_preflight_core.py"
SPEC = importlib.util.spec_from_file_location("xarm_randomization_preflight_core_test", CORE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def test_default_profiles_match_reviewed_preflight_tiers():
    documents = CORE.profile_documents()
    assert [document["name"] for document in documents] == ["current", "medium", "hard"]
    assert documents[0]["grasp_translation_abs_mm"] == [3.0, 3.0, 5.0]
    assert documents[1]["grasp_translation_abs_mm"] == [4.0, 4.0, 8.0]
    assert documents[2]["grasp_translation_abs_mm"] == [5.0, 5.0, 10.0]
    assert documents[0]["slot_clearance_range_mm"] == [2.0, 4.0]
    assert documents[1]["slot_clearance_range_mm"] == [1.0, 5.0]
    assert documents[2]["slot_clearance_range_mm"] == [0.0, 6.0]
    assert math.isclose(documents[2]["grasp_yaw_abs_deg"], 7.0)
    assert math.isclose(documents[2]["arm_joint_noise_abs_deg"], 3.0)


def test_summary_selects_hardest_profile_meeting_threshold():
    rows = []
    for profile, pass_count in (("current", 10), ("medium", 9), ("hard", 8)):
        for index in range(10):
            rows.append(
                {
                    "profile": profile,
                    "missing_book_index": index,
                    "passed": index < pass_count,
                    "failure_reasons": "" if index < pass_count else "translation_drift",
                }
            )
    summary = CORE.summarize_preflight_rows(
        rows,
        profile_order=("current", "medium", "hard"),
        minimum_pass_rate=0.90,
    )
    assert summary["recommended_profile"] == "medium"
    assert summary["profiles"]["current"]["pass_rate"] == 1.0
    assert summary["profiles"]["medium"]["pass_rate"] == 0.9
    assert summary["profiles"]["hard"]["failure_reasons"] == {
        "translation_drift": 2
    }


def test_summary_does_not_skip_a_failed_easier_tier():
    rows = []
    for profile, passed in (("current", False), ("medium", True), ("hard", True)):
        rows.append(
            {
                "profile": profile,
                "missing_book_index": 0,
                "passed": passed,
                "failure_reasons": "" if passed else "translation_drift",
            }
        )
    summary = CORE.summarize_preflight_rows(
        rows,
        profile_order=("current", "medium", "hard"),
        minimum_pass_rate=1.0,
    )
    assert summary["recommended_profile"] is None


def test_preflight_records_requested_and_measured_values():
    script = (ROOT / "scripts/xarm_randomization_preflight.py").read_text(encoding="utf-8")
    environment = (
        ROOT
        / "source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_v4.py"
    ).read_text(encoding="utf-8")
    for expected in (
        '"requested_values.json"',
        '"samples.csv"',
        '"summary.json"',
        '"preflight_failure.json"',
        '"--disable_fabric"',
        '"--phase"',
        '"grasp_only"',
        '"shelf_standoff"',
        '"--gripper_settle_steps"',
        '"initial_placement_error_mm"',
        '"initial_rotation_error_deg"',
        '"maximum_world_downward_motion_mm"',
        '"--ground_contact_height_mm"',
        '"physics_randomization"',
        '"grasp_jitter_z_depth_mm"',
        'f"arm_joint_{joint_index + 1}_noise_deg"',
        'f"applied_arm_joint_{joint_index + 1}_noise_deg"',
        '"initial_book_grasp_qw"',
        '"maximum_translation_drift_mm"',
        '"maximum_rotation_drift_deg"',
        'missing_book_indices=range(10)',
    ):
        assert expected in script
    assert "def debug_grasp_batch_snapshot" in environment
    assert '"scenario_bank_index"' in environment
    assert '"expected_book_position_in_grasp_frame_m"' in environment
    assert '"expected_book_quaternion_in_grasp_frame_wxyz"' in environment
    assert '"book_position_in_grasp_frame_m"' in environment
    assert '"book_position_env_m"' in environment
    assert 'reasons.append("initial_placement")' not in script
    assert 'reasons.append("initial_orientation")' not in script
    assert "if downward_drift_mm >" not in script
    assert '"book_dropped" if reached_ground else "translation_drift"' in script


def test_xarm_slot_reset_preserves_sampled_slot_and_joint_noise():
    environment = (
        ROOT
        / "source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env.py"
    ).read_text(encoding="utf-8")
    assert "self._slot_center_y()[env_ids_t] + offset[1]" in environment
    assert "arm_des = nominal_arm_pos + arm_noise" in environment
    assert "self._scenario_applied_joint_noise_env[env_ids_t]" in environment
    reset_method = environment.index("def _reset_to_slot_relative_tool_pose")
    snap = environment.index(
        "snapped_book_state = self._snap_book_to_measured_grasp(env_ids_t)",
        reset_method,
    )
    hold_target = environment.index(
        "float(self.cfg.gripper_closed_joint_pos)",
        reset_method,
    )
    assert hold_target < snap
    assert environment.index("arm_des = nominal_arm_pos + arm_noise") < environment.index(
        "snapped_book_state = self._snap_book_to_measured_grasp(env_ids_t)",
        reset_method,
    )


def test_grasp_only_obstacles_are_disabled_before_xarm_grasp_settling():
    environment = (
        ROOT
        / "source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env.py"
    ).read_text(encoding="utf-8")
    v5_environment = (
        ROOT
        / "source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_v5.py"
    ).read_text(encoding="utf-8")
    reset = environment.index("def _reset_idx", environment.index("class BookshelfEnv"))
    omit = environment.index("self._omit_bookshelf_obstacles(env_ids_t)", reset)
    slot_reset = environment.index("self._reset_to_slot_relative_tool_pose(env_ids_t)", reset)
    assert omit < slot_reset
    omit_method = environment.index("def _omit_bookshelf_obstacles")
    capture_method = environment.index("def _capture_fixed_tool_to_book_transform")
    assert "delete_prim" not in environment[omit_method:capture_method]
    assert "collision_enabled=not bool(" in v5_environment
    assert "active = torch.zeros_like(active)" in v5_environment
