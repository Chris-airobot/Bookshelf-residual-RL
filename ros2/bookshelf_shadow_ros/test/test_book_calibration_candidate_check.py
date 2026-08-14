import json
from pathlib import Path

import pytest
import yaml

from bookshelf_shadow_ros.book_calibration_candidate_check import (
    evaluate_book_calibration_candidate,
    load_ros_parameters,
)


PACKAGE_ROOT = Path(__file__).parents[1]
BASE_CONFIG = PACKAGE_ROOT / "config" / "calibrated_preinsert_target.yaml"
CANDIDATE_CONFIG = (
    PACKAGE_ROOT / "config" / "spine_mount_book_calibration_candidate.yaml"
)
RECORDED_CONTEXT = (
    PACKAGE_ROOT / "config" / "recorded_preinsert_context_2026_08_14.json"
)
LAUNCH = (
    PACKAGE_ROOT
    / "launch"
    / "calibrated_preinsert_spine_mount_candidate.launch.py"
)
SOURCE = (
    PACKAGE_ROOT
    / "bookshelf_shadow_ros"
    / "book_calibration_candidate_check.py"
)


def _inputs():
    base = load_ros_parameters(BASE_CONFIG, "calibrated_preinsert_target")
    candidate = load_ros_parameters(
        CANDIDATE_CONFIG, "calibrated_preinsert_target"
    )
    context = json.loads(RECORDED_CONTEXT.read_text(encoding="utf-8"))
    return base, candidate, context


def test_candidate_reproduces_failure_then_removes_orientation_rejection():
    report = evaluate_book_calibration_candidate(*_inputs())

    assert report["check_passed"] is True
    assert report["regression"]["stale_failure_reproduced"] is True
    assert report["regression"]["candidate_passed"] is True
    stale = report["stale_configuration"]
    candidate = report["candidate_configuration"]
    assert stale["book_orientation_error_deg"] == pytest.approx(
        84.72575101288639
    )
    assert candidate["book_orientation_error_deg"] == pytest.approx(
        12.591772771889197
    )
    assert stale["unexpected_observation_clips"] == ["yaw_err"]
    assert candidate["unexpected_observation_clips"] == []
    assert candidate["target_tcp_translation_m"] == pytest.approx(
        0.08673730499281967
    )
    assert report["regression"]["unexpected_clips_removed"] == ["yaw_err"]


def test_candidate_preserves_simulator_book_to_policy_tool_transform():
    report = evaluate_book_calibration_candidate(*_inputs())

    parity = report["candidate_transform_parity"]
    assert parity["passed"] is True
    assert parity["translation_error_m"] < 1.0e-12
    assert parity["rotation_error_deg"] < 1.0e-6


def test_candidate_report_remains_fail_closed():
    report = evaluate_book_calibration_candidate(*_inputs())

    assert report["safety"] == {
        "shadow_only": True,
        "hardware_commanded": False,
        "plan_requested": False,
        "execution_authorized": False,
        "active_configuration_modified": False,
        "selection_authorized": False,
    }


def test_candidate_config_does_not_replace_active_calibration():
    base, candidate, _ = _inputs()
    adapter = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))[
        "policy_observation_adapter"
    ]["ros__parameters"]

    assert base["eef_book_translation_xyz"] == [
        0.008124357683356356,
        -0.010156856549182705,
        0.12425871757162561,
    ]
    assert candidate["eef_book_translation_xyz"] == [
        -0.0015624371872782563,
        0.0027389359956003947,
        0.18481640358639723,
    ]
    assert "candidate" in candidate["eef_book_transform_status"]
    assert candidate["eef_book_translation_xyz"] == adapter[
        "eef_book_translation_xyz"
    ]
    assert candidate["eef_policy_tool_translation_xyz"] == adapter[
        "tool_offset_xyz"
    ]
    assert candidate["policy_tool_transform_status"].startswith(
        "derived_unverified_"
    )


def test_candidate_status_must_remain_unapproved():
    base, candidate, context = _inputs()
    candidate = dict(candidate)
    candidate["eef_book_transform_status"] = "verified"

    with pytest.raises(ValueError, match="candidate-labelled"):
        evaluate_book_calibration_candidate(base, candidate, context)


def test_candidate_checker_and_launch_have_no_execution_interface():
    text = SOURCE.read_text(encoding="utf-8") + LAUNCH.read_text(encoding="utf-8")
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "guarded_policy_tool_executor",
    )
    for token in forbidden:
        assert token not in text
