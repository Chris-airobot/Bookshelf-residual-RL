import copy
import json
from pathlib import Path

import pytest
import yaml

from bookshelf_shadow_ros.policy_observation_math import (
    invert_transform,
    make_transform,
)
from bookshelf_shadow_ros.supervised_book_calibration_candidate import (
    generate_supervised_candidate,
)


PACKAGE_ROOT = Path(__file__).parents[1]
SCENE = (
    PACKAGE_ROOT.parent
    / "bookshelf_guarded_control_ros"
    / "config"
    / "bookshelf_scene_physical.yaml"
)
CONTEXT = PACKAGE_ROOT / "config" / "recorded_preinsert_context_2026_08_14.json"
BASE_TARGET = PACKAGE_ROOT / "config" / "calibrated_preinsert_target.yaml"
SOURCE = PACKAGE_ROOT / "bookshelf_shadow_ros" / "supervised_book_calibration_candidate.py"


def _report():
    return {
        "accepted_unique_samples": 30,
        "required_stable_samples": 30,
        "live_candidate_stable": True,
        "live_candidate_transform_tcp_book": {
            "translation_xyz_m": [
                0.0005474978894211335,
                0.002451248163760185,
                0.007558206070405952,
            ],
            "quaternion_xyzw": [
                0.6988075572523508,
                0.005700399708734395,
                0.7150326864340115,
                0.019072511662696166,
            ],
        },
        "sample_spread": {
            "translation_m": 0.00066827372226159,
            "rotation_deg": 0.4937559786973031,
        },
        "tolerances": {
            "maximum_sample_translation_spread_m": 0.003,
            "maximum_sample_rotation_spread_deg": 2.0,
        },
        "tcp_frame": "link_tcp",
        "detected_book_frame": "calibration_detected_book",
        "generated_at": "2026-08-17T00:01:23+10:00",
        "scene_config": {"path": "/tmp/source.yaml", "sha256": "old"},
        "hardware_commanded": False,
        "execution_authorized": False,
        "active_configuration_modified": False,
    }


def _inputs():
    scene = yaml.safe_load(SCENE.read_text(encoding="utf-8"))
    context = json.loads(CONTEXT.read_text(encoding="utf-8"))
    base_target = yaml.safe_load(BASE_TARGET.read_text(encoding="utf-8"))[
        "calibrated_preinsert_target"
    ]["ros__parameters"]
    return _report(), scene, context, base_target


def _generate(report=None):
    values = _inputs()
    return generate_supervised_candidate(
        report or values[0],
        values[1],
        values[2],
        values[3],
        source_report_sha256="a" * 64,
        source_scene_sha256="b" * 64,
        source_context_sha256="c" * 64,
        source_base_target_sha256="d" * 64,
    )


def test_candidate_synchronizes_book_transform_and_closes_scene_holds():
    target, scene, report = _generate()
    target_params = target["calibrated_preinsert_target"]["ros__parameters"]
    adapter_params = target["policy_observation_adapter"]["ros__parameters"]
    scene_params = scene["bookshelf_scene_manager"]["ros__parameters"]

    transform_eef_tcp = make_transform([0.0, 0.0, 0.172])
    transform_tcp_book = make_transform(
        scene_params["held_book_center_tcp_xyz"],
        scene_params["held_book_quaternion_tcp_xyzw"],
    )
    expected_eef_book = transform_eef_tcp @ transform_tcp_book
    actual_eef_book = make_transform(
        target_params["eef_book_translation_xyz"],
        target_params["eef_book_quaternion_xyzw"],
    )
    assert actual_eef_book == pytest.approx(expected_eef_book)
    assert target_params["eef_book_translation_xyz"] == adapter_params[
        "eef_book_translation_xyz"
    ]
    assert target_params["eef_policy_tool_translation_xyz"] == adapter_params[
        "tool_offset_xyz"
    ]
    assert scene_params["hardware_measurements_confirmed"] is False
    assert scene_params["allow_local_insertion"] is False
    assert scene_params["require_held_book_pose_check"] is True
    assert report["safety"]["execution_authorized"] is False
    assert report["safety"]["hardware_commanded"] is False


def test_candidate_preserves_book_to_policy_tool_semantics():
    target, _, report = _generate()
    params = target["calibrated_preinsert_target"]["ros__parameters"]
    transform_eef_book = make_transform(
        params["eef_book_translation_xyz"],
        params["eef_book_quaternion_xyzw"],
    )
    transform_eef_tool = make_transform(
        params["eef_policy_tool_translation_xyz"],
        params["eef_policy_tool_quaternion_xyzw"],
    )
    actual_book_tool = invert_transform(transform_eef_book) @ transform_eef_tool
    expected = report["transforms"]["transform_book_policy_tool_preserved"]
    expected_book_tool = make_transform(
        expected["translation_xyz_m"], expected["quaternion_xyzw"]
    )
    assert actual_book_tool == pytest.approx(expected_book_tool, abs=1.0e-12)
    assert report["policy_tool_parity"]["passed"] is True
    assert report["zero_delta_tcp_identity"]["passed"] is True
    regression = report["offline_preinsert_regression"]
    assert regression["check_passed"] is True
    assert regression["candidate_configuration"][
        "unexpected_observation_clips"
    ] == []


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"accepted_unique_samples": 29}, "29/30"),
        ({"live_candidate_stable": False}, "stable live candidate"),
        ({"hardware_commanded": True}, "hardware_commanded=true"),
        ({"execution_authorized": True}, "execution_authorized=true"),
    ],
)
def test_candidate_rejects_unsafe_or_incomplete_reports(mutation, message):
    report = copy.deepcopy(_report())
    report.update(mutation)
    with pytest.raises(ValueError, match=message):
        _generate(report)


def test_candidate_generator_has_no_motion_interface():
    text = SOURCE.read_text(encoding="utf-8")
    for token in (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "apply_planning_scene",
    ):
        assert token not in text
