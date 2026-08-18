import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.stationary_capture_bundle import (
    RAW_REPLAY_TOPICS,
    build_cross_view_slot_candidate,
    build_stationary_calibration_bundle,
    inspect_capture_run,
    summarize_fixed_transform,
)
from bookshelf_shadow_ros.stationary_capture_pipeline import (
    bag_play_command,
    book_launch_command,
    main as pipeline_main,
    slot_launch_command,
)


def _slot_report(translation, yaw_deg=0.0, width=0.040, confidence=0.9):
    yaw = math.radians(yaw_deg)
    return {
        "schema_version": 1,
        "kind": "bookshelf_static_slot_capture_candidate",
        "valid": True,
        "reason": None,
        "base_frame": "link_base",
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "candidate": {
            "translation_xyz": list(translation),
            "quaternion_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
            "width_m": width,
            "confidence": confidence,
            "transform_status": "captured_rgbd_static_unapproved",
        },
        "statistics": {"inlier_samples": 120},
    }


def _write_capture_run(tmp_path: Path, *, condition="no_book", hardware=False):
    run = tmp_path / condition
    bag = run / "rosbag"
    bag.mkdir(parents=True)
    (bag / "rosbag_0.db3.zstd").write_bytes(b"small-test-bag")
    manifest = {
        "schema_version": 1,
        "completed_cleanly": True,
        "raw_replay_inputs_recorded": True,
        "hardware_commanded_by_logger": hardware,
        "capture_condition": condition,
        "trial_name": condition,
        "repository": {"commit": "abc"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    counts = {topic: 10 for topic in RAW_REPLAY_TOPICS}
    counts.update({"/joint_states": 10, "/robot_description": 1})
    if condition == "book_attached":
        counts["/bookshelf_policy/book_boxes"] = 10
    metadata = {
        "rosbag2_bagfile_information": {
            "duration": {"nanoseconds": 2_000_000_000},
            "message_count": sum(counts.values()),
            "relative_file_paths": ["rosbag_0.db3.zstd"],
            "topics_with_message_count": [
                {
                    "topic_metadata": {"name": topic},
                    "message_count": count,
                }
                for topic, count in counts.items()
            ],
        }
    }
    (bag / "metadata.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )
    return run


def _book_report(transform_eef_book):
    return {
        "schema_version": 1,
        "calibration_valid": True,
        "hardware_commanded": False,
        "read_only": True,
        "minimum_inlier_samples": 30,
        "minimum_inlier_fraction": 0.7,
        "frame_convention": {
            "transform_output": "T_eef_book (book pose expressed in link_eef)"
        },
        "result": {
            "transform_eef_book": transform_eef_book.tolist(),
            "translation_xyz_m": transform_eef_book[:3, 3].tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "input_samples": 250,
            "inlier_samples": 240,
            "inlier_fraction": 0.96,
        },
    }


def _eef_tcp_context(transform_eef_tcp):
    return {
        "schema_version": 1,
        "valid": True,
        "hardware_commanded": False,
        "parent_frame": "link_eef",
        "child_frame": "link_tcp",
        "transform_eef_tcp": {
            "translation_xyz_m": transform_eef_tcp[:3, 3].tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_capture_run_validation_accepts_required_raw_inputs(tmp_path):
    run = _write_capture_run(tmp_path)

    result = inspect_capture_run(
        run,
        role="view_a",
        expected_condition="no_book",
    )

    assert result["validated"] is True
    assert result["duration_s"] == 2.0
    assert len(result["bag_files"][0]["sha256"]) == 64


def test_capture_run_validation_rejects_wrong_condition_and_hardware_claim(tmp_path):
    run = _write_capture_run(tmp_path, hardware=True)

    with pytest.raises(ValueError, match="hardware_commanded=false"):
        inspect_capture_run(
            run,
            role="book_attached",
            expected_condition="book_attached",
        )


def test_cross_view_slot_candidate_uses_view_a_after_cross_view_validation():
    result = build_cross_view_slot_candidate(
        _slot_report([0.890, 0.093, 0.154], yaw_deg=0.5, width=0.039),
        _slot_report([0.894, 0.094, 0.155], yaw_deg=2.5, width=0.041),
    )

    assert result["valid"] is True
    assert result["execution_authorized"] is False
    assert result["active_configuration_modified"] is False
    np.testing.assert_allclose(
        result["candidate"]["translation_xyz"], [0.890, 0.093, 0.154]
    )
    assert result["candidate"]["width_m"] == pytest.approx(0.039)
    assert result["cross_view_disagreement"]["rotation_deg"] == pytest.approx(2.0)
    assert result["cross_view_validation"]["pose_source"] == "view_a"
    assert result["warnings"] == []


def test_cross_view_slot_candidate_warns_on_angled_view_b_orientation():
    result = build_cross_view_slot_candidate(
        _slot_report([0.890, 0.093, 0.154], yaw_deg=0.0),
        _slot_report([0.890, 0.093, 0.154], yaw_deg=8.0),
    )

    assert result["valid"] is True
    assert result["reason"] is None
    assert result["cross_view_validation"][
        "rotation_within_diagnostic_tolerance"
    ] is False
    assert len(result["warnings"]) == 1
    assert result["candidate"]["quaternion_xyzw"] == pytest.approx(
        _slot_report([0.890, 0.093, 0.154], yaw_deg=0.0)["candidate"][
            "quaternion_xyzw"
        ]
    )


def test_cross_view_slot_candidate_fails_closed_on_gross_orientation_mismatch():
    result = build_cross_view_slot_candidate(
        _slot_report([0.890, 0.093, 0.154], yaw_deg=0.0),
        _slot_report([0.890, 0.093, 0.154], yaw_deg=20.0),
    )

    assert result["valid"] is False
    assert "gross sanity tolerance" in result["reason"]
    assert "candidate" not in result


def test_cross_view_slot_candidate_rejects_invalid_rotation_tolerance_order():
    with pytest.raises(ValueError, match="sanity tolerance"):
        build_cross_view_slot_candidate(
            _slot_report([0.890, 0.093, 0.154]),
            _slot_report([0.890, 0.093, 0.154]),
            maximum_rotation_disagreement_deg=10.0,
            maximum_rotation_sanity_disagreement_deg=5.0,
        )


def test_fixed_transform_summary_rejects_motion():
    transforms = [make_transform([0.0, 0.0, 0.172]) for _ in range(9)]
    transforms.append(make_transform([0.0, 0.0, 0.180]))

    with pytest.raises(ValueError, match="translation spread"):
        summarize_fixed_transform(transforms)


def test_bundle_derives_tcp_book_and_keeps_every_review_hold_closed():
    slot_report = build_cross_view_slot_candidate(
        _slot_report([0.890, 0.093, 0.154]),
        _slot_report([0.892, 0.093, 0.154]),
    )
    transform_eef_tcp = make_transform([0.0, 0.0, 0.172])
    expected_tcp_book = make_transform([0.010, -0.020, -0.050])
    transform_eef_book = transform_eef_tcp @ expected_tcp_book

    bundle, candidate = build_stationary_calibration_bundle(
        slot_report,
        _book_report(transform_eef_book),
        _eef_tcp_context(transform_eef_tcp),
        capture_provenance={
            "view_a": {"validated": True},
            "view_b": {"validated": True},
            "book_attached": {"validated": True},
        },
        source_hashes={"a": "1", "b": "2", "c": "3"},
    )

    np.testing.assert_allclose(
        bundle["book_calibration"]["transform_tcp_book"]["translation_xyz_m"],
        [0.010, -0.020, -0.050],
        atol=1.0e-12,
    )
    assert bundle["candidate_selected"] is False
    assert bundle["policy_tool"]["verified"] is False
    assert bundle["safety"]["execution_authorized"] is False
    assert bundle["safety"]["hardware_commanded"] is False
    scene = candidate["bookshelf_scene_manager"]["ros__parameters"]
    assert scene["hardware_measurements_confirmed"] is False
    assert scene["allow_local_insertion"] is False
    adapter = candidate["policy_observation_adapter"]["ros__parameters"]
    assert adapter["require_verified_policy_tool_transform"] is True


def test_pipeline_commands_replay_raw_perception_only():
    play = bag_play_command("/tmp/bag", rate=1.0)
    command_text = " ".join(play)

    assert tuple(play[play.index("--topics") + 1 :]) == RAW_REPLAY_TOPICS
    assert "/slot_detector/slot_pose" not in command_text
    assert "/bookshelf_policy/book_boxes" not in command_text
    assert "execute_trajectory" not in command_text
    assert "follow_joint_trajectory" not in command_text
    slot = slot_launch_command(
        "/tmp/a",
        "/repo",
        120,
        0.55,
        roi_x_min=0.20,
        roi_x_max=0.52,
        minimum_slot_width_m=0.032,
        maximum_slot_width_m=0.044,
    )
    assert "use_sim_time:=true" in slot
    assert "minimum_confidence:=0.55" in slot
    assert "detector_roi_x_min:=0.2" in slot
    assert "detector_roi_x_max:=0.52" in slot
    assert "detector_minimum_slot_width_m:=0.032" in slot
    assert "detector_maximum_slot_width_m:=0.044" in slot
    assert "capture_use_latest_tf:=true" in slot
    book = book_launch_command("/tmp/c", 250)
    assert "enable_rviz:=false" in book
    assert "capture_eef_tcp_context:=true" in book


def test_bag_launches_forward_sim_time_and_capture_eef_tcp_context():
    package_root = Path(__file__).parents[1]
    slot_launch = (package_root / "launch/static_slot_capture.launch.py").read_text(
        encoding="utf-8"
    )
    book_launch = (
        package_root / "launch/marker_book_bag_calibration.launch.py"
    ).read_text(encoding="utf-8")
    setup = (package_root / "setup.py").read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument(\n            "use_sim_time"' in slot_launch
    assert 'LaunchConfiguration("use_sim_time")' in slot_launch
    assert 'DeclareLaunchArgument(\n            "show_debug_image"' in slot_launch
    assert 'arguments=["/slot_detector/debug_image"]' in slot_launch
    assert 'DeclareLaunchArgument(\n            "capture_use_latest_tf"' in slot_launch
    assert 'LaunchConfiguration("capture_use_latest_tf")' in slot_launch
    assert 'DeclareLaunchArgument(\n            "use_sim_time"' in book_launch
    assert "eef_tcp_context_capture" in book_launch
    assert "stationary_capture_pipeline" in setup
    assert "eef_tcp_context_capture" in setup


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"roi_x_min": 0.8, "roi_x_max": 0.2}, "ROI"),
        (
            {
                "minimum_slot_width_m": 0.050,
                "maximum_slot_width_m": 0.040,
            },
            "width limits",
        ),
    ],
)
def test_slot_launch_command_rejects_invalid_search_constraints(overrides, message):
    with pytest.raises(ValueError, match=message):
        slot_launch_command("/tmp/a", "/repo", 120, **overrides)


def test_pipeline_refuses_to_modify_a_nonempty_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    result = pipeline_main(
        [
            "--view-a-run",
            "/missing/a",
            "--view-b-run",
            "/missing/b",
            "--book-run",
            "/missing/c",
            "--output-dir",
            str(output),
            "--repository",
            "/missing/repository",
        ]
    )

    assert result == 1
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (output / "pipeline_failure.json").exists()
