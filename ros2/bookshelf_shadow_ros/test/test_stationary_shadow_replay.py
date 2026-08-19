import json

import pytest
import yaml

from bookshelf_shadow_ros.policy_observation_math import OBSERVATION_LABELS
from bookshelf_shadow_ros.stationary_capture_bundle import REQUIRED_CAPTURE_TOPICS
from bookshelf_shadow_ros.stationary_shadow_replay import (
    StationaryShadowReplayAccumulator,
    build_shadow_adapter_configuration,
    load_stationary_calibration,
    shadow_bag_play_command,
    shadow_launch_command,
)


def _bundle():
    return {
        "kind": "bookshelf_stationary_capture_calibration_bundle_candidate",
        "candidate_id": "abc123",
        "candidate_valid": True,
        "candidate_selected": False,
        "slot": {
            "translation_xyz": [0.85, 0.08, 0.17],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "width_m": 0.038,
            "confidence": 0.86,
            "transform_status": "captured_view_a_unapproved",
        },
        "safety": {
            "hardware_commanded": False,
            "execution_authorized": False,
        },
    }


def _candidate(book_source="marker"):
    return {
        "policy_observation_adapter": {
            "ros__parameters": {
                "book_pose_source": book_source,
                "latch_eef_book_from_marker": False,
                "use_configured_eef_book_transform": True,
                "eef_book_translation_xyz": [0.0, 0.0, 0.18],
                "eef_book_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "eef_book_transform_status": "measured_candidate",
                "tool_offset_xyz": [0.0, 0.0, 0.14],
                "tool_offset_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "policy_tool_transform_status": "derived_unverified_candidate",
                "require_verified_policy_tool_transform": True,
            }
        }
    }


def _payload(book_source="marker"):
    return {
        "valid": True,
        "book_pose_source": book_source,
        "slot_pose_source": "configured_static",
        "slot_width_m": 0.038,
        "observation_12d": [0.0] * len(OBSERVATION_LABELS),
        "raw_metrics": {label: 0.0 for label in OBSERVATION_LABELS},
    }


def test_shadow_adapter_uses_frozen_view_a_slot_and_live_marker_book():
    result = build_shadow_adapter_configuration(_bundle(), _candidate())
    parameters = result["policy_observation_adapter"]["ros__parameters"]

    assert parameters["book_pose_source"] == "marker"
    assert parameters["latch_eef_book_from_marker"] is False
    assert parameters["slot_pose_source"] == "configured_static"
    assert parameters["configured_static_slot_translation_xyz"] == [
        0.85,
        0.08,
        0.17,
    ]
    assert parameters["configured_static_slot_width_m"] == pytest.approx(0.038)
    assert parameters["require_verified_policy_tool_transform"] is False
    assert parameters["use_configured_eef_book_transform"] is True


def test_shadow_adapter_converts_matching_legacy_candidate_to_live_marker():
    bundle = _bundle()
    bundle["book_calibration"] = {
        "transform_eef_book": {
            "translation_xyz": [0.0, 0.0, 0.18],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    result = build_shadow_adapter_configuration(
        bundle, _candidate("eef_fixed")
    )
    parameters = result["policy_observation_adapter"]["ros__parameters"]
    provenance = result["stationary_shadow_replay_provenance"][
        "ros__parameters"
    ]
    assert parameters["book_pose_source"] == "marker"
    assert provenance["source_candidate_book_pose_source"] == "eef_fixed"
    assert provenance["legacy_candidate_transform_matched_bundle"] is True


def test_shadow_adapter_rejects_fixed_book_transform_mismatch():
    bundle = _bundle()
    bundle["book_calibration"] = {
        "transform_eef_book": {
            "translation_xyz": [0.0, 0.0, 0.19],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    with pytest.raises(ValueError, match="does not match"):
        build_shadow_adapter_configuration(bundle, _candidate("eef_fixed"))


def test_existing_calibration_requires_matching_candidate_hash(tmp_path):
    bundle = _bundle()
    bundle["book_calibration"] = {
        "transform_eef_book": {
            "translation_xyz": [0.0, 0.0, 0.18],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    candidate_path = tmp_path / "stationary_calibration_candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(_candidate("eef_fixed")), encoding="utf-8"
    )
    bundle["output_hashes"] = {
        "unapproved_parameter_candidate_sha256": "wrong"
    }
    (tmp_path / "stationary_calibration_bundle_candidate.json").write_text(
        json.dumps(bundle), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hash does not match"):
        load_stationary_calibration(tmp_path)


def test_shadow_replay_commands_only_use_recorded_inputs_and_shadow_nodes():
    play = shadow_bag_play_command("/tmp/c", rate=1.0)
    assert tuple(play[play.index("--topics") + 1 :]) == REQUIRED_CAPTURE_TOPICS
    text = " ".join(play)
    assert "/bookshelf_policy/observation_12d" not in text
    assert "execute_trajectory" not in text

    launch = shadow_launch_command(
        adapter_config="/tmp/adapter.yaml",
        mount_yaml="/tmp/mount.yaml",
        output_dir="/tmp/output",
        policy_bundle="/tmp/policy.npz",
        activation_envelope="/tmp/envelope.json",
        candidate_id="abc123",
        minimum_valid_samples=30,
        enable_rviz=False,
    )
    launch_text = " ".join(launch)
    assert "stationary_shadow_replay.launch.py" in launch_text
    assert "executor" not in launch_text
    assert "controller" not in launch_text


def test_accumulator_passes_marker_book_with_frozen_slot():
    audit = StationaryShadowReplayAccumulator(minimum_valid_samples=3)
    for index in range(3):
        assert audit.add_adapter_sample(
            _payload(),
            book_position=[0.70 + index * 0.0001, 0.08, 0.17],
            book_quaternion=[0.0, 0.0, 0.0, 1.0],
            slot_position=[0.85, 0.08, 0.17],
            slot_quaternion=[0.0, 0.0, 0.0, 1.0],
        )
    audit.add_policy_debug(
        {
            "valid": False,
            "reason": "policy activation not ready: outside envelope",
            "observation_12d": [0.0] * len(OBSERVATION_LABELS),
            "normalized_observation": [0.0] * len(OBSERVATION_LABELS),
            "policy_activation": {"ready": False},
        }
    )

    summary = audit.summary()
    assert summary["passed"] is True
    assert summary["book_pose_sources"] == {"marker": 3}
    assert summary["slot_pose_sources"] == {"configured_static": 3}
    assert summary["policy_diagnostics"]["valid_inference_messages"] == 0
    assert summary["policy_diagnostics"][
        "activation_is_required_for_replay_pass"
    ] is False


def test_accumulator_rejects_fixed_fallback_and_large_marker_jump():
    audit = StationaryShadowReplayAccumulator(
        minimum_valid_samples=2,
        maximum_book_translation_jump_m=0.01,
    )
    first = _payload(book_source="configured_eef_book")
    second = _payload()
    for payload, x in ((first, 0.70), (second, 0.75)):
        audit.add_adapter_sample(
            payload,
            book_position=[x, 0.08, 0.17],
            book_quaternion=[0.0, 0.0, 0.0, 1.0],
            slot_position=[0.85, 0.08, 0.17],
            slot_quaternion=[0.0, 0.0, 0.0, 1.0],
        )

    summary = audit.summary()
    assert summary["passed"] is False
    assert any("non-marker" in value for value in summary["failure_reasons"])
    assert any("jumped" in value for value in summary["failure_reasons"])


def test_accumulator_reports_clipped_observations_without_forcing_failure():
    audit = StationaryShadowReplayAccumulator(minimum_valid_samples=1)
    payload = _payload()
    payload["observation_12d"][0] = -1.0
    audit.add_adapter_sample(
        payload,
        book_position=[0.70, 0.08, 0.17],
        book_quaternion=[0.0, 0.0, 0.0, 1.0],
        slot_position=[0.85, 0.08, 0.17],
        slot_quaternion=[0.0, 0.0, 0.0, 1.0],
    )
    audit.add_policy_debug(
        {
            "valid": False,
            "reason": "policy activation not ready: outside envelope",
            "observation_12d": payload["observation_12d"],
            "normalized_observation": [0.0] * len(OBSERVATION_LABELS),
        }
    )

    summary = audit.summary()
    assert summary["passed"] is True
    assert summary["observation_clip_fraction"] == 1.0
    assert summary["clipped_observation_counts"][OBSERVATION_LABELS[0]] == 1
