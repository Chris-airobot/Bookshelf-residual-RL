import hashlib
import json

import pytest
import yaml

from bookshelf_guarded_control_ros.rehearsal_configuration import (
    APPROVAL_TOKEN,
    validate_shadow_rehearsal_assets,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path):
    candidate = tmp_path / "static_slot_cross_view_candidate.json"
    candidate.write_text('{"candidate": true}\n', encoding="utf-8")
    config = tmp_path / "trial_static_slot.yaml"
    document = {
        "static_slot_environment_check": {
            "ros__parameters": {
                "static_slot_width_m": 0.038,
                "static_slot_transform_status": (
                    "captured_rgbd_static_human_approved_abc123"
                ),
            }
        },
        "calibrated_preinsert_target": {
            "ros__parameters": {
                "static_slot_transform_status": (
                    "captured_rgbd_static_human_approved_abc123"
                )
            }
        },
        "policy_observation_adapter": {
            "ros__parameters": {
                "slot_pose_source": "configured_static",
                "allow_configured_static_slot": True,
                "static_slot_transform_status": (
                    "captured_rgbd_static_human_approved_abc123"
                ),
                "book_pose_source": "marker",
                "latch_eef_book_from_marker": False,
                "use_configured_eef_book_transform": True,
                "require_verified_policy_tool_transform": True,
                "policy_tool_transform_status": (
                    "verified_stationary_bag_policy_tool_abc123"
                ),
            }
        },
        "bookshelf_scene_manager": {
            "ros__parameters": {
                "hardware_measurements_confirmed": True,
                "allow_local_insertion": False,
                "held_book_enabled": True,
                "require_held_book_pose_check": True,
            }
        },
    }
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    provenance = {
        "candidate_id": "abc123",
        "human_approval_recorded": True,
        "approval_token": APPROVAL_TOKEN,
        "hardware_commanded": False,
        "execution_authorized": False,
        "trial_config_sha256": _sha256(config),
        "candidate_report": str(candidate),
        "candidate_report_sha256": _sha256(candidate),
    }
    config.with_suffix(".provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    policy = tmp_path / "policy.npz"
    policy.write_bytes(b"portable-policy")
    envelope = tmp_path / "envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "labels": [f"value_{index}" for index in range(12)],
                "lower": [-1.0] * 12,
                "upper": [1.0] * 12,
            }
        ),
        encoding="utf-8",
    )
    return config, policy, envelope


def test_rehearsal_assets_accept_approved_frozen_slot_and_live_marker(tmp_path):
    config, policy, envelope = _write_inputs(tmp_path)
    result = validate_shadow_rehearsal_assets(config, policy, envelope)

    assert result["candidate_id"] == "abc123"
    assert result["slot_pose_source"] == "configured_static"
    assert result["book_pose_source"] == "marker"
    assert result["allow_local_insertion"] is False
    assert result["execution_authorized"] is False
    assert result["hardware_commanded"] is False


def test_rehearsal_assets_reject_fixed_book_fallback(tmp_path):
    config, policy, envelope = _write_inputs(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["policy_observation_adapter"]["ros__parameters"][
        "book_pose_source"
    ] = "eef_fixed"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    provenance_path = config.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["trial_config_sha256"] = _sha256(config)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="continuous marker"):
        validate_shadow_rehearsal_assets(config, policy, envelope)


def test_rehearsal_assets_reject_local_insertion_permission(tmp_path):
    config, policy, envelope = _write_inputs(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["bookshelf_scene_manager"]["ros__parameters"][
        "allow_local_insertion"
    ] = True
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    provenance_path = config.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["trial_config_sha256"] = _sha256(config)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="rehearsal-safe"):
        validate_shadow_rehearsal_assets(config, policy, envelope)


def test_rehearsal_assets_reject_tampered_approved_config(tmp_path):
    config, policy, envelope = _write_inputs(tmp_path)
    config.write_text(config.read_text(encoding="utf-8") + "# changed\n")

    with pytest.raises(ValueError, match="hash differs"):
        validate_shadow_rehearsal_assets(config, policy, envelope)


def test_rehearsal_assets_reject_incomplete_activation_envelope(tmp_path):
    config, policy, envelope = _write_inputs(tmp_path)
    envelope.write_text(
        json.dumps({"labels": ["only_one"], "lower": [0.0], "upper": [1.0]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="12 labels"):
        validate_shadow_rehearsal_assets(config, policy, envelope)
