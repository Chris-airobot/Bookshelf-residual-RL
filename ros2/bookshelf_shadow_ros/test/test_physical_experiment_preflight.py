import json
from pathlib import Path

import pytest
import yaml

from bookshelf_shadow_ros.physical_experiment_preflight import (
    validate_fail_closed_executor,
    validate_policy_assets,
    validate_provenance,
    validate_trial_configuration,
)
from bookshelf_shadow_ros.static_slot_capture import (
    APPROVAL_TOKEN,
    promote_capture_candidate,
)


PACKAGE = Path(__file__).parents[1]
REPOSITORY = PACKAGE.parents[1]


def _candidate(path: Path):
    document = {
        "schema_version": 1,
        "kind": "bookshelf_static_slot_capture_candidate",
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "valid": True,
        "candidate": {
            "translation_xyz": [0.891, 0.093, 0.155],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "width_m": 0.0397,
            "confidence": 0.94,
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _trial(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "trial_static_slot.yaml"
    _candidate(candidate)
    promote_capture_candidate(
        candidate,
        PACKAGE / "config",
        output,
        approval_token=APPROVAL_TOKEN,
    )
    return output


def test_trial_and_provenance_preflight_accept_promoted_configuration(tmp_path):
    output = _trial(tmp_path)

    slot = validate_trial_configuration(output)
    provenance = validate_provenance(output)

    assert slot["width_m"] == pytest.approx(0.0397)
    assert slot["transform_status"].startswith(
        "captured_rgbd_static_human_approved_"
    )
    assert provenance["candidate_report"] == str(tmp_path / "candidate.json")


def test_trial_preflight_rejects_ros_incompatible_aliases(tmp_path):
    output = _trial(tmp_path)
    output.write_text(
        output.read_text(encoding="utf-8") + "alias: &id001 [1, 2]\ncopy: *id001\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="aliases"):
        validate_trial_configuration(output)


def test_policy_assets_and_executor_defaults_agree(tmp_path):
    bundle = tmp_path / "actor.npz"
    bundle.write_bytes(b"actor")
    labels = [f"channel_{index}" for index in range(12)]
    envelope = tmp_path / "envelope.json"
    envelope.write_text(
        json.dumps({"labels": labels, "lower": [-1.0] * 12, "upper": [1.0] * 12}),
        encoding="utf-8",
    )
    assets = validate_policy_assets(bundle, envelope)
    executor = yaml.safe_load(
        (PACKAGE.parents[0] / "bookshelf_guarded_control_ros/config/guarded_policy_tool_executor.yaml").read_text(
            encoding="utf-8"
        )
    )
    executor["guarded_policy_tool_executor"]["ros__parameters"][
        "expected_bundle_sha256"
    ] = assets["policy_bundle_sha256"]
    executor_path = tmp_path / "executor.yaml"
    executor_path.write_text(yaml.safe_dump(executor), encoding="utf-8")

    result = validate_fail_closed_executor(
        executor_path, assets["policy_bundle_sha256"]
    )

    assert result["execution_authorized"] is False
