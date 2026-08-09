import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.static_slot_capture import (
    APPROVAL_TOKEN,
    StaticSlotCaptureAccumulator,
    StaticSlotSample,
    promote_capture_candidate,
)


def _sample(index, translation, yaw_deg=0.0, width=0.040, confidence=0.90):
    yaw = math.radians(yaw_deg)
    quaternion = [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)]
    return StaticSlotSample(
        stamp_ns=index,
        transform_base_slot=make_transform(translation, quaternion),
        width_m=width,
        confidence=confidence,
    )


def test_capture_accumulator_rejects_one_large_pose_and_width_outlier():
    accumulator = StaticSlotCaptureAccumulator(
        minimum_samples=4,
        minimum_inlier_fraction=0.60,
        maximum_translation_deviation_m=0.010,
        maximum_rotation_deviation_deg=5.0,
        maximum_width_deviation_m=0.005,
    )
    for index, offset in enumerate((-0.0004, -0.0002, 0.0, 0.0002, 0.0004)):
        accumulator.add(
            _sample(
                index,
                [0.90 + offset, 0.10, 0.15],
                yaw_deg=0.1 * index,
                width=0.040 + 0.0001 * index,
                confidence=0.91,
            )
        )
    accumulator.add(
        _sample(99, [1.10, -0.20, 0.40], yaw_deg=25.0, width=0.070)
    )

    result = accumulator.result()

    assert result["input_samples"] == 6
    assert result["inlier_samples"] == 5
    np.testing.assert_allclose(
        result["translation_xyz"], [0.90, 0.10, 0.15], atol=1.0e-9
    )
    assert np.isclose(result["width_m"], 0.0402)
    assert result["translation_residual_m"]["max"] < 0.001
    assert result["rotation_residual_deg"]["max"] < 1.0


def test_capture_requires_enough_stable_inliers():
    accumulator = StaticSlotCaptureAccumulator(
        minimum_samples=4,
        minimum_inlier_fraction=0.81,
    )
    for index in range(4):
        accumulator.add(_sample(index, [0.9, 0.1, 0.15]))
    accumulator.add(_sample(4, [1.2, 0.1, 0.15], yaw_deg=20.0, width=0.070))
    with pytest.raises(ValueError, match="inlier fraction"):
        accumulator.result()


def _candidate_report(path: Path):
    report = {
        "schema_version": 1,
        "kind": "bookshelf_static_slot_capture_candidate",
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "human_approval_required": True,
        "valid": True,
        "reason": None,
        "candidate": {
            "translation_xyz": [0.8985, 0.0968, 0.1544],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "width_m": 0.03956,
            "confidence": 0.943,
            "transform_status": "captured_rgbd_static_unapproved",
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_promotion_requires_literal_manual_approval_token(tmp_path):
    candidate = tmp_path / "candidate.json"
    _candidate_report(candidate)
    config_dir = Path(__file__).parents[1] / "config"
    with pytest.raises(ValueError, match="Promotion requires"):
        promote_capture_candidate(
            candidate,
            config_dir,
            tmp_path / "trial.yaml",
            approval_token="yes",
        )


def test_promotion_generates_one_consistent_trial_config(tmp_path):
    candidate = tmp_path / "candidate.json"
    _candidate_report(candidate)
    config_dir = Path(__file__).parents[1] / "config"
    output = tmp_path / "trial_static_slot.yaml"

    provenance = promote_capture_candidate(
        candidate,
        config_dir,
        output,
        approval_token=APPROVAL_TOKEN,
    )

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    check = generated["static_slot_environment_check"]["ros__parameters"]
    target = generated["calibrated_preinsert_target"]["ros__parameters"]
    adapter = generated["policy_observation_adapter"]["ros__parameters"]
    expected_translation = [0.8985, 0.0968, 0.1544]
    assert check["static_slot_translation_xyz"] == expected_translation
    assert target["static_slot_translation_xyz"] == expected_translation
    assert adapter["configured_static_slot_translation_xyz"] == expected_translation
    assert check["static_slot_width_m"] == 0.03956
    assert target["static_slot_confidence"] == 0.943
    assert adapter["configured_static_slot_confidence"] == 0.943
    statuses = {
        check["static_slot_transform_status"],
        target["static_slot_transform_status"],
        adapter["static_slot_transform_status"],
    }
    assert len(statuses) == 1
    assert next(iter(statuses)).startswith("captured_rgbd_static_human_approved_")
    assert provenance["human_approval_recorded"] is True
    assert provenance["hardware_commanded"] is False
    assert output.with_suffix(".provenance.json").is_file()
