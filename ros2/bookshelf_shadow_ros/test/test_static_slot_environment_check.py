import math
from pathlib import Path

import numpy as np
import yaml

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.static_slot_environment_check import (
    ConsecutiveMatchGate,
    SlotCheckTolerances,
    compare_slot_measurement,
)


TOLERANCES = SlotCheckTolerances(
    maximum_translation_error_m=0.010,
    maximum_rotation_error_deg=5.0,
    maximum_width_error_m=0.005,
    minimum_confidence=0.60,
)


def _z_quaternion(degrees):
    angle = math.radians(degrees)
    return [0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)]


def test_matching_measurement_reports_metric_errors():
    reference = make_transform([0.8, 0.1, 0.25])
    measured = make_transform([0.806, 0.1, 0.25], _z_quaternion(3.0))
    result = compare_slot_measurement(
        reference,
        measured,
        reference_width_m=0.036,
        measured_width_m=0.038,
        confidence=0.75,
        tolerances=TOLERANCES,
    )
    assert result["matches"]
    assert np.isclose(result["translation_error_m"], 0.006)
    assert np.isclose(result["rotation_error_deg"], 3.0)
    assert np.isclose(result["width_error_m"], 0.002)


def test_each_gate_can_fail_and_boundaries_are_inclusive():
    reference = make_transform([0.0, 0.0, 0.0])
    measured = make_transform([0.010, 0.0, 0.0], _z_quaternion(5.0))
    boundary = compare_slot_measurement(
        reference,
        measured,
        reference_width_m=0.036,
        measured_width_m=0.041,
        confidence=0.60,
        tolerances=TOLERANCES,
    )
    assert boundary["matches"]

    failed = compare_slot_measurement(
        reference,
        make_transform([0.011, 0.0, 0.0], _z_quaternion(5.1)),
        reference_width_m=0.036,
        measured_width_m=0.042,
        confidence=0.59,
        tolerances=TOLERANCES,
    )
    assert not failed["matches"]
    assert failed["failed_checks"] == [
        "confidence",
        "translation",
        "rotation",
        "width",
    ]


def test_consecutive_gate_resets_immediately_after_a_mismatch():
    gate = ConsecutiveMatchGate(3)
    assert not gate.update(True)
    assert not gate.update(True)
    assert gate.matching_samples == 2
    assert not gate.update(False)
    assert gate.matching_samples == 0
    assert not gate.update(True)
    assert not gate.update(True)
    assert gate.update(True)
    assert gate.passed


def test_environment_check_reference_matches_preinsert_reference():
    config_dir = Path(__file__).parents[1] / "config"
    check = yaml.safe_load(
        (config_dir / "static_slot_environment_check.yaml").read_text()
    )["static_slot_environment_check"]["ros__parameters"]
    target = yaml.safe_load(
        (config_dir / "calibrated_preinsert_target.yaml").read_text()
    )["calibrated_preinsert_target"]["ros__parameters"]
    for suffix in (
        "translation_xyz",
        "quaternion_xyzw",
        "width_m",
        "transform_status",
    ):
        assert check[f"static_slot_{suffix}"] == target[f"static_slot_{suffix}"]
