import math

import pytest

from bookshelf_shadow_ros.slot_orientation_audit import (
    SlotOrientationAuditAccumulator,
)


def _quaternion(axis, degrees):
    radians = math.radians(degrees)
    scale = math.sin(0.5 * radians)
    return [axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(0.5 * radians)]


def test_stable_reference_difference_is_classified_as_systematic():
    audit = SlotOrientationAuditAccumulator()
    reference = [0.0, 0.0, 0.0, 1.0]
    for angle in (4.8, 5.0, 5.1, 4.9):
        audit.add(_quaternion([1.0, 0.0, 0.0], angle), reference, 0.78)

    summary = audit.summary()

    assert summary["classification"] == "stable_systematic_orientation_difference"
    assert summary["temporally_stable"] is True
    assert summary["live_orientation_spread_deg"]["p95"] < 0.2
    assert summary["live_to_reference_rotation_error_deg"]["mean"] == pytest.approx(
        4.95, abs=0.02
    )
    assert summary["mean_live_axes"]["up_tilt_from_base_vertical_deg"] == pytest.approx(
        4.95, abs=0.02
    )


def test_variable_orientation_is_classified_as_temporal_noise():
    audit = SlotOrientationAuditAccumulator(stable_spread_p95_deg=1.0)
    reference = [0.0, 0.0, 0.0, 1.0]
    for angle in (-4.0, 0.0, 4.0, 0.0):
        audit.add(_quaternion([0.0, 1.0, 0.0], angle), reference, 0.80)

    summary = audit.summary()

    assert summary["classification"] == "temporally_variable_orientation_detection"
    assert summary["temporally_stable"] is False
    assert summary["live_orientation_spread_deg"]["p95"] > 1.0


def test_low_confidence_and_malformed_samples_are_rejected():
    audit = SlotOrientationAuditAccumulator(minimum_confidence=0.60)
    reference = [0.0, 0.0, 0.0, 1.0]
    assert not audit.add(reference, reference, 0.59)
    assert not audit.add([0.0, 0.0, 0.0, 0.0], reference, 0.90)

    summary = audit.summary()

    assert summary["classification"] == "insufficient_valid_samples"
    assert summary["samples"] == 2
    assert summary["accepted_samples"] == 0
