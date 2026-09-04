import math

import numpy as np
import pytest

from bookshelf_simple_experiment_ros.per_grasp_calibration import (
    FreshMarkerSampleGate,
    robust_average_transforms,
    select_eef_book_transform,
    semantic_held_gripper_observation,
)


def _transform(translation, yaw=0.0):
    value = np.eye(4)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    value[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    value[:3, 3] = translation
    return value


def test_robust_transform_mean_rejects_translation_and_rotation_outliers():
    samples = [
        _transform([0.005 + index * 1e-5, -0.002, 0.183], math.radians(2.0))
        for index in range(20)
    ]
    samples.extend([
        _transform([0.050, -0.002, 0.183], math.radians(2.0)),
        _transform([0.005, -0.002, 0.183], math.radians(40.0)),
    ])
    result, diagnostics = robust_average_transforms(samples)
    assert diagnostics["accepted_count"] == 20
    assert diagnostics["rejected_count"] == 2
    np.testing.assert_allclose(result[:3, 3], [0.005095, -0.002, 0.183], atol=1e-12)
    np.testing.assert_allclose(
        result[:3, :3], _transform([0, 0, 0], math.radians(2.0))[:3, :3], atol=1e-12
    )


def test_transform_selection_uses_frozen_value_or_explicit_fallback():
    fixed = _transform([1.0, 2.0, 3.0])
    frozen = _transform([4.0, 5.0, 6.0])
    np.testing.assert_array_equal(select_eef_book_transform(None, fixed), fixed)
    np.testing.assert_array_equal(select_eef_book_transform(frozen, fixed), frozen)


def test_semantic_gripper_mapping_preserves_measurement_only_for_diagnostics():
    assert semantic_held_gripper_observation(0.388235, 0.009838026859259968) == pytest.approx(
        0.009838026859259968
    )
    with pytest.raises(ValueError):
        semantic_held_gripper_observation(0.388235, 1.1)


def test_stale_cached_transform_is_rejected_on_every_read():
    gate = FreshMarkerSampleGate(0.25)
    for index in range(30):
        assert not gate.accept(8_123_000_000, 10_000_000_000 + index * 100_000_000)
    diagnostics = gate.diagnostics(12_000_000_000)
    assert diagnostics["unique_fresh_samples"] == 0
    assert diagnostics["stale_samples_rejected"] == 30


def test_repeated_fresh_timestamp_counts_only_once():
    gate = FreshMarkerSampleGate(0.25)
    stamp = 10_000_000_000
    assert gate.accept(stamp, stamp + 10_000_000)
    for index in range(29):
        assert not gate.accept(stamp, stamp + 20_000_000 + index * 1_000_000)
    diagnostics = gate.diagnostics(stamp + 100_000_000)
    assert diagnostics["unique_fresh_samples"] == 1
    assert diagnostics["duplicate_samples_rejected"] == 29


def test_enough_fresh_unique_samples_pass_minimum():
    gate = FreshMarkerSampleGate(0.25)
    for index in range(20):
        stamp = 10_000_000_000 + index * 100_000_000
        assert gate.accept(stamp, stamp + 20_000_000)
    gate.require_minimum(20)
    diagnostics = gate.diagnostics(stamp + 20_000_000)
    assert diagnostics["unique_fresh_samples"] == 20
    assert diagnostics["minimum_marker_age_at_read_s"] == pytest.approx(0.02)
    assert diagnostics["maximum_marker_age_at_read_s"] == pytest.approx(0.02)
    assert diagnostics["newest_accepted_sample_age_s"] == pytest.approx(0.02)
    assert diagnostics["oldest_accepted_sample_age_s"] == pytest.approx(1.92)


def test_mixed_samples_count_only_fresh_unique_timestamps():
    gate = FreshMarkerSampleGate(0.25)
    now = 20_000_000_000
    fresh = [now - 10_000_000, now - 20_000_000, now - 30_000_000]
    for stamp in fresh:
        assert gate.accept(stamp, now)
    assert not gate.accept(fresh[0], now)
    assert not gate.accept(now - 1_000_000_000, now)
    gate.reject_lookup()
    diagnostics = gate.diagnostics()
    assert diagnostics["total_reads_attempted"] == 6
    assert diagnostics["unique_fresh_samples"] == 3
    assert diagnostics["duplicate_samples_rejected"] == 1
    assert diagnostics["stale_samples_rejected"] == 1
    assert diagnostics["lookup_samples_rejected"] == 1


def test_insufficient_fresh_unique_samples_fail_requirement():
    gate = FreshMarkerSampleGate(0.25)
    for index in range(19):
        stamp = 10_000_000_000 + index * 10_000_000
        assert gate.accept(stamp, stamp + 5_000_000)
    with pytest.raises(ValueError, match="19/20 required"):
        gate.require_minimum(20)
