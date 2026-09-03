import math

import numpy as np
import pytest

from bookshelf_simple_experiment_ros.per_grasp_calibration import (
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
