import numpy as np
import pytest

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.policy_tool_transform_extraction import (
    derive_xarm_policy_tool_transform,
    representative_transform,
)


def test_representative_transform_uses_median_translation():
    transforms = np.asarray(
        [
            make_transform([0.030, 0.001, -0.002]),
            make_transform([0.032, 0.000, -0.001]),
            make_transform([0.034, -0.001, 0.000]),
        ]
    )

    representative, dispersion = representative_transform(transforms)

    np.testing.assert_allclose(representative[:3, 3], [0.032, 0.0, -0.001])
    assert dispersion["samples"] == 3
    assert dispersion["translation_norm_m"]["max"] > 0.0


def test_xarm_virtual_tool_reconstructs_simulator_book_tool_transform():
    transform_book_policy_tool = make_transform(
        [0.032, -0.004, 0.002], [0.0, 0.0, 0.0871557, 0.9961947]
    )
    transform_eef_book = make_transform(
        [0.008, -0.010, 0.124], [0.47, 0.46, 0.53, 0.52]
    )
    transform_eef_tcp = make_transform([0.0, 0.0, 0.172])

    result = derive_xarm_policy_tool_transform(
        transform_book_policy_tool,
        transform_eef_book,
        transform_eef_tcp,
    )

    np.testing.assert_allclose(
        result["reconstructed_policy_book_policy_tool"],
        transform_book_policy_tool,
        atol=1.0e-10,
    )
    assert result["round_trip_translation_error_m"] == pytest.approx(0.0)
    assert result["round_trip_rotation_error_deg"] == pytest.approx(0.0)
