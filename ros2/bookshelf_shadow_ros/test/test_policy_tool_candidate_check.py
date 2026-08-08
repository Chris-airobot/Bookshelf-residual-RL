import numpy as np

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.policy_tool_candidate_check import (
    SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
)
from bookshelf_shadow_ros.policy_observation_math import invert_transform


def test_candidate_eef_tool_reproduces_nominal_simulator_book_tool_transform():
    transform_eef_book = make_transform(
        [0.008124357683356356, -0.010156856549182705, 0.12425871757162561],
        [
            0.47569361268145194,
            0.46802866719711034,
            0.5317004002617697,
            0.5214973038254745,
        ],
    )
    transform_eef_policy_tool = make_transform(
        [0.009119326451322136, -0.04695895701970321, 0.12260082121430517],
        [
            -0.7018277662283674,
            -0.03279647137253109,
            -0.00294897656373922,
            0.7115851892455586,
        ],
    )
    actual = invert_transform(transform_eef_book) @ transform_eef_policy_tool
    expected = make_transform(
        SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
        SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)
