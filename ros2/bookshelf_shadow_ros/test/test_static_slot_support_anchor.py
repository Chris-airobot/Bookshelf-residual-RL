import numpy as np
import pytest

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.static_slot_environment_check_node import (
    _anchor_slot_lower_edge_to_height,
)


def test_slot_lower_edge_is_anchored_to_measured_support_height():
    transform = make_transform([0.80, 0.10, 0.195])

    anchored, correction = _anchor_slot_lower_edge_to_height(
        transform,
        slot_height_m=0.236,
        support_height_base_m=0.015,
    )

    assert correction == pytest.approx(-0.062)
    assert anchored[2, 3] == pytest.approx(0.133)
    assert anchored[2, 3] - 0.5 * 0.236 == pytest.approx(0.015)
    assert transform[2, 3] == pytest.approx(0.195)


def test_support_anchor_moves_along_tilted_slot_up_axis():
    angle = np.deg2rad(10.0)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    transform[:3, 3] = [0.80, 0.10, 0.20]

    anchored, _ = _anchor_slot_lower_edge_to_height(
        transform,
        slot_height_m=0.236,
        support_height_base_m=0.015,
    )

    up = anchored[:3, 2]
    lower_edge = anchored[:3, 3] - 0.5 * 0.236 * up
    assert lower_edge[2] == pytest.approx(0.015)
    assert anchored[0, 3] == pytest.approx(transform[0, 3])
    assert anchored[1, 3] == pytest.approx(transform[1, 3])


def test_support_anchor_rejects_a_horizontal_up_axis():
    transform = np.eye(4)
    transform[:3, 2] = [1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="not sufficiently vertical"):
        _anchor_slot_lower_edge_to_height(
            transform,
            slot_height_m=0.236,
            support_height_base_m=0.015,
        )
