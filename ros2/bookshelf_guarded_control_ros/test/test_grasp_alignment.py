import numpy as np
import pytest

from bookshelf_guarded_control_ros.grasp_alignment import (
    derive_simulation_grasp_setback,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import (
    invert_transform,
    make_transform,
)


def _document():
    return {
        "calibrated_preinsert_target": {
            "ros__parameters": {
                "eef_book_translation_xyz": [0.0, 0.0, 0.18],
                "eef_book_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "eef_policy_tool_translation_xyz": [0.0, 0.0, 0.14],
                "eef_policy_tool_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "policy_observation_adapter": {
            "ros__parameters": {
                "eef_book_translation_xyz": [0.0, 0.0, 0.18],
                "eef_book_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "tool_offset_xyz": [0.0, 0.0, 0.14],
                "tool_offset_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "bookshelf_scene_manager": {
            "ros__parameters": {
                "held_book_center_tcp_xyz": [0.0, 0.0, 0.008],
                "held_book_quaternion_tcp_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    }


def _transform(parameters, translation_key, quaternion_key):
    return make_transform(parameters[translation_key], parameters[quaternion_key])


def test_grasp_setback_preserves_book_relative_policy_tool():
    original = _document()
    adjusted, report = derive_simulation_grasp_setback(original, 0.028)

    old_target = original["calibrated_preinsert_target"]["ros__parameters"]
    new_target = adjusted["calibrated_preinsert_target"]["ros__parameters"]
    old_book = _transform(
        old_target, "eef_book_translation_xyz", "eef_book_quaternion_xyzw"
    )
    old_policy = _transform(
        old_target,
        "eef_policy_tool_translation_xyz",
        "eef_policy_tool_quaternion_xyzw",
    )
    new_book = _transform(
        new_target, "eef_book_translation_xyz", "eef_book_quaternion_xyzw"
    )
    new_policy = _transform(
        new_target,
        "eef_policy_tool_translation_xyz",
        "eef_policy_tool_quaternion_xyzw",
    )

    assert invert_transform(new_book) @ new_policy == pytest.approx(
        invert_transform(old_book) @ old_policy
    )
    assert report["book_to_policy_tool_preserved"] is True
    assert (
        report["adjusted_book_to_tcp_translation_xyz_m"][0]
        - report["original_book_to_tcp_translation_xyz_m"][0]
    ) == pytest.approx(-0.028)


def test_grasp_setback_updates_adapter_and_scene_consistently():
    adjusted, _ = derive_simulation_grasp_setback(_document(), 0.028)
    target = adjusted["calibrated_preinsert_target"]["ros__parameters"]
    adapter = adjusted["policy_observation_adapter"]["ros__parameters"]

    assert adapter["eef_book_translation_xyz"] == pytest.approx(
        target["eef_book_translation_xyz"]
    )
    assert adapter["tool_offset_xyz"] == pytest.approx(
        target["eef_policy_tool_translation_xyz"]
    )


@pytest.mark.parametrize("value", [-0.001, 0.061, float("nan")])
def test_grasp_setback_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="physical_grasp_setback_m"):
        derive_simulation_grasp_setback(_document(), value)
