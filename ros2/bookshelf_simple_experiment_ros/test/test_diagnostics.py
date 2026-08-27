import math

import numpy as np
import yaml

from bookshelf_simple_experiment_ros.preinsert_node import (
    _compact_pose,
    _frozen_slot_document,
    _rotation_matrix_to_rpy_degrees,
)
from bookshelf_simple_experiment_ros.saved_slot_node import load_saved_slot


def test_identity_relative_orientation_reports_zero_rpy():
    assert _rotation_matrix_to_rpy_degrees(np.eye(3)) == (0.0, -0.0, 0.0)


def test_diagnostic_rpy_uses_xyz_roll_pitch_yaw_degrees():
    yaw = math.radians(30.0)
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(
        _rotation_matrix_to_rpy_degrees(rotation), [0.0, 0.0, 30.0], atol=1e-12
    )


def test_compact_pose_has_position_and_xyzw_quaternion():
    text = _compact_pose(np.eye(4))
    assert text == "p=[+0.000000,+0.000000,+0.000000] q=[+0.000000,+0.000000,+0.000000,+1.000000]"


def test_frozen_slot_document_uses_base_frame_pose():
    transform = np.eye(4)
    transform[:3, 3] = [0.8, 0.1, 0.2]
    document = _frozen_slot_document("link_base", transform, 0.038, 0.86)
    assert document == {
        "static_slot_environment_check": {"ros__parameters": {
            "base_frame": "link_base",
            "static_slot_translation_xyz": [0.8, 0.1, 0.2],
            "static_slot_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "static_slot_width_m": 0.038,
        }},
        "calibrated_preinsert_target": {"ros__parameters": {
            "static_slot_confidence": 0.86,
        }},
    }


def test_frozen_slot_document_can_be_replayed_by_saved_slot_mode(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(
        yaml.safe_dump(_frozen_slot_document("link_base", np.eye(4), 0.038, 0.86)),
        encoding="utf-8",
    )
    slot = load_saved_slot(path)
    assert slot.base_frame == "link_base"
    assert slot.translation_xyz == (0.0, 0.0, 0.0)
    assert slot.width_m == 0.038
    assert slot.confidence == 0.86
