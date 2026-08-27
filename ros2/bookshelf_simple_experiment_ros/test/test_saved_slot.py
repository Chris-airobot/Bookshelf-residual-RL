from pathlib import Path

import pytest
import yaml

from bookshelf_simple_experiment_ros.saved_slot_node import (
    DEFAULT_SAVED_SLOT_CONFIG,
    load_saved_slot,
)


def _document():
    return {
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


def test_saved_slot_loader_reads_reviewed_schema(tmp_path):
    path = tmp_path / "slot.yaml"
    path.write_text(yaml.safe_dump(_document()), encoding="utf-8")
    slot = load_saved_slot(path)
    assert slot.base_frame == "link_base"
    assert slot.translation_xyz == (0.8, 0.1, 0.2)
    assert slot.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert slot.width_m == 0.038
    assert slot.confidence == 0.86


def test_saved_slot_loader_rejects_zero_quaternion(tmp_path):
    document = _document()
    document["static_slot_environment_check"]["ros__parameters"][
        "static_slot_quaternion_xyzw"
    ] = [0.0, 0.0, 0.0, 0.0]
    path = tmp_path / "slot.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="quaternion is zero"):
        load_saved_slot(path)


@pytest.mark.skipif(
    not Path(DEFAULT_SAVED_SLOT_CONFIG).expanduser().exists(),
    reason="approved workstation slot YAML is not present",
)
def test_workstation_approved_slot_values():
    slot = load_saved_slot(DEFAULT_SAVED_SLOT_CONFIG)
    assert slot.translation_xyz == (
        0.8554391824906817,
        0.08412625750748041,
        0.17092253331755783,
    )
    assert slot.quaternion_xyzw == pytest.approx((
        0.0010688607844502452,
        -0.02066739075286218,
        0.03919702261622847,
        0.9990171719816006,
    ))
    assert slot.width_m == 0.037844330072402954
    assert slot.confidence == 0.8621852695941925
