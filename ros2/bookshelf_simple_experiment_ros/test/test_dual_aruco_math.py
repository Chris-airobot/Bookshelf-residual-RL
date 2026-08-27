from pathlib import Path

import numpy as np
import yaml

from bookshelf_simple_experiment_ros.dual_aruco_math import (
    RobustTransformAccumulator, derive_secondary_book, load_reference_book_transform,
    load_secondary_book_transform, marker_object_points)
from bookshelf_simple_experiment_ros.geometry import invert_transform, make_transform


PACKAGE = Path(__file__).resolve().parents[1]


def test_reference_mount_is_exact_reviewed_marker_zero_geometry():
    copied, transform_reference_book = load_reference_book_transform(
        PACKAGE / "config/reference_marker0_book_mount.yaml")
    with (PACKAGE.parent / "bookshelf_shadow_ros/config/real_book_aruco0_mount.yaml").open() as stream:
        reviewed = yaml.safe_load(stream)
    for key in ("dictionary", "marker_id", "marker_black_size_m",
                "marker_center_in_book_m", "rotation_book_marker"):
        assert copied[key] == reviewed[key]
    assert np.all(np.isfinite(transform_reference_book))


def test_simultaneous_frame_composition_and_secondary_reconstruction():
    camera_reference = make_transform([0.1, -0.2, 0.7])
    reference_secondary = make_transform([0.08, 0.01, -0.02])
    camera_secondary = camera_reference @ reference_secondary
    computed = invert_transform(camera_reference) @ camera_secondary
    reference_book = make_transform([-0.03, 0.04, 0.02])
    secondary_book = derive_secondary_book(computed, reference_book)
    assert np.allclose(computed, reference_secondary)
    assert np.allclose(camera_secondary @ secondary_book, camera_reference @ reference_book)


def test_robust_accumulator_rejects_obvious_outlier():
    accumulator = RobustTransformAccumulator(0.01, 5.0)
    for offset in (0.0, 0.0002, -0.0001):
        accumulator.add(make_transform([0.08 + offset, 0.01, -0.02]))
    accumulator.add(make_transform([0.5, 0.5, 0.5]))
    result = accumulator.result()
    assert result["input_samples"] == 4
    assert result["accepted_samples"] == 3
    assert np.allclose(result["transform"][:3, 3], [0.080033333333, 0.01, -0.02])


def test_saved_secondary_schema_loads_runtime_transform(tmp_path):
    path = tmp_path / "secondary.yaml"
    path.write_text(yaml.safe_dump({"dictionary": "DICT_ARUCO_ORIGINAL",
        "marker_id": 10, "marker_black_size_m": 0.039,
        "transform_secondary_book": {"direction": "T_secondary_book",
        "translation_xyz_m": [0.1, 0.2, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}), encoding="utf-8")
    loaded, transform = load_secondary_book_transform(path)
    assert loaded["marker_id"] == 10
    assert np.allclose(transform, make_transform([0.1, 0.2, 0.3]))


def test_each_marker_uses_its_configured_black_square_size():
    assert np.isclose(np.ptp(marker_object_points(0.039)[:, 0]), 0.039)
    assert np.isclose(np.ptp(marker_object_points(0.052)[:, 0]), 0.052)
