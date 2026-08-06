import math
from pathlib import Path

import numpy as np
import yaml

from bookshelf_shadow_ros.marker_book_calibration import (
    CalibrationSample,
    DEFAULT_BOOK_FROM_MARKER_ROTATION,
    MarkerBookCalibrationAccumulator,
    average_quaternions_xyzw,
    compose_eef_book_transform,
    make_book_marker_transform,
    quaternion_medoid_xyzw,
)
from bookshelf_shadow_ros.policy_observation_math import make_transform


def _sample(index, transform):
    return CalibrationSample(
        frame_index=index,
        stamp_ns=index,
        reprojection_error_px=0.2,
        marker_depth_m=0.4,
        depth_error_m=0.001,
        transform_camera_marker=np.eye(4),
        transform_eef_book=transform,
    )


def test_marker_axis_mapping_matches_spine_cardboard_mount():
    rotation = DEFAULT_BOOK_FROM_MARKER_ROTATION
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0])
    assert np.linalg.det(rotation) == 1.0


def test_committed_marker_center_matches_physical_edge_measurements():
    config = Path(__file__).parents[1] / "config" / "real_book_aruco0_mount.yaml"
    mount = yaml.safe_load(config.read_text(encoding="utf-8"))
    depth, thickness, height = mount["book_size_xyz_m"]
    marker_size = mount["marker_black_size_m"]
    expected = [
        -0.5 * depth - mount["cardboard_thickness_m"],
        0.5 * thickness
        + mount["spine_edge_to_marker_black_edge_m"]
        - 0.5 * marker_size,
        0.5 * height
        - mount["top_edge_to_marker_black_edge_m"]
        - 0.5 * marker_size,
    ]
    center = mount["marker_center_in_book_m"]
    actual = [center["x"], center["y"], center["z"]]
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_transform_chain_recovers_known_eef_book_pose():
    transform_eef_camera = make_transform(
        [0.08, -0.03, 0.02], [0.0, 0.0, math.sin(0.2), math.cos(0.2)]
    )
    transform_book_marker = make_book_marker_transform([-0.0235, -0.019, 0.0635])
    expected_eef_book = make_transform(
        [0.02, 0.01, 0.16], [math.sin(0.1), 0.0, 0.0, math.cos(0.1)]
    )
    transform_camera_marker = (
        np.linalg.inv(transform_eef_camera)
        @ expected_eef_book
        @ transform_book_marker
    )

    actual = compose_eef_book_transform(
        transform_eef_camera,
        transform_camera_marker,
        transform_book_marker,
    )
    np.testing.assert_allclose(actual, expected_eef_book, atol=1.0e-10)


def test_quaternion_average_handles_antipodal_values():
    expected = np.array([0.0, 0.0, math.sin(0.15), math.cos(0.15)])
    actual = average_quaternions_xyzw([expected, -expected, expected])
    assert abs(float(np.dot(actual, expected))) > 1.0 - 1.0e-12


def test_quaternion_medoid_is_not_pulled_by_large_outlier():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    outlier = np.array([0.0, 0.0, 0.5, 0.8660254])
    actual = quaternion_medoid_xyzw([identity] * 5 + [outlier])
    np.testing.assert_allclose(actual, identity, atol=1.0e-12)


def test_accumulator_rejects_large_pose_outlier():
    accumulator = MarkerBookCalibrationAccumulator(
        maximum_translation_deviation_m=0.010,
        maximum_rotation_deviation_deg=5.0,
    )
    for index, offset in enumerate((-0.0004, -0.0002, 0.0, 0.0002, 0.0004)):
        accumulator.add(_sample(index, make_transform([0.10 + offset, -0.02, 0.05])))
    accumulator.add(
        _sample(99, make_transform([0.20, 0.10, -0.04], [0.0, 0.0, 0.5, 0.8660254]))
    )

    result = accumulator.result()

    assert result["input_samples"] == 6
    assert result["inlier_samples"] == 5
    assert result["inlier_fraction"] == 5.0 / 6.0
    np.testing.assert_allclose(result["translation_xyz_m"], [0.10, -0.02, 0.05], atol=1.0e-9)
