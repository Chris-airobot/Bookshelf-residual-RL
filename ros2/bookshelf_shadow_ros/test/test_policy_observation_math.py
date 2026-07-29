import math

import numpy as np
import pytest

from bookshelf_shadow_ros.policy_observation_math import (
    compute_policy_observation,
    invert_transform,
    make_transform,
    simulator_root_to_policy_book_transform,
    validate_detector_measurement,
)


def test_aligned_book_observation_matches_geometry():
    transform_slot_book = make_transform([0.078, 0.002, 0.003])
    transform_slot_tool = make_transform([0.020, -0.001, 0.010])

    raw, observation = compute_policy_observation(
        transform_slot_book,
        transform_slot_tool,
        book_size=(0.156, 0.034, 0.236),
        slot_depth=0.20,
        mode_observation=0.0,
        gripper_open=0.25,
    )

    np.testing.assert_allclose(
        raw,
        [
            0.0,
            0.0,
            0.044,
            -0.002,
            0.003,
            0.0,
            -0.058,
            -0.003,
            0.007,
            0.25,
            0.0,
            0.0,
        ],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(observation[1], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(observation[2], 0.55, atol=1.0e-6)
    np.testing.assert_allclose(observation[3], -0.04, atol=1.0e-6)
    np.testing.assert_allclose(observation[4], 0.06, atol=1.0e-6)
    np.testing.assert_allclose(observation[9], 0.25, atol=1.0e-6)


def test_yaw_and_tilt_use_slot_relative_book_axes():
    yaw = math.radians(10.0)
    pitch = math.radians(5.0)
    rotation_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_y = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    transform_slot_book = np.eye(4)
    transform_slot_book[:3, :3] = rotation_z @ rotation_y
    transform_slot_tool = np.eye(4)

    raw, _ = compute_policy_observation(transform_slot_book, transform_slot_tool)

    np.testing.assert_allclose(raw[5], yaw, atol=1.0e-6)
    expected_up = transform_slot_book[:3, :3] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(raw[10:12], expected_up[:2], atol=1.0e-6)


def test_transform_inverse_round_trip():
    transform = make_transform(
        [0.4, -0.2, 0.7],
        [0.1, -0.2, 0.3, 0.9],
    )
    np.testing.assert_allclose(invert_transform(transform) @ transform, np.eye(4), atol=1.0e-9)


def test_translation_and_rotation_error_signs_are_slot_relative():
    transform_slot_book = make_transform([0.078, 0.004, -0.003])
    transform_slot_tool = make_transform([0.020, 0.0, 0.0])
    raw, _ = compute_policy_observation(transform_slot_book, transform_slot_tool)

    assert raw[3] == pytest.approx(-0.004)
    assert raw[4] == pytest.approx(-0.003)

    yaw = math.radians(-7.0)
    transform_slot_book[:3, :3] = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    raw, _ = compute_policy_observation(transform_slot_book, transform_slot_tool)
    assert raw[5] == pytest.approx(yaw)


def test_observation_clips_large_but_finite_errors():
    raw, observation = compute_policy_observation(
        make_transform([0.078, 0.20, -0.20]),
        make_transform([1.0, -1.0, 1.0]),
    )
    assert np.all(np.isfinite(raw))
    assert np.all(observation <= 1.0)
    assert np.all(observation >= -1.0)
    assert observation[3] == -1.0
    assert observation[4] == -1.0


@pytest.mark.parametrize(
    ("width", "confidence", "expected"),
    [
        (None, 0.8, "waiting"),
        (0.04, None, "waiting"),
        (float("nan"), 0.8, "non-finite"),
        (0.04, float("nan"), "non-finite"),
        (0.01, 0.8, "outside"),
        (0.04, 0.2, "below"),
    ],
)
def test_detector_measurement_validation_fails_closed(width, confidence, expected):
    error = validate_detector_measurement(width, confidence)
    assert expected in error


def test_detector_measurement_validation_accepts_fresh_valid_values():
    assert (
        validate_detector_measurement(
            0.037,
            0.83,
            slot_width_age_s=0.01,
            confidence_age_s=0.01,
        )
        is None
    )


def test_detector_measurement_validation_rejects_stale_values():
    assert "slot width callback is stale" == validate_detector_measurement(
        0.037,
        0.83,
        slot_width_age_s=0.51,
        confidence_age_s=0.01,
    )


def test_semantic_policy_book_frame_matches_standing_simulator_cuboid():
    simulator_standing_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    transform_slot_simulator_root = np.eye(4)
    transform_slot_simulator_root[:3, :3] = simulator_standing_rotation
    transform_slot_simulator_root[:3, 3] = [0.078, 0.0, 0.006]

    transform_slot_policy_book = simulator_root_to_policy_book_transform(
        transform_slot_simulator_root
    )
    np.testing.assert_allclose(
        transform_slot_policy_book[:3, :3],
        np.eye(3),
        atol=1.0e-12,
    )

    raw, _ = compute_policy_observation(
        transform_slot_policy_book,
        make_transform([0.020, 0.0, 0.006]),
        book_size=(0.156, 0.034, 0.236),
    )
    np.testing.assert_allclose(raw[10:12], [0.0, 0.0], atol=1.0e-12)

    simulator_half = 0.5 * np.array([0.156, 0.236, 0.034])
    semantic_half = 0.5 * np.array([0.156, 0.034, 0.236])
    signs = np.array(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    simulator_corners = (
        simulator_standing_rotation @ (signs * simulator_half).T
    ).T
    semantic_corners = signs * semantic_half
    np.testing.assert_allclose(
        np.sort(simulator_corners, axis=0),
        np.sort(semantic_corners, axis=0),
        atol=1.0e-12,
    )
