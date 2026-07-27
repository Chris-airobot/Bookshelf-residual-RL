import math

import numpy as np

from bookshelf_shadow_ros.policy_observation_math import (
    compute_policy_observation,
    invert_transform,
    make_transform,
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
