"""Simulation-only xArm grasp alignment while preserving the policy tool."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .policy_tool_control_math import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


def derive_simulation_grasp_setback(document: dict, setback_m: float):
    """Move the book away from the TCP while preserving book-to-policy-tool."""

    setback_m = float(setback_m)
    if not math.isfinite(setback_m) or not 0.0 <= setback_m <= 0.06:
        raise ValueError("physical_grasp_setback_m must be in [0.0, 0.06]")

    result = deepcopy(document)
    target = _parameters(result, "calibrated_preinsert_target")
    adapter = _parameters(result, "policy_observation_adapter")
    scene = _parameters(result, "bookshelf_scene_manager")

    transform_eef_book = make_transform(
        target["eef_book_translation_xyz"],
        target["eef_book_quaternion_xyzw"],
    )
    transform_eef_policy_tool = make_transform(
        target["eef_policy_tool_translation_xyz"],
        target["eef_policy_tool_quaternion_xyzw"],
    )
    transform_tcp_book = make_transform(
        scene["held_book_center_tcp_xyz"],
        scene["held_book_quaternion_tcp_xyzw"],
    )
    transform_eef_tcp = transform_eef_book @ invert_transform(transform_tcp_book)
    transform_book_policy_tool = (
        invert_transform(transform_eef_book) @ transform_eef_policy_tool
    )

    transform_eef_book_adjusted = transform_eef_book @ make_transform(
        [setback_m, 0.0, 0.0]
    )
    transform_eef_policy_tool_adjusted = (
        transform_eef_book_adjusted @ transform_book_policy_tool
    )
    transform_tcp_book_adjusted = (
        invert_transform(transform_eef_tcp) @ transform_eef_book_adjusted
    )

    _set_transform(
        target,
        "eef_book",
        transform_eef_book_adjusted,
    )
    _set_transform(
        adapter,
        "eef_book",
        transform_eef_book_adjusted,
    )
    _set_transform(
        target,
        "eef_policy_tool",
        transform_eef_policy_tool_adjusted,
    )
    _set_transform(
        adapter,
        "tool_offset",
        transform_eef_policy_tool_adjusted,
    )
    _set_transform(
        scene,
        "held_book_center_tcp",
        transform_tcp_book_adjusted,
        quaternion_key="held_book_quaternion_tcp_xyzw",
    )

    reconstructed_book_policy_tool = (
        invert_transform(transform_eef_book_adjusted)
        @ transform_eef_policy_tool_adjusted
    )
    if not np.allclose(
        reconstructed_book_policy_tool,
        transform_book_policy_tool,
        atol=1.0e-10,
    ):
        raise RuntimeError("grasp setback changed the book-relative policy tool")

    report = {
        "simulation_only": True,
        "physical_grasp_setback_m": setback_m,
        "original_book_to_tcp_translation_xyz_m": (
            invert_transform(transform_tcp_book)[:3, 3].tolist()
        ),
        "adjusted_book_to_tcp_translation_xyz_m": (
            invert_transform(transform_tcp_book_adjusted)[:3, 3].tolist()
        ),
        "book_to_policy_tool_preserved": True,
    }
    return result, report


def _parameters(document: dict, node_name: str) -> dict:
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"configuration is missing {node_name}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"invalid parameters for {node_name}")
    return parameters


def _set_transform(
    parameters: dict,
    prefix: str,
    transform,
    *,
    quaternion_key: str | None = None,
) -> None:
    transform = np.asarray(transform, dtype=np.float64)
    translation_key = f"{prefix}_translation_xyz"
    if prefix == "tool_offset":
        translation_key = "tool_offset_xyz"
    elif prefix == "held_book_center_tcp":
        translation_key = "held_book_center_tcp_xyz"
    if quaternion_key is None:
        quaternion_key = f"{prefix}_quaternion_xyzw"
    parameters[translation_key] = transform[:3, 3].tolist()
    parameters[quaternion_key] = matrix_to_quaternion_xyzw(
        transform[:3, :3]
    ).tolist()
