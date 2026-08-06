"""Pure offline checks for an unverified real-robot policy-tool candidate."""

from __future__ import annotations

import math

import numpy as np

from .calibrated_preinsert_target_math import (
    PreinsertTargetSpec,
    compute_calibrated_preinsert_target,
    transform_to_dict,
)
from .policy_observation_math import (
    OBSERVATION_LABELS,
    ObservationScales,
    invert_transform,
    make_transform,
)
from .policy_shadow_math import (
    MOTION_LABELS,
    POLICY_ACTION_LABELS,
    NumpyActorBundle,
    NominalInsertConfig,
    ResidualMotionConfig,
    combine_motion_delta,
    compute_insert_nominal_delta,
    scale_residual_action,
)


SIM_NOMINAL_BOOK_TOOL_TRANSLATION = np.array(
    [-0.03682894911272874, -0.0010947493841520109, 0.0007504753567338686],
    dtype=np.float64,
)
SIM_NOMINAL_BOOK_TOOL_QUATERNION = np.array(
    [
        0.7205555086260095,
        -0.021613755787561972,
        0.6927624553486299,
        -0.020317111231816707,
    ],
    dtype=np.float64,
)


def evaluate_hypothetical_preinsert_candidate(
    parameters: dict,
    bundle: NumpyActorBundle,
) -> dict:
    """Evaluate geometry and deterministic policy output without ROS or hardware."""

    status = str(parameters["policy_tool_transform_status"])
    require_verified = bool(parameters["require_verified_policy_tool_transform"])
    if not status.startswith("derived_unverified_"):
        raise ValueError(
            "Candidate policy-tool status must start with 'derived_unverified_'."
        )
    if require_verified:
        raise ValueError(
            "The dedicated offline candidate must not claim a verified transform."
        )

    transform_base_slot = make_transform(
        _vector(parameters, "configured_static_slot_translation_xyz", 3),
        _vector(parameters, "configured_static_slot_quaternion_xyzw", 4),
    )
    transform_eef_book = make_transform(
        _vector(parameters, "eef_book_translation_xyz", 3),
        _vector(parameters, "eef_book_quaternion_xyzw", 4),
    )
    transform_eef_policy_tool = make_transform(
        _vector(parameters, "tool_offset_xyz", 3),
        _vector(parameters, "tool_offset_quaternion_xyzw", 4),
    )
    scales = ObservationScales(
        rear_to_mouth=float(parameters["rear_to_mouth_obs_scale"]),
        front_to_back=float(parameters["front_to_back_obs_scale"]),
        lateral=float(parameters["lat_err_obs_scale"]),
        vertical=float(parameters["z_err_obs_scale"]),
        yaw=math.radians(float(parameters["yaw_err_obs_scale_deg"])),
        tool_to_book=float(parameters["tool_to_book_obs_scale"]),
    )
    spec = PreinsertTargetSpec(
        book_size=tuple(_vector(parameters, "book_size_xyz", 3)),
        slot_depth=float(parameters["slot_depth_m"]),
        standoff=float(parameters.get("offline_preinsert_standoff_m", 0.030)),
        vertical_offset=float(
            parameters.get("offline_preinsert_vertical_offset_m", 0.006)
        ),
        gripper_open=float(parameters.get("offline_target_gripper_open", 0.0)),
        observation_scales=scales,
    )
    target = compute_calibrated_preinsert_target(
        transform_base_slot,
        transform_eef_book,
        transform_eef_policy_tool=transform_eef_policy_tool,
        spec=spec,
    )

    normalized, actor_mean, action = bundle.predict(target.observation_12d)
    motion_config = ResidualMotionConfig()
    nominal_config = NominalInsertConfig()
    nominal = compute_insert_nominal_delta(target.raw_metrics, nominal_config)
    residual = scale_residual_action(action, motion_config)
    final = combine_motion_delta(nominal, residual, motion_config)

    transform_book_policy_tool = (
        invert_transform(transform_eef_book) @ transform_eef_policy_tool
    )
    expected_book_policy_tool = make_transform(
        SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
        SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    )
    transform_error = (
        invert_transform(expected_book_policy_tool) @ transform_book_policy_tool
    )
    translation_error = float(np.linalg.norm(transform_error[:3, 3]))
    rotation_error_deg = _rotation_angle_deg(transform_error[:3, :3])

    action_saturated = _labels_at_limit(action, POLICY_ACTION_LABELS, 1.0)
    normalized_saturated = _labels_at_limit(
        normalized,
        OBSERVATION_LABELS,
        float(bundle.obs_clip),
    )
    release_requested = bool(action[-1] > motion_config.release_threshold)
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            target.raw_metrics,
            target.observation_12d,
            normalized,
            actor_mean,
            action,
            nominal,
            residual,
            final,
        )
    )
    geometry_match = translation_error < 1.0e-9 and rotation_error_deg < 1.0e-6
    check_passed = bool(
        finite
        and geometry_match
        and not target.unexpected_clipped_labels
        and not release_requested
    )

    return {
        "check": "hypothetical_calibrated_preinsert_shadow",
        "check_passed": check_passed,
        "status": (
            "offline_candidate_passed"
            if check_passed
            else "offline_candidate_requires_review"
        ),
        "safety": {
            "shadow_only": True,
            "hardware_commanded": False,
            "execution_authorized": False,
            "ik_checked": False,
            "collision_checked": False,
            "reachability_checked": False,
        },
        "provenance": {
            "policy_tool_transform_status": status,
            "static_slot_transform_status": str(
                parameters["static_slot_transform_status"]
            ),
            "eef_book_transform_status": str(
                parameters["eef_book_transform_status"]
            ),
            "policy_bundle": str(bundle.path),
            "policy_bundle_sha256": bundle.sha256,
        },
        "candidate_transform_parity": {
            "transform_book_policy_tool": transform_to_dict(
                transform_book_policy_tool
            ),
            "expected_sim_nominal_transform_book_policy_tool": transform_to_dict(
                expected_book_policy_tool
            ),
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error_deg,
            "passed": geometry_match,
        },
        "target": {
            "preinsert_standoff_m": spec.standoff,
            "preinsert_vertical_offset_m": spec.vertical_offset,
            "transform_base_book": transform_to_dict(
                target.transform_base_book_target
            ),
            "transform_base_eef": transform_to_dict(
                target.transform_base_eef_target
            ),
            "transform_base_policy_tool": transform_to_dict(
                target.transform_base_policy_tool_target
            ),
        },
        "observation": {
            "raw_metrics": _labelled(target.raw_metrics, OBSERVATION_LABELS),
            "observation_12d": _labelled(
                target.observation_12d, OBSERVATION_LABELS
            ),
            "normalized_observation": _labelled(
                normalized, OBSERVATION_LABELS
            ),
            "expected_clipped_labels": list(target.expected_clipped_labels),
            "unexpected_clipped_labels": list(target.unexpected_clipped_labels),
            "vecnormalize_saturated_labels": normalized_saturated,
        },
        "policy": {
            "actor_mean": _labelled(actor_mean, POLICY_ACTION_LABELS),
            "action": _labelled(action, POLICY_ACTION_LABELS),
            "action_saturated_labels": action_saturated,
            "release_requested": release_requested,
            "nominal_delta": _labelled(nominal, MOTION_LABELS),
            "residual_delta": _labelled(residual, MOTION_LABELS),
            "final_delta": _labelled(final, MOTION_LABELS),
        },
    }


def _vector(parameters: dict, key: str, size: int) -> np.ndarray:
    value = np.asarray(parameters[key], dtype=np.float64).reshape(-1)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must be a finite {size}D vector.")
    return value


def _labelled(values, labels) -> dict[str, float]:
    return {
        label: float(value)
        for label, value in zip(labels, np.asarray(values).reshape(-1))
    }


def _labels_at_limit(values, labels, limit: float) -> list[str]:
    threshold = abs(float(limit)) - 1.0e-6
    return [
        label
        for label, value in zip(labels, np.asarray(values).reshape(-1))
        if abs(float(value)) >= threshold
    ]


def _rotation_angle_deg(rotation) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))
