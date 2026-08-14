"""Offline regression for a regenerated EEF-to-book calibration candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .calibrated_preinsert_target_math import (
    PreinsertTargetSpec,
    compute_preserved_tcp_orientation_preinsert_target,
    transform_to_dict,
)
from .policy_observation_math import (
    OBSERVATION_LABELS,
    ObservationScales,
    invert_transform,
    make_transform,
)
from .policy_tool_candidate_check import (
    SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
)


def evaluate_book_calibration_candidate(
    base_parameters: dict,
    candidate_parameters: dict,
    recorded_context: dict,
) -> dict:
    """Compare stale and regenerated calibrations using one recorded robot pose."""

    candidate_book_status = str(candidate_parameters["eef_book_transform_status"])
    candidate_tool_status = str(candidate_parameters["policy_tool_transform_status"])
    if "candidate" not in candidate_book_status:
        raise ValueError("EEF-to-book override must remain explicitly candidate-labelled.")
    if not candidate_tool_status.startswith("derived_unverified_"):
        raise ValueError("Policy-tool override must remain derived_unverified.")
    if "candidate" not in candidate_tool_status:
        raise ValueError("Policy-tool override must remain explicitly candidate-labelled.")

    merged_candidate = dict(base_parameters)
    merged_candidate.update(candidate_parameters)
    stale = _evaluate_one(base_parameters, recorded_context)
    candidate = _evaluate_one(merged_candidate, recorded_context)

    transform_book_tool = (
        invert_transform(_eef_book_transform(merged_candidate))
        @ _eef_policy_tool_transform(merged_candidate)
    )
    expected_book_tool = make_transform(
        SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
        SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    )
    parity_error = invert_transform(expected_book_tool) @ transform_book_tool
    parity_translation_error_m = float(np.linalg.norm(parity_error[:3, 3]))
    parity_rotation_error_deg = _rotation_angle_deg(parity_error[:3, :3])
    parity_passed = bool(
        parity_translation_error_m < 1.0e-9
        and parity_rotation_error_deg < 1.0e-6
    )

    maximum_orientation_error_deg = float(
        recorded_context["maximum_preserved_book_orientation_error_deg"]
    )
    maximum_translation_m = float(
        recorded_context["maximum_candidate_target_translation_m"]
    )
    stale_reproduced = bool(
        stale["book_orientation_error_deg"] > maximum_orientation_error_deg
        and "yaw_err" in stale["unexpected_observation_clips"]
    )
    candidate_passed = bool(
        candidate["finite"]
        and candidate["book_orientation_error_deg"]
        <= maximum_orientation_error_deg
        and not candidate["unexpected_observation_clips"]
        and candidate["book_center_error_m"] < 1.0e-9
        and candidate["tcp_orientation_change_deg"] < 1.0e-6
        and candidate["target_tcp_translation_m"] <= maximum_translation_m
        and parity_passed
    )
    check_passed = bool(stale_reproduced and candidate_passed)

    return {
        "schema_version": 1,
        "kind": "bookshelf_book_calibration_candidate_preinsert_regression",
        "check_passed": check_passed,
        "status": (
            "candidate_passed_offline_regression"
            if check_passed
            else "candidate_requires_review"
        ),
        "safety": {
            "shadow_only": True,
            "hardware_commanded": False,
            "plan_requested": False,
            "execution_authorized": False,
            "active_configuration_modified": False,
            "selection_authorized": False,
        },
        "limits": {
            "maximum_preserved_book_orientation_error_deg": (
                maximum_orientation_error_deg
            ),
            "maximum_candidate_target_translation_m": maximum_translation_m,
        },
        "stale_configuration": stale,
        "candidate_configuration": candidate,
        "candidate_transform_parity": {
            "passed": parity_passed,
            "translation_error_m": parity_translation_error_m,
            "rotation_error_deg": parity_rotation_error_deg,
            "transform_book_policy_tool": transform_to_dict(transform_book_tool),
            "expected_simulator_transform_book_policy_tool": transform_to_dict(
                expected_book_tool
            ),
        },
        "regression": {
            "stale_failure_reproduced": stale_reproduced,
            "candidate_passed": candidate_passed,
            "orientation_error_improvement_deg": (
                stale["book_orientation_error_deg"]
                - candidate["book_orientation_error_deg"]
            ),
            "unexpected_clips_removed": sorted(
                set(stale["unexpected_observation_clips"])
                - set(candidate["unexpected_observation_clips"])
            ),
        },
        "provenance": {
            "recorded_context_source": str(recorded_context.get("source", "")),
            "stale_eef_book_transform_status": str(
                base_parameters["eef_book_transform_status"]
            ),
            "candidate_eef_book_transform_status": candidate_book_status,
            "candidate_policy_tool_transform_status": candidate_tool_status,
        },
    }


def _evaluate_one(parameters: dict, context: dict) -> dict:
    transform_base_slot = _payload_transform(context["transform_base_slot"])
    transform_base_eef = _matrix_transform(
        context["transform_base_eef_current"]["matrix"]
    )
    transform_eef_tcp = _payload_transform(context["transform_eef_tcp"])
    transform_base_tcp = transform_base_eef @ transform_eef_tcp
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
        standoff=float(parameters["preinsert_standoff_m"]),
        vertical_offset=float(parameters["preinsert_vertical_offset_m"]),
        gripper_open=float(parameters["target_gripper_open"]),
        observation_scales=scales,
    )
    target, diagnostics = compute_preserved_tcp_orientation_preinsert_target(
        transform_base_slot,
        _eef_book_transform(parameters),
        transform_base_eef,
        transform_base_tcp,
        transform_eef_policy_tool=_eef_policy_tool_transform(parameters),
        spec=spec,
    )
    target_tcp_delta = (
        diagnostics["transform_base_tcp_target"][:3, 3]
        - transform_base_tcp[:3, 3]
    )
    finite_values = (
        target.raw_metrics,
        target.observation_12d,
        diagnostics["transform_base_tcp_target"],
    )
    return {
        "finite": bool(all(np.all(np.isfinite(value)) for value in finite_values)),
        "eef_book_transform_status": str(parameters["eef_book_transform_status"]),
        "policy_tool_transform_status": str(
            parameters["policy_tool_transform_status"]
        ),
        "book_orientation_error_deg": float(
            diagnostics["book_orientation_error_deg"]
        ),
        "book_center_error_m": float(diagnostics["book_center_error_m"]),
        "tcp_orientation_change_deg": float(
            diagnostics["tcp_orientation_change_deg"]
        ),
        "target_tcp_translation_m": float(np.linalg.norm(target_tcp_delta)),
        "target_minus_current_tcp_translation_xyz_m": [
            float(value) for value in target_tcp_delta
        ],
        "expected_observation_clips": list(target.expected_clipped_labels),
        "unexpected_observation_clips": list(target.unexpected_clipped_labels),
        "raw_metrics": _labelled(target.raw_metrics, OBSERVATION_LABELS),
        "observation_12d": _labelled(target.observation_12d, OBSERVATION_LABELS),
        "transform_base_tcp_current": transform_to_dict(transform_base_tcp),
        "transform_base_tcp_target": transform_to_dict(
            diagnostics["transform_base_tcp_target"]
        ),
        "transform_base_book_target": transform_to_dict(
            target.transform_base_book_target
        ),
    }


def _eef_book_transform(parameters: dict) -> np.ndarray:
    return make_transform(
        _vector(parameters, "eef_book_translation_xyz", 3),
        _vector(parameters, "eef_book_quaternion_xyzw", 4),
    )


def _eef_policy_tool_transform(parameters: dict) -> np.ndarray:
    return make_transform(
        _vector(parameters, "eef_policy_tool_translation_xyz", 3),
        _vector(parameters, "eef_policy_tool_quaternion_xyzw", 4),
    )


def _payload_transform(payload: dict) -> np.ndarray:
    return make_transform(
        payload["translation_xyz_m"], payload["quaternion_xyzw"]
    )


def _matrix_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Recorded transform matrix must be finite and 4x4.")
    return transform


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


def _rotation_angle_deg(rotation) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def load_ros_parameters(path, node_name: str) -> dict:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} has no parameters for {node_name}.") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} parameters for {node_name} must be a mapping.")
    return parameters


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check a book-calibration candidate without ROS or hardware."
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--recorded-context", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    base = load_ros_parameters(args.base_config, "calibrated_preinsert_target")
    candidate = load_ros_parameters(
        args.candidate_config, "calibrated_preinsert_target"
    )
    context = json.loads(
        Path(args.recorded_context).read_text(encoding="utf-8")
    )
    report = evaluate_book_calibration_candidate(base, candidate, context)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Report: {output}")
    print(f"Passed: {report['check_passed']}")
    print(
        "Orientation error: "
        f"{report['stale_configuration']['book_orientation_error_deg']:.3f} -> "
        f"{report['candidate_configuration']['book_orientation_error_deg']:.3f} deg"
    )
    print(
        "Unexpected clips: "
        f"{report['stale_configuration']['unexpected_observation_clips']} -> "
        f"{report['candidate_configuration']['unexpected_observation_clips']}"
    )
    print(
        "Candidate TCP translation: "
        f"{report['candidate_configuration']['target_tcp_translation_m']:.6f} m"
    )
    print("Hardware commanded: False")
    return 0 if report["check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
