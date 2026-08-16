"""Generate synchronized, unapproved held-book calibration candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .book_calibration_candidate_check import (
    evaluate_book_calibration_candidate,
)
from .calibrated_preinsert_target_math import transform_to_dict
from .policy_observation_math import invert_transform, make_transform
from .policy_tool_candidate_check import (
    SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
)
from .policy_tool_transform_extraction import derive_xarm_policy_tool_transform


def generate_supervised_candidate(
    held_book_report: dict,
    scene_document: dict,
    eef_tcp_context: dict,
    base_target_parameters: dict,
    *,
    source_report_sha256: str,
    source_scene_sha256: str,
    source_context_sha256: str,
    source_base_target_sha256: str,
) -> tuple[dict, dict, dict]:
    """Build matching target and scene candidates from one stable live mean."""

    _validate_held_book_report(held_book_report)
    scene_candidate = deepcopy(scene_document)
    scene_parameters = _ros_parameters(
        scene_candidate, "bookshelf_scene_manager", "scene configuration"
    )

    live_payload = held_book_report["live_candidate_transform_tcp_book"]
    transform_tcp_book = _payload_transform(live_payload)
    transform_eef_tcp = _payload_transform(eef_tcp_context["transform_eef_tcp"])
    transform_eef_book = transform_eef_tcp @ transform_tcp_book
    transform_book_policy_tool = make_transform(
        SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
        SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    )
    derived = derive_xarm_policy_tool_transform(
        transform_book_policy_tool,
        transform_eef_book,
        transform_eef_tcp,
    )
    transform_eef_policy_tool = derived["transform_eef_policy_tool"]
    transform_tcp_policy_tool = (
        invert_transform(transform_eef_tcp) @ transform_eef_policy_tool
    )

    identity_error = (
        invert_transform(transform_book_policy_tool)
        @ invert_transform(transform_eef_book)
        @ transform_eef_policy_tool
    )
    parity_translation_error_m = float(np.linalg.norm(identity_error[:3, 3]))
    parity_rotation_error_deg = _rotation_angle_deg(identity_error[:3, :3])
    if parity_translation_error_m >= 1.0e-9 or parity_rotation_error_deg >= 1.0e-6:
        raise ValueError("derived policy-tool transform failed simulator parity")

    zero_delta_target_tcp = (
        transform_tcp_policy_tool @ invert_transform(transform_tcp_policy_tool)
    )
    zero_delta_translation_error_m = float(
        np.linalg.norm(zero_delta_target_tcp[:3, 3])
    )
    zero_delta_rotation_error_deg = _rotation_angle_deg(
        zero_delta_target_tcp[:3, :3]
    )
    if (
        zero_delta_translation_error_m >= 1.0e-9
        or zero_delta_rotation_error_deg >= 1.0e-6
    ):
        raise ValueError("zero policy delta did not preserve the current TCP")

    source_scene_transform = make_transform(
        scene_parameters["held_book_center_tcp_xyz"],
        scene_parameters["held_book_quaternion_tcp_xyzw"],
    )
    source_difference = _transform_difference(
        source_scene_transform, transform_tcp_book
    )

    candidate_id = source_report_sha256[:12]
    book_status = f"measured_live_supervised_candidate_{candidate_id}"
    tool_status = f"derived_unverified_live_supervised_candidate_{candidate_id}"
    eef_book_payload = transform_to_dict(transform_eef_book)
    eef_tool_payload = transform_to_dict(transform_eef_policy_tool)
    tcp_book_payload = transform_to_dict(transform_tcp_book)

    target_candidate = {
        "calibrated_preinsert_target": {
            "ros__parameters": {
                "eef_book_translation_xyz": eef_book_payload[
                    "translation_xyz_m"
                ],
                "eef_book_quaternion_xyzw": eef_book_payload[
                    "quaternion_xyzw"
                ],
                "eef_book_transform_status": book_status,
                "eef_policy_tool_translation_xyz": eef_tool_payload[
                    "translation_xyz_m"
                ],
                "eef_policy_tool_quaternion_xyzw": eef_tool_payload[
                    "quaternion_xyzw"
                ],
                "policy_tool_transform_status": tool_status,
            }
        },
        "policy_observation_adapter": {
            "ros__parameters": {
                "book_pose_source": "eef_fixed",
                "latch_eef_book_from_marker": False,
                "use_configured_eef_book_transform": True,
                "eef_book_translation_xyz": eef_book_payload[
                    "translation_xyz_m"
                ],
                "eef_book_quaternion_xyzw": eef_book_payload[
                    "quaternion_xyzw"
                ],
                "eef_book_transform_status": book_status,
                "tool_offset_xyz": eef_tool_payload["translation_xyz_m"],
                "tool_offset_quaternion_xyzw": eef_tool_payload[
                    "quaternion_xyzw"
                ],
                "policy_tool_transform_status": tool_status,
                "require_verified_policy_tool_transform": False,
            }
        },
    }
    offline_regression = evaluate_book_calibration_candidate(
        base_target_parameters,
        target_candidate["calibrated_preinsert_target"]["ros__parameters"],
        eef_tcp_context,
    )
    if not bool(offline_regression["check_passed"]):
        raise ValueError("candidate failed the recorded pre-insertion regression")

    scene_parameters["hardware_measurements_confirmed"] = False
    scene_parameters["allow_local_insertion"] = False
    scene_parameters["held_book_enabled"] = True
    scene_parameters["require_held_book_pose_check"] = True
    scene_parameters["held_book_pose_check_topic"] = (
        "/bookshelf_scene/held_book_pose_check_passed"
    )
    scene_parameters.setdefault("held_book_pose_check_max_age_s", 1.0)
    scene_parameters["held_book_center_tcp_xyz"] = tcp_book_payload[
        "translation_xyz_m"
    ]
    scene_parameters["held_book_quaternion_tcp_xyzw"] = tcp_book_payload[
        "quaternion_xyzw"
    ]

    report = {
        "schema_version": 1,
        "kind": "bookshelf_supervised_book_calibration_candidate",
        "candidate_id": candidate_id,
        "candidate_generated": True,
        "candidate_selected": False,
        "input_validation": {
            "accepted_unique_samples": int(
                held_book_report["accepted_unique_samples"]
            ),
            "required_stable_samples": int(
                held_book_report["required_stable_samples"]
            ),
            "live_candidate_stable": True,
            "sample_spread": deepcopy(held_book_report["sample_spread"]),
            "tolerances": deepcopy(held_book_report["tolerances"]),
        },
        "transforms": {
            "transform_eef_tcp": transform_to_dict(transform_eef_tcp),
            "transform_tcp_book_candidate": tcp_book_payload,
            "transform_eef_book_candidate": eef_book_payload,
            "transform_eef_policy_tool_candidate": eef_tool_payload,
            "transform_book_policy_tool_preserved": transform_to_dict(
                transform_book_policy_tool
            ),
        },
        "source_scene_difference": source_difference,
        "policy_tool_parity": {
            "passed": True,
            "translation_error_m": parity_translation_error_m,
            "rotation_error_deg": parity_rotation_error_deg,
            "round_trip_translation_error_m": derived[
                "round_trip_translation_error_m"
            ],
            "round_trip_rotation_error_deg": derived[
                "round_trip_rotation_error_deg"
            ],
        },
        "zero_delta_tcp_identity": {
            "passed": True,
            "translation_error_m": zero_delta_translation_error_m,
            "rotation_error_deg": zero_delta_rotation_error_deg,
        },
        "offline_preinsert_regression": offline_regression,
        "provenance": {
            "held_book_report_sha256": source_report_sha256,
            "scene_config_sha256": source_scene_sha256,
            "eef_tcp_context_sha256": source_context_sha256,
            "base_target_config_sha256": source_base_target_sha256,
            "held_book_report_generated_at": held_book_report.get(
                "generated_at"
            ),
            "held_book_report_scene_config": deepcopy(
                held_book_report.get("scene_config")
            ),
            "tcp_frame": held_book_report["tcp_frame"],
            "detected_book_frame": held_book_report["detected_book_frame"],
        },
        "review_holds": {
            "hardware_measurements_confirmed": False,
            "allow_local_insertion": False,
            "held_book_pose_check_required": True,
        },
        "safety": {
            "shadow_only": True,
            "plan_requested": False,
            "execution_authorized": False,
            "hardware_commanded": False,
            "active_configuration_modified": False,
            "human_approval_required": True,
        },
    }
    return target_candidate, scene_candidate, report


def _validate_held_book_report(report: dict) -> None:
    required = int(report["required_stable_samples"])
    accepted = int(report["accepted_unique_samples"])
    if accepted < required:
        raise ValueError(
            f"held-book report has only {accepted}/{required} stable samples"
        )
    if not bool(report.get("live_candidate_stable")):
        raise ValueError("held-book report does not contain a stable live candidate")
    if str(report.get("tcp_frame")) != "link_tcp":
        raise ValueError("held-book report must express the candidate in link_tcp")
    if not str(report.get("detected_book_frame", "")):
        raise ValueError("held-book report has no detected book frame")
    for key in (
        "hardware_commanded",
        "execution_authorized",
        "active_configuration_modified",
    ):
        if bool(report.get(key)):
            raise ValueError(f"held-book report is not read-only: {key}=true")

    spread = report["sample_spread"]
    tolerances = report["tolerances"]
    if float(spread["translation_m"]) > float(
        tolerances["maximum_sample_translation_spread_m"]
    ):
        raise ValueError("held-book translation spread exceeds its recorded limit")
    if float(spread["rotation_deg"]) > float(
        tolerances["maximum_sample_rotation_spread_deg"]
    ):
        raise ValueError("held-book rotation spread exceeds its recorded limit")
    _payload_transform(report["live_candidate_transform_tcp_book"])


def _ros_parameters(document: dict, node: str, label: str) -> dict:
    try:
        parameters = document[node]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} has no parameters for {node}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"{label} parameters for {node} must be a mapping")
    return parameters


def _payload_transform(payload: dict) -> np.ndarray:
    if "translation_xyz_m" in payload:
        translation = payload["translation_xyz_m"]
    else:
        translation = payload["translation_xyz"]
    return make_transform(translation, payload["quaternion_xyzw"])


def _transform_difference(left: np.ndarray, right: np.ndarray) -> dict:
    delta = invert_transform(left) @ right
    return {
        "translation_m": float(np.linalg.norm(left[:3, 3] - right[:3, 3])),
        "rotation_deg": _rotation_angle_deg(delta[:3, :3]),
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite existing candidate: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synchronized, unapproved target and MoveIt scene candidates "
            "from a stable held-book report."
        )
    )
    parser.add_argument("--held-book-report", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--eef-tcp-context", required=True)
    parser.add_argument("--base-target-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.held_book_report).expanduser().resolve()
    scene_path = Path(args.scene_config).expanduser().resolve()
    context_path = Path(args.eef_tcp_context).expanduser().resolve()
    base_target_path = Path(args.base_target_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target, scene, report = generate_supervised_candidate(
        _load_json(report_path),
        _load_yaml(scene_path),
        _load_json(context_path),
        _ros_parameters(
            _load_yaml(base_target_path),
            "calibrated_preinsert_target",
            "base target configuration",
        ),
        source_report_sha256=_sha256(report_path),
        source_scene_sha256=_sha256(scene_path),
        source_context_sha256=_sha256(context_path),
        source_base_target_sha256=_sha256(base_target_path),
    )
    target_path = output_dir / "supervised_book_calibration_candidate.yaml"
    scene_candidate_path = output_dir / "supervised_bookshelf_scene_candidate.yaml"
    candidate_report_path = output_dir / "supervised_book_calibration_report.json"
    for path in (target_path, scene_candidate_path, candidate_report_path):
        if path.exists():
            raise ValueError(f"refusing to overwrite existing candidate: {path}")
    _write_new(target_path, yaml.safe_dump(target, sort_keys=False))
    _write_new(scene_candidate_path, yaml.safe_dump(scene, sort_keys=False))
    report["outputs"] = {
        "target_candidate": str(target_path),
        "scene_candidate": str(scene_candidate_path),
    }
    _write_new(
        candidate_report_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )

    print(f"Target candidate: {target_path}")
    print(f"Scene candidate: {scene_candidate_path}")
    print(f"Report: {candidate_report_path}")
    print(
        "Input samples: "
        f"{report['input_validation']['accepted_unique_samples']}/"
        f"{report['input_validation']['required_stable_samples']} stable"
    )
    print("Policy-tool parity: PASS")
    print("Review holds: CLOSED")
    print("Execution authorized: False")
    print("Hardware commanded: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
