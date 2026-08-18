"""Fail-closed validation and calibration assembly for stationary A/B/C bags."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .calibrated_preinsert_target_math import transform_to_dict
from .marker_book_calibration import (
    average_quaternions_xyzw,
    quaternion_angle_deg,
)
from .policy_observation_math import invert_transform, make_transform
from .policy_tool_candidate_check import (
    SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
)
from .policy_tool_transform_extraction import derive_xarm_policy_tool_transform


RAW_REPLAY_TOPICS = (
    "/camera/color/image_raw",
    "/camera/aligned_depth_to_color/image_raw",
    "/camera/color/camera_info",
    "/tf",
    "/tf_static",
)

REQUIRED_CAPTURE_TOPICS = RAW_REPLAY_TOPICS + (
    "/joint_states",
    "/robot_description",
)


def summarize_fixed_transform(
    transforms,
    *,
    minimum_samples: int = 10,
    maximum_translation_spread_m: float = 0.0005,
    maximum_rotation_spread_deg: float = 0.25,
) -> dict:
    """Summarize repeated observations of one nominally fixed TF."""

    if int(minimum_samples) < 1:
        raise ValueError("minimum_samples must be positive")
    if min(
        float(maximum_translation_spread_m),
        float(maximum_rotation_spread_deg),
    ) <= 0.0:
        raise ValueError("fixed-transform spread tolerances must be positive")
    values = [np.asarray(value, dtype=np.float64) for value in transforms]
    if len(values) < int(minimum_samples):
        raise ValueError(
            f"need {minimum_samples} fixed-transform samples; got {len(values)}"
        )
    translations = []
    quaternions = []
    for value in values:
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("fixed-transform sample must be a finite 4x4 matrix")
        normalized = make_transform(
            value[:3, 3], transform_to_dict(value)["quaternion_xyzw"]
        )
        translations.append(normalized[:3, 3])
        quaternions.append(transform_to_dict(normalized)["quaternion_xyzw"])
    translations = np.asarray(translations, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    mean_translation = np.mean(translations, axis=0)
    mean_quaternion = average_quaternions_xyzw(quaternions)
    translation_spread = float(
        np.max(np.linalg.norm(translations - mean_translation[None, :], axis=1))
    )
    rotation_spread = float(
        max(quaternion_angle_deg(value, mean_quaternion) for value in quaternions)
    )
    if translation_spread > maximum_translation_spread_m:
        raise ValueError("fixed-transform translation spread exceeds tolerance")
    if rotation_spread > maximum_rotation_spread_deg:
        raise ValueError("fixed-transform rotation spread exceeds tolerance")
    return {
        "transform": make_transform(mean_translation, mean_quaternion),
        "sample_count": len(values),
        "translation_spread_m": translation_spread,
        "rotation_spread_deg": rotation_spread,
        "maximum_translation_spread_m": float(maximum_translation_spread_m),
        "maximum_rotation_spread_deg": float(maximum_rotation_spread_deg),
    }


def sha256_file(path) -> str:
    """Hash a file without loading a large rosbag into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_capture_run(
    run_directory,
    *,
    role: str,
    expected_condition: str,
    hash_bag_files: bool = True,
) -> dict:
    """Validate one logger run and return reproducible source provenance."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "manifest.json"
    bag_directory = run_directory / "rosbag"
    metadata_path = bag_directory / "metadata.yaml"
    manifest = _load_json(manifest_path)
    metadata = _load_yaml(metadata_path)
    information = metadata.get("rosbag2_bagfile_information")
    if not isinstance(information, dict):
        raise ValueError(f"{metadata_path} has no rosbag2 bag information")

    failures = []
    if manifest.get("schema_version") != 1:
        failures.append("unsupported logger manifest schema")
    if manifest.get("completed_cleanly") is not True:
        failures.append("capture did not complete cleanly")
    if manifest.get("raw_replay_inputs_recorded") is not True:
        failures.append("raw replay inputs were not recorded")
    if manifest.get("hardware_commanded_by_logger") is not False:
        failures.append("logger does not prove hardware_commanded=false")
    if manifest.get("capture_condition") != expected_condition:
        failures.append(
            "capture condition is "
            f"{manifest.get('capture_condition')!r}, expected {expected_condition!r}"
        )

    topic_counts = _topic_counts(information)
    for topic in REQUIRED_CAPTURE_TOPICS:
        if topic_counts.get(topic, 0) <= 0:
            failures.append(f"required replay topic has no messages: {topic}")
    if expected_condition == "book_attached":
        if topic_counts.get("/bookshelf_policy/book_boxes", 0) <= 0:
            failures.append("book-attached capture has no recorded marker detections")

    relative_paths = information.get("relative_file_paths")
    if not isinstance(relative_paths, list) or not relative_paths:
        failures.append("rosbag metadata contains no data files")
        relative_paths = []
    bag_files = []
    for relative_path in relative_paths:
        path = bag_directory / str(relative_path)
        if not path.is_file():
            failures.append(f"rosbag data file is missing: {path}")
            continue
        item = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        if hash_bag_files:
            item["sha256"] = sha256_file(path)
        bag_files.append(item)

    if failures:
        raise ValueError(f"invalid {role} capture: " + "; ".join(failures))

    duration_ns = int(information.get("duration", {}).get("nanoseconds", 0))
    return {
        "role": str(role),
        "run_directory": str(run_directory),
        "bag_directory": str(bag_directory),
        "condition": str(expected_condition),
        "duration_s": duration_ns * 1.0e-9,
        "message_count": int(information.get("message_count", 0)),
        "topic_counts": topic_counts,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "trial_name": manifest.get("trial_name"),
            "started_at": manifest.get("started_at"),
            "completed_at": manifest.get("completed_at"),
            "repository": manifest.get("repository"),
        },
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "bag_files": bag_files,
        "validated": True,
    }


def build_cross_view_slot_candidate(
    view_a_report: dict,
    view_b_report: dict,
    *,
    view_a_provenance: dict | None = None,
    view_b_provenance: dict | None = None,
    maximum_translation_disagreement_m: float = 0.010,
    maximum_rotation_disagreement_deg: float = 5.0,
    maximum_rotation_sanity_disagreement_deg: float = 15.0,
    maximum_width_disagreement_m: float = 0.005,
) -> dict:
    """Use frontal View A after an angled View B independently validates it."""

    if min(
        float(maximum_translation_disagreement_m),
        float(maximum_rotation_disagreement_deg),
        float(maximum_rotation_sanity_disagreement_deg),
        float(maximum_width_disagreement_m),
    ) <= 0.0:
        raise ValueError("cross-view disagreement tolerances must be positive")
    if (
        maximum_rotation_sanity_disagreement_deg
        < maximum_rotation_disagreement_deg
    ):
        raise ValueError(
            "rotation sanity tolerance must not be smaller than the "
            "diagnostic rotation tolerance"
        )
    candidate_a = _validated_slot_candidate(view_a_report, "view_a")
    candidate_b = _validated_slot_candidate(view_b_report, "view_b")
    base_frame_a = str(view_a_report.get("base_frame", ""))
    base_frame_b = str(view_b_report.get("base_frame", ""))

    translation_a = np.asarray(candidate_a["translation_xyz"], dtype=np.float64)
    translation_b = np.asarray(candidate_b["translation_xyz"], dtype=np.float64)
    quaternion_a = np.asarray(candidate_a["quaternion_xyzw"], dtype=np.float64)
    quaternion_b = np.asarray(candidate_b["quaternion_xyzw"], dtype=np.float64)
    width_a = float(candidate_a["width_m"])
    width_b = float(candidate_b["width_m"])

    disagreement = {
        "translation_m": float(np.linalg.norm(translation_a - translation_b)),
        "rotation_deg": quaternion_angle_deg(quaternion_a, quaternion_b),
        "width_m": abs(width_a - width_b),
    }
    tolerances = {
        "maximum_translation_disagreement_m": float(
            maximum_translation_disagreement_m
        ),
        "maximum_rotation_disagreement_deg": float(
            maximum_rotation_disagreement_deg
        ),
        "maximum_rotation_sanity_disagreement_deg": float(
            maximum_rotation_sanity_disagreement_deg
        ),
        "maximum_width_disagreement_m": float(maximum_width_disagreement_m),
    }
    failures = []
    warnings = []
    if not base_frame_a or base_frame_a != base_frame_b:
        failures.append(
            f"base frames do not match: {base_frame_a!r} versus {base_frame_b!r}"
        )
    if disagreement["translation_m"] > maximum_translation_disagreement_m:
        failures.append("cross-view translation disagreement exceeds tolerance")
    if disagreement["rotation_deg"] > maximum_rotation_sanity_disagreement_deg:
        failures.append(
            "cross-view rotation disagreement exceeds gross sanity tolerance"
        )
    elif disagreement["rotation_deg"] > maximum_rotation_disagreement_deg:
        warnings.append(
            "angled View B rotation exceeds the diagnostic tolerance; "
            "View A remains the sole pose source"
        )
    if disagreement["width_m"] > maximum_width_disagreement_m:
        failures.append("cross-view width disagreement exceeds tolerance")

    valid = not failures
    report = {
        "schema_version": 1,
        # Retain the promotable kind while explicitly recording the derivation.
        "kind": "bookshelf_static_slot_capture_candidate",
        "derivation": "view_a_primary_cross_view_validated",
        "generated_at": datetime.now().astimezone().isoformat(),
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "execution_authorized": False,
        "human_approval_required": True,
        "valid": valid,
        "reason": None if valid else "; ".join(failures),
        "base_frame": base_frame_a if base_frame_a == base_frame_b else None,
        "cross_view_disagreement": disagreement,
        "cross_view_validation": {
            "pose_source": "view_a",
            "validation_source": "view_b",
            "translation_used_for_acceptance": True,
            "width_used_for_acceptance": True,
            "rotation_used_for_acceptance": "gross_sanity_only",
            "rotation_within_diagnostic_tolerance": (
                disagreement["rotation_deg"]
                <= maximum_rotation_disagreement_deg
            ),
        },
        "tolerances": tolerances,
        "warnings": warnings,
        "views": {
            "view_a": _source_report_payload(
                view_a_report, view_a_provenance
            ),
            "view_b": _source_report_payload(
                view_b_report, view_b_provenance
            ),
        },
        "limitations": [
            "View A is the sole slot-pose source; View B validates identity, position, and width.",
            "Angled View B orientation is diagnostic except for the gross sanity gate.",
            "Cross-view validation is not absolute ground truth.",
            "A human must review both debug images and the View A marker in RViz.",
            "This candidate does not change active ROS or policy configuration.",
        ],
    }
    if valid:
        report["candidate"] = {
            "translation_xyz": translation_a.tolist(),
            "quaternion_xyzw": quaternion_a.tolist(),
            "width_m": width_a,
            "confidence": float(candidate_a["confidence"]),
            "transform_status": (
                "captured_rgbd_static_view_a_primary_cross_view_validated_unapproved"
            ),
        }
    return report


def build_stationary_calibration_bundle(
    slot_report: dict,
    book_report: dict,
    eef_tcp_context: dict,
    *,
    capture_provenance: dict,
    source_hashes: dict,
) -> tuple[dict, dict]:
    """Build synchronized, unapproved slot/book/tool calibration candidates."""

    if slot_report.get("derivation") != "view_a_primary_cross_view_validated":
        raise ValueError("slot candidate is not View A with cross-view validation")
    required_roles = ("view_a", "view_b", "book_attached")
    for role in required_roles:
        provenance = capture_provenance.get(role)
        if not isinstance(provenance, dict) or provenance.get("validated") is not True:
            raise ValueError(f"source capture provenance is not validated: {role}")
    slot_candidate = _validated_slot_candidate(slot_report, "cross_view_slot")
    book_result = _validated_book_result(book_report)
    transform_eef_tcp = _context_transform(eef_tcp_context)
    transform_eef_book = _result_transform(book_result)
    transform_tcp_book = invert_transform(transform_eef_tcp) @ transform_eef_book
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
    candidate_id = _bundle_candidate_id(source_hashes)
    book_status = f"measured_stationary_bag_candidate_{candidate_id}"
    tool_status = f"derived_unverified_stationary_bag_candidate_{candidate_id}"

    candidate_parameters = {
        "calibrated_preinsert_target": {
            "ros__parameters": {
                "eef_book_translation_xyz": transform_eef_book[:3, 3].tolist(),
                "eef_book_quaternion_xyzw": transform_to_dict(
                    transform_eef_book
                )["quaternion_xyzw"],
                "eef_book_transform_status": book_status,
                "eef_policy_tool_translation_xyz": (
                    transform_eef_policy_tool[:3, 3].tolist()
                ),
                "eef_policy_tool_quaternion_xyzw": transform_to_dict(
                    transform_eef_policy_tool
                )["quaternion_xyzw"],
                "policy_tool_transform_status": tool_status,
            }
        },
        "policy_observation_adapter": {
            "ros__parameters": {
                "book_pose_source": "eef_fixed",
                "latch_eef_book_from_marker": False,
                "use_configured_eef_book_transform": True,
                "eef_book_translation_xyz": transform_eef_book[:3, 3].tolist(),
                "eef_book_quaternion_xyzw": transform_to_dict(
                    transform_eef_book
                )["quaternion_xyzw"],
                "eef_book_transform_status": book_status,
                "tool_offset_xyz": transform_eef_policy_tool[:3, 3].tolist(),
                "tool_offset_quaternion_xyzw": transform_to_dict(
                    transform_eef_policy_tool
                )["quaternion_xyzw"],
                "policy_tool_transform_status": tool_status,
                "require_verified_policy_tool_transform": True,
            }
        },
        "bookshelf_scene_manager": {
            "ros__parameters": {
                "hardware_measurements_confirmed": False,
                "allow_local_insertion": False,
                "held_book_enabled": True,
                "require_held_book_pose_check": True,
                "held_book_center_tcp_xyz": transform_tcp_book[:3, 3].tolist(),
                "held_book_quaternion_tcp_xyzw": transform_to_dict(
                    transform_tcp_book
                )["quaternion_xyzw"],
            }
        },
    }
    bundle = {
        "schema_version": 1,
        "kind": "bookshelf_stationary_capture_calibration_bundle_candidate",
        "candidate_id": candidate_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "candidate_valid": True,
        "candidate_selected": False,
        "source_captures": capture_provenance,
        "source_hashes": dict(source_hashes),
        "slot": slot_candidate,
        "book_calibration": {
            "input_samples": int(book_result["input_samples"]),
            "inlier_samples": int(book_result["inlier_samples"]),
            "inlier_fraction": float(book_result["inlier_fraction"]),
            "transform_eef_book": transform_to_dict(transform_eef_book),
            "transform_eef_tcp": transform_to_dict(transform_eef_tcp),
            "transform_tcp_book": transform_to_dict(transform_tcp_book),
        },
        "policy_tool": {
            "transform_book_policy_tool": transform_to_dict(
                transform_book_policy_tool
            ),
            "transform_eef_policy_tool": transform_to_dict(
                transform_eef_policy_tool
            ),
            "round_trip_translation_error_m": float(
                derived["round_trip_translation_error_m"]
            ),
            "round_trip_rotation_error_deg": float(
                derived["round_trip_rotation_error_deg"]
            ),
            "verified": False,
        },
        "review_holds": {
            "cross_view_slot_human_review_required": True,
            "book_projection_human_review_required": True,
            "policy_tool_verification_required": True,
            "hardware_measurements_confirmed": False,
            "allow_local_insertion": False,
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
    return bundle, candidate_parameters


def _topic_counts(information: dict) -> dict:
    counts = {}
    values = information.get("topics_with_message_count", [])
    if not isinstance(values, list):
        return counts
    for value in values:
        if not isinstance(value, dict):
            continue
        metadata = value.get("topic_metadata", {})
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if name:
            counts[str(name)] = int(value.get("message_count", 0))
    return counts


def _validated_slot_candidate(report: dict, label: str) -> dict:
    if report.get("schema_version") != 1:
        raise ValueError(f"{label} has an unsupported schema")
    if report.get("kind") != "bookshelf_static_slot_capture_candidate":
        raise ValueError(f"{label} is not a static-slot candidate")
    if report.get("valid") is not True:
        raise ValueError(f"{label} is invalid: {report.get('reason')}")
    for key in ("hardware_commanded", "active_configuration_modified"):
        if report.get(key) is not False:
            raise ValueError(f"{label} does not prove {key}=false")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{label} contains no candidate pose")
    transform = make_transform(
        candidate.get("translation_xyz"), candidate.get("quaternion_xyzw")
    )
    width = float(candidate.get("width_m", 0.0))
    confidence = float(candidate.get("confidence", -1.0))
    if width <= 0.0 or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{label} has an invalid width or confidence")
    validated = dict(candidate)
    validated["translation_xyz"] = transform[:3, 3].tolist()
    validated["quaternion_xyzw"] = transform_to_dict(transform)[
        "quaternion_xyzw"
    ]
    validated["width_m"] = width
    validated["confidence"] = confidence
    return validated


def _validated_book_result(report: dict) -> dict:
    if report.get("schema_version") != 1:
        raise ValueError("book calibration has an unsupported schema")
    if report.get("calibration_valid") is not True:
        raise ValueError("book calibration is not valid")
    if report.get("hardware_commanded") is not False:
        raise ValueError("book calibration does not prove hardware_commanded=false")
    if report.get("read_only") is not True:
        raise ValueError("book calibration is not marked read-only")
    frame_convention = report.get("frame_convention")
    if not isinstance(frame_convention, dict) or frame_convention.get(
        "transform_output"
    ) != "T_eef_book (book pose expressed in link_eef)":
        raise ValueError("book calibration does not declare T_link_eef_book output")
    result = report.get("result")
    if not isinstance(result, dict):
        raise ValueError("book calibration contains no result")
    _result_transform(result)
    if int(result.get("inlier_samples", 0)) < int(
        report.get("minimum_inlier_samples", 1)
    ):
        raise ValueError("book calibration has too few inliers")
    if float(result.get("inlier_fraction", 0.0)) < float(
        report.get("minimum_inlier_fraction", 1.0)
    ):
        raise ValueError("book calibration inlier fraction is too low")
    return result


def _context_transform(context: dict) -> np.ndarray:
    if context.get("schema_version") != 1 or context.get("valid") is not True:
        raise ValueError("EEF/TCP context is not valid")
    if context.get("hardware_commanded") is not False:
        raise ValueError("EEF/TCP context does not prove hardware_commanded=false")
    if context.get("parent_frame") != "link_eef":
        raise ValueError("EEF/TCP context parent must be link_eef")
    if context.get("child_frame") != "link_tcp":
        raise ValueError("EEF/TCP context child must be link_tcp")
    payload = context.get("transform_eef_tcp")
    if not isinstance(payload, dict):
        raise ValueError("EEF/TCP context contains no transform")
    return make_transform(
        payload.get("translation_xyz_m"), payload.get("quaternion_xyzw")
    )


def _result_transform(result: dict) -> np.ndarray:
    matrix = result.get("transform_eef_book")
    if matrix is not None:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("book calibration transform matrix is invalid")
        return make_transform(value[:3, 3], transform_to_dict(value)["quaternion_xyzw"])
    return make_transform(
        result.get("translation_xyz_m"), result.get("quaternion_xyzw")
    )


def _source_report_payload(report: dict, provenance: dict | None) -> dict:
    return {
        "candidate": dict(report["candidate"]),
        "statistics": report.get("statistics"),
        "counters": report.get("counters"),
        "report": provenance,
    }


def _bundle_candidate_id(source_hashes: dict) -> str:
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required YAML file is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value
