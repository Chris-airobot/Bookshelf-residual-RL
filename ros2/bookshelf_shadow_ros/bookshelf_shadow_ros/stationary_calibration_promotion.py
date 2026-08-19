"""Atomically promote one reviewed stationary A/B/C calibration bundle."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import yaml

from .calibrated_preinsert_target_math import transform_to_dict
from .policy_observation_math import invert_transform, make_transform
from .policy_tool_candidate_check import (
    SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
)
from .static_slot_capture import APPROVAL_TOKEN, dump_ros_parameter_yaml
from .stationary_capture_bundle import sha256_file


BOOK_APPROVAL_TOKEN = "VISUALLY_APPROVED_BOOK_PROJECTION"
POLICY_TOOL_APPROVAL_TOKEN = "VERIFIED_POLICY_TOOL_FRAME"
SCENE_APPROVAL_TOKEN = "REVIEWED_PHYSICAL_SCENE"

TRIAL_CONFIG_NAME = "trial_static_slot.yaml"
PROVENANCE_NAME = "trial_static_slot.provenance.json"


def promote_stationary_calibration_bundle(
    bundle_path,
    candidate_config_path,
    template_directory,
    scene_template_path,
    output_directory,
    *,
    shadow_replay_pipeline_report_path,
    reviewer: str,
    slot_approval_token: str,
    book_approval_token: str,
    policy_tool_approval_token: str,
    scene_approval_token: str,
) -> dict:
    """Write one synchronized, reviewed trial config without launching ROS."""

    _validate_approvals(
        reviewer=reviewer,
        slot_approval_token=slot_approval_token,
        book_approval_token=book_approval_token,
        policy_tool_approval_token=policy_tool_approval_token,
        scene_approval_token=scene_approval_token,
    )
    bundle_path = _required_file(bundle_path, "calibration bundle")
    candidate_config_path = _required_file(
        candidate_config_path, "unapproved parameter candidate"
    )
    template_directory = Path(template_directory).expanduser().resolve()
    if not template_directory.is_dir():
        raise ValueError(f"template directory is missing: {template_directory}")
    scene_template_path = _required_file(scene_template_path, "scene template")
    output_directory = Path(output_directory).expanduser().resolve()
    if output_directory.exists():
        raise ValueError(
            "output directory already exists; refusing to overwrite: "
            f"{output_directory}"
        )

    bundle = _load_json(bundle_path)
    candidate_document = _load_yaml(candidate_config_path)
    source_checks = _validate_bundle_and_sources(
        bundle,
        bundle_path=bundle_path,
        candidate_config_path=candidate_config_path,
    )
    shadow_evidence, shadow_checks = _validate_shadow_replay_evidence(
        bundle,
        candidate_document,
        bundle_path=bundle_path,
        candidate_config_path=candidate_config_path,
        pipeline_report_path=shadow_replay_pipeline_report_path,
    )
    geometry_checks = _validate_candidate_geometry(
        bundle,
        candidate_document,
        allow_legacy_fixed_source=bool(
            shadow_checks["continuous_marker_tracking_verified"]
        ),
    )
    approved_document, statuses = _build_approved_document(
        bundle,
        candidate_document,
        template_directory=template_directory,
        scene_template_path=scene_template_path,
    )

    config_text = (
        "# Generated from one reviewed stationary A/B/C calibration bundle.\n"
        "# Do not edit slot, book, scene, or policy-tool values independently.\n"
        + dump_ros_parameter_yaml(approved_document)
    )
    config_sha256 = _sha256_bytes(config_text.encode("utf-8"))
    final_config = output_directory / TRIAL_CONFIG_NAME
    final_provenance = output_directory / PROVENANCE_NAME
    cross_view_path = Path(
        bundle["outputs"]["cross_view_slot_candidate"]
    ).expanduser().resolve()
    provenance = {
        "schema_version": 1,
        "kind": "bookshelf_stationary_calibration_trial_configuration",
        "generated_at": datetime.now().astimezone().isoformat(),
        "candidate_id": str(bundle["candidate_id"]),
        "reviewer": reviewer.strip(),
        "human_approval_recorded": True,
        # Retain compatibility with physical_experiment_preflight.
        "approval_token": APPROVAL_TOKEN,
        "approval_tokens": {
            "cross_view_slot": APPROVAL_TOKEN,
            "book_projection": BOOK_APPROVAL_TOKEN,
            "policy_tool_frame": POLICY_TOOL_APPROVAL_TOKEN,
            "physical_scene": SCENE_APPROVAL_TOKEN,
        },
        "hardware_commanded": False,
        "execution_authorized": False,
        "allow_local_insertion": False,
        "candidate_report": str(cross_view_path),
        "candidate_report_sha256": sha256_file(cross_view_path),
        "bundle_candidate": str(bundle_path),
        "bundle_candidate_sha256": sha256_file(bundle_path),
        "unapproved_parameter_candidate": str(candidate_config_path),
        "unapproved_parameter_candidate_sha256": sha256_file(
            candidate_config_path
        ),
        "shadow_replay_evidence": shadow_evidence,
        "scene_template": str(scene_template_path),
        "scene_template_sha256": sha256_file(scene_template_path),
        "trial_config": str(final_config),
        "trial_config_sha256": config_sha256,
        "source_hashes": copy.deepcopy(bundle["source_hashes"]),
        "validation": {
            **source_checks,
            **shadow_checks,
            **geometry_checks,
            "all_checks_passed": True,
        },
        "transform_status": statuses["slot"],
        "transform_statuses": statuses,
        "runtime_source_migration": {
            "candidate_book_pose_source": str(
                _parameters(
                    candidate_document, "policy_observation_adapter"
                ).get("book_pose_source")
            ),
            "promoted_book_pose_source": "marker",
            "authorized_by_shadow_replay": True,
        },
        "slot": copy.deepcopy(bundle["slot"]),
        "safety": {
            "offline_only": True,
            "plan_requested": False,
            "hardware_commanded": False,
            "execution_authorized": False,
            "allow_local_insertion": False,
        },
    }
    provenance_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    _write_new_directory_atomically(
        output_directory,
        {
            TRIAL_CONFIG_NAME: config_text,
            PROVENANCE_NAME: provenance_text,
        },
    )
    return provenance


def _validate_approvals(
    *,
    reviewer: str,
    slot_approval_token: str,
    book_approval_token: str,
    policy_tool_approval_token: str,
    scene_approval_token: str,
) -> None:
    if not str(reviewer).strip():
        raise ValueError("reviewer must be a non-empty name or identifier")
    expected = {
        "slot approval": (slot_approval_token, APPROVAL_TOKEN),
        "book projection approval": (
            book_approval_token,
            BOOK_APPROVAL_TOKEN,
        ),
        "policy-tool approval": (
            policy_tool_approval_token,
            POLICY_TOOL_APPROVAL_TOKEN,
        ),
        "physical-scene approval": (
            scene_approval_token,
            SCENE_APPROVAL_TOKEN,
        ),
    }
    for label, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(f"{label} requires token {required}")


def _validate_bundle_and_sources(
    bundle: dict,
    *,
    bundle_path: Path,
    candidate_config_path: Path,
) -> dict:
    if bundle.get("schema_version") != 1:
        raise ValueError("unsupported stationary calibration bundle schema")
    if bundle.get("kind") != (
        "bookshelf_stationary_capture_calibration_bundle_candidate"
    ):
        raise ValueError("input is not a stationary calibration bundle candidate")
    if bundle.get("candidate_valid") is not True:
        raise ValueError("stationary calibration candidate is not valid")
    if bundle.get("candidate_selected") is not False:
        raise ValueError("stationary calibration candidate was already selected")
    safety = bundle.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("bundle has no safety declaration")
    required_safety = {
        "shadow_only": True,
        "plan_requested": False,
        "execution_authorized": False,
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "human_approval_required": True,
    }
    wrong = {
        key: safety.get(key)
        for key, expected in required_safety.items()
        if safety.get(key) is not expected
    }
    if wrong:
        raise ValueError(f"bundle safety declaration is not fail-closed: {wrong}")
    policy_tool = bundle.get("policy_tool")
    if not isinstance(policy_tool, dict) or policy_tool.get("verified") is not False:
        raise ValueError("input policy-tool transform must be an unverified candidate")

    source_hashes = bundle.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("bundle contains no source hashes")
    expected_candidate_id = _candidate_id(source_hashes)
    if bundle.get("candidate_id") != expected_candidate_id:
        raise ValueError("bundle candidate ID does not match its source hashes")

    cross_view_path = _required_file(
        bundle["outputs"]["cross_view_slot_candidate"],
        "cross-view slot report",
    )
    cross_view = _load_json(cross_view_path)
    paths = {
        "view_a_report_sha256": cross_view["views"]["view_a"]["report"]["path"],
        "view_b_report_sha256": cross_view["views"]["view_b"]["report"]["path"],
        "cross_view_slot_report_sha256": cross_view_path,
        "book_report_sha256": bundle["outputs"]["book_calibration_summary"],
        "eef_tcp_context_sha256": bundle["outputs"]["eef_tcp_context"],
    }
    if set(source_hashes) != set(paths):
        raise ValueError("bundle source hash keys do not match required source reports")
    for key, value in paths.items():
        path = _required_file(value, key)
        if sha256_file(path) != source_hashes[key]:
            raise ValueError(f"source hash mismatch: {key}")

    output_hashes = bundle.get("output_hashes")
    expected_hash = (
        output_hashes.get("unapproved_parameter_candidate_sha256")
        if isinstance(output_hashes, dict)
        else None
    )
    if not expected_hash or sha256_file(candidate_config_path) != expected_hash:
        raise ValueError("unapproved parameter candidate hash mismatch")
    declared_candidate = Path(
        bundle["outputs"]["unapproved_parameter_candidate"]
    ).expanduser().resolve()
    if declared_candidate != candidate_config_path:
        raise ValueError("candidate config path differs from the bundle declaration")

    return {
        "bundle_hash_verified": len(sha256_file(bundle_path)) == 64,
        "source_hashes_verified": True,
        "candidate_config_hash_verified": True,
        "candidate_id_verified": True,
    }


def _validate_shadow_replay_evidence(
    bundle: dict,
    candidate: dict,
    *,
    bundle_path: Path,
    candidate_config_path: Path,
    pipeline_report_path,
) -> tuple[dict, dict]:
    pipeline_report_path = _required_file(
        pipeline_report_path, "stationary shadow replay pipeline report"
    )
    pipeline = _load_json(pipeline_report_path)
    if pipeline.get("schema_version") != 1 or pipeline.get("kind") != (
        "bookshelf_stationary_shadow_replay_pipeline"
    ):
        raise ValueError("input is not a stationary shadow replay pipeline report")
    if pipeline.get("passed") is not True:
        raise ValueError("stationary shadow replay pipeline did not pass")
    if str(pipeline.get("candidate_id")) != str(bundle["candidate_id"]):
        raise ValueError("shadow replay candidate ID does not match the bundle")

    calibration = pipeline.get("calibration_bundle")
    if not isinstance(calibration, dict):
        raise ValueError("shadow replay has no calibration-bundle provenance")
    declared_bundle_path = Path(
        str(calibration.get("path", ""))
    ).expanduser().resolve()
    if declared_bundle_path != bundle_path:
        raise ValueError("shadow replay references a different calibration bundle")
    if calibration.get("sha256") != sha256_file(bundle_path):
        raise ValueError("shadow replay calibration-bundle hash mismatch")

    calibration_source = pipeline.get("calibration_source")
    if not isinstance(calibration_source, dict):
        raise ValueError("shadow replay has no calibration-source provenance")
    declared_candidate_path = Path(
        str(calibration_source.get("candidate_path", ""))
    ).expanduser().resolve()
    if declared_candidate_path != candidate_config_path:
        raise ValueError("shadow replay references a different candidate YAML")
    if calibration_source.get("candidate_sha256") != sha256_file(
        candidate_config_path
    ):
        raise ValueError("shadow replay candidate-YAML hash mismatch")

    runtime = pipeline.get("runtime_adapter")
    if not isinstance(runtime, dict):
        raise ValueError("shadow replay has no runtime-adapter provenance")
    if runtime.get("book_pose_source") != "marker":
        raise ValueError("shadow replay did not require marker book poses")
    if runtime.get("slot_pose_source") != "configured_static":
        raise ValueError("shadow replay did not freeze the configured slot")
    if runtime.get("policy_tool_candidate_used_for_diagnostics_only") is not True:
        raise ValueError("shadow replay policy-tool candidate was not diagnostic only")
    runtime_path = _required_file(runtime.get("path"), "shadow runtime adapter")
    if runtime.get("sha256") != sha256_file(runtime_path):
        raise ValueError("shadow runtime-adapter hash mismatch")

    observation_entry = pipeline.get("observation_audit")
    if not isinstance(observation_entry, dict) or observation_entry.get(
        "passed"
    ) is not True:
        raise ValueError("shadow observation audit did not pass")
    observation_path = _required_file(
        observation_entry.get("path"), "shadow observation audit"
    )
    if observation_entry.get("sha256") != sha256_file(observation_path):
        raise ValueError("shadow observation-audit hash mismatch")
    observation = _load_json(observation_path)
    if observation.get("schema_version") != 1 or observation.get("kind") != (
        "bookshelf_stationary_shadow_replay_audit"
    ):
        raise ValueError("shadow observation evidence has an unexpected schema")
    if observation.get("passed") is not True:
        raise ValueError("shadow observation evidence did not pass")
    if str(observation.get("candidate_id")) != str(bundle["candidate_id"]):
        raise ValueError("shadow observation candidate ID does not match")
    summary = observation.get("observation_pipeline")
    if not isinstance(summary, dict):
        raise ValueError("shadow observation report has no pipeline summary")
    valid_samples = int(summary.get("valid_samples", 0))
    minimum_valid_samples = max(int(summary.get("minimum_valid_samples", 0)), 30)
    if valid_samples < minimum_valid_samples:
        raise ValueError("shadow replay has insufficient valid marker observations")
    if summary.get("book_pose_sources") != {"marker": valid_samples}:
        raise ValueError("shadow replay did not use marker poses for every book sample")
    if summary.get("slot_pose_sources") != {"configured_static": valid_samples}:
        raise ValueError("shadow replay did not use one frozen slot for every sample")
    if summary.get("failure_reasons") not in ([], None):
        raise ValueError("shadow observation report contains failure reasons")
    policy = summary.get("policy_diagnostics")
    if not isinstance(policy, dict) or int(policy.get("messages", 0)) < 1:
        raise ValueError("shadow replay received no policy diagnostics")
    if int(policy.get("adapter_policy_observation_mismatches", -1)) != 0:
        raise ValueError("shadow adapter and policy observations did not match")
    if not isinstance(policy.get("normalized_observation"), dict):
        raise ValueError("shadow policy never normalized the real observations")

    marker_entry = pipeline.get("marker_detection")
    if not isinstance(marker_entry, dict) or marker_entry.get(
        "calibration_valid"
    ) is not True:
        raise ValueError("shadow marker calibration did not pass")
    marker_path = _required_file(
        marker_entry.get("path"), "shadow marker calibration report"
    )
    if marker_entry.get("sha256") != sha256_file(marker_path):
        raise ValueError("shadow marker-calibration hash mismatch")
    marker = _load_json(marker_path)
    if marker.get("calibration_valid") is not True:
        raise ValueError("shadow marker report is not valid")
    if marker.get("read_only") is not True or marker.get("hardware_commanded") is not False:
        raise ValueError("shadow marker report is not read-only and fail-closed")

    expected_safety = {
        "shadow_only": True,
        "plan_requested": False,
        "execution_authorized": False,
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "candidate_selected": False,
    }
    for label, payload in (
        ("pipeline", pipeline.get("safety")),
        ("observation", observation.get("safety")),
    ):
        if not isinstance(payload, dict) or any(
            payload.get(key) is not expected
            for key, expected in expected_safety.items()
        ):
            raise ValueError(f"shadow {label} safety declaration is not fail-closed")

    candidate_source = str(
        _parameters(candidate, "policy_observation_adapter").get(
            "book_pose_source", ""
        )
    )
    evidence = {
        "pipeline_report": str(pipeline_report_path),
        "pipeline_report_sha256": sha256_file(pipeline_report_path),
        "observation_report": str(observation_path),
        "observation_report_sha256": sha256_file(observation_path),
        "marker_report": str(marker_path),
        "marker_report_sha256": sha256_file(marker_path),
        "runtime_adapter": str(runtime_path),
        "runtime_adapter_sha256": sha256_file(runtime_path),
        "valid_marker_observations": valid_samples,
    }
    checks = {
        "shadow_replay_pipeline_verified": True,
        "shadow_replay_candidate_id_verified": True,
        "shadow_replay_bundle_hash_verified": True,
        "shadow_replay_candidate_hash_verified": True,
        "continuous_marker_tracking_verified": True,
        "frozen_slot_source_verified": True,
        "adapter_policy_observation_parity_verified": True,
        "shadow_safety_verified": True,
        "legacy_fixed_source_migration_required": candidate_source == "eef_fixed",
    }
    return evidence, checks


def _validate_candidate_geometry(
    bundle: dict,
    candidate: dict,
    *,
    allow_legacy_fixed_source: bool = False,
) -> dict:
    target = _parameters(candidate, "calibrated_preinsert_target")
    adapter = _parameters(candidate, "policy_observation_adapter")
    scene = _parameters(candidate, "bookshelf_scene_manager")

    transform_eef_book = _payload_transform(
        bundle["book_calibration"]["transform_eef_book"]
    )
    transform_eef_tcp = _payload_transform(
        bundle["book_calibration"]["transform_eef_tcp"]
    )
    transform_tcp_book = _payload_transform(
        bundle["book_calibration"]["transform_tcp_book"]
    )
    transform_eef_policy_tool = _payload_transform(
        bundle["policy_tool"]["transform_eef_policy_tool"]
    )
    transform_book_policy_tool = _payload_transform(
        bundle["policy_tool"]["transform_book_policy_tool"]
    )

    _assert_parameter_transform(
        target,
        "eef_book_translation_xyz",
        "eef_book_quaternion_xyzw",
        transform_eef_book,
    )
    _assert_parameter_transform(
        target,
        "eef_policy_tool_translation_xyz",
        "eef_policy_tool_quaternion_xyzw",
        transform_eef_policy_tool,
    )
    _assert_parameter_transform(
        adapter,
        "eef_book_translation_xyz",
        "eef_book_quaternion_xyzw",
        transform_eef_book,
    )
    _assert_parameter_transform(
        adapter,
        "tool_offset_xyz",
        "tool_offset_quaternion_xyzw",
        transform_eef_policy_tool,
    )
    _assert_parameter_transform(
        scene,
        "held_book_center_tcp_xyz",
        "held_book_quaternion_tcp_xyzw",
        transform_tcp_book,
    )
    if adapter.get("require_verified_policy_tool_transform") is not True:
        raise ValueError("candidate adapter does not require a verified policy tool")
    book_pose_source = adapter.get("book_pose_source")
    if book_pose_source != "marker" and not (
        book_pose_source == "eef_fixed" and allow_legacy_fixed_source
    ):
        raise ValueError("candidate adapter must require live marker book poses")
    if adapter.get("latch_eef_book_from_marker") is not False:
        raise ValueError("candidate adapter must not latch one marker observation")
    if adapter.get("use_configured_eef_book_transform") is not True:
        raise ValueError("candidate adapter must retain the fixed grasp reference")
    if scene.get("hardware_measurements_confirmed") is not False:
        raise ValueError("candidate scene unexpectedly opened its measurement hold")
    if scene.get("allow_local_insertion") is not False:
        raise ValueError("candidate scene unexpectedly permits local insertion")

    candidate_id = str(bundle["candidate_id"])
    target_book_status = str(target.get("eef_book_transform_status", ""))
    adapter_book_status = str(adapter.get("eef_book_transform_status", ""))
    target_tool_status = str(target.get("policy_tool_transform_status", ""))
    adapter_tool_status = str(adapter.get("policy_tool_transform_status", ""))
    if (
        target_book_status != adapter_book_status
        or "candidate" not in target_book_status
        or candidate_id not in target_book_status
    ):
        raise ValueError("book transform status is inconsistent or not this candidate")
    if (
        target_tool_status != adapter_tool_status
        or not target_tool_status.startswith("derived_unverified_")
        or candidate_id not in target_tool_status
    ):
        raise ValueError(
            "policy-tool status is inconsistent or not this unverified candidate"
        )

    expected_book_tool = make_transform(
        SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
        SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    )
    parity_error = invert_transform(expected_book_tool) @ transform_book_policy_tool
    parity_translation_m, parity_rotation_deg = _transform_error(parity_error)
    if parity_translation_m > 1.0e-9 or parity_rotation_deg > 1.0e-6:
        raise ValueError("policy-tool transform does not preserve simulator semantics")

    book_composition_error = (
        invert_transform(transform_eef_book)
        @ transform_eef_tcp
        @ transform_tcp_book
    )
    book_translation_m, book_rotation_deg = _transform_error(
        book_composition_error
    )
    if book_translation_m > 1.0e-9 or book_rotation_deg > 1.0e-6:
        raise ValueError("EEF/TCP/book transforms do not compose consistently")

    transform_tcp_policy_tool = (
        invert_transform(transform_eef_tcp) @ transform_eef_policy_tool
    )
    transform_base_tcp = make_transform(
        [0.52, -0.11, 0.37], [0.11, -0.07, 0.19, 0.972]
    )
    transform_base_policy_tool = (
        transform_base_tcp @ transform_tcp_policy_tool
    )
    reconstructed_tcp = (
        transform_base_policy_tool @ invert_transform(transform_tcp_policy_tool)
    )
    zero_delta_error = invert_transform(transform_base_tcp) @ reconstructed_tcp
    zero_translation_m, zero_rotation_deg = _transform_error(zero_delta_error)
    # acos(trace(R)) resolves identity rotations at roughly micro-degree
    # precision after several otherwise exact matrix multiplications.
    if zero_translation_m > 1.0e-12 or zero_rotation_deg > 1.0e-5:
        raise ValueError("zero policy displacement does not reproduce current TCP")

    return {
        "candidate_parameters_match_bundle": True,
        "candidate_book_pose_source": str(book_pose_source),
        "promoted_book_pose_source": "marker",
        "simulator_policy_tool_parity": {
            "passed": True,
            "translation_error_m": parity_translation_m,
            "rotation_error_deg": parity_rotation_deg,
        },
        "eef_tcp_book_composition": {
            "passed": True,
            "translation_error_m": book_translation_m,
            "rotation_error_deg": book_rotation_deg,
        },
        "zero_delta_tcp_identity": {
            "passed": True,
            "translation_error_m": zero_translation_m,
            "rotation_error_deg": zero_rotation_deg,
        },
    }


def _build_approved_document(
    bundle: dict,
    candidate: dict,
    *,
    template_directory: Path,
    scene_template_path: Path,
) -> tuple[dict, dict]:
    templates = {
        "static_slot_environment_check": "static_slot_environment_check.yaml",
        "calibrated_preinsert_target": "calibrated_preinsert_target.yaml",
        "policy_observation_adapter": (
            "policy_observation_adapter_policy_tool_candidate.yaml"
        ),
    }
    combined = {}
    for node_name, filename in templates.items():
        document = _load_yaml(template_directory / filename)
        combined[node_name] = copy.deepcopy(document[node_name])
    scene_document = _load_yaml(scene_template_path)
    combined["bookshelf_scene_manager"] = copy.deepcopy(
        scene_document["bookshelf_scene_manager"]
    )

    candidate_id = str(bundle["candidate_id"])
    statuses = {
        "slot": f"captured_rgbd_static_human_approved_{candidate_id}",
        "book": f"measured_stationary_bag_human_approved_{candidate_id}",
        "policy_tool": f"verified_stationary_bag_policy_tool_{candidate_id}",
    }
    slot = bundle["slot"]
    check = _parameters(combined, "static_slot_environment_check")
    check.update(
        {
            "static_slot_translation_xyz": list(slot["translation_xyz"]),
            "static_slot_quaternion_xyzw": list(slot["quaternion_xyzw"]),
            "static_slot_width_m": float(slot["width_m"]),
            "static_slot_transform_status": statuses["slot"],
        }
    )
    target = _parameters(combined, "calibrated_preinsert_target")
    target.update(
        {
            "static_slot_translation_xyz": list(slot["translation_xyz"]),
            "static_slot_quaternion_xyzw": list(slot["quaternion_xyzw"]),
            "static_slot_width_m": float(slot["width_m"]),
            "static_slot_confidence": float(slot["confidence"]),
            "static_slot_transform_status": statuses["slot"],
        }
    )
    adapter = _parameters(combined, "policy_observation_adapter")
    adapter.update(
        {
            "configured_static_slot_translation_xyz": list(
                slot["translation_xyz"]
            ),
            "configured_static_slot_quaternion_xyzw": list(
                slot["quaternion_xyzw"]
            ),
            "configured_static_slot_width_m": float(slot["width_m"]),
            "configured_static_slot_confidence": float(slot["confidence"]),
            "static_slot_transform_status": statuses["slot"],
        }
    )

    candidate_target = _parameters(candidate, "calibrated_preinsert_target")
    target.update(
        {
            "eef_book_translation_xyz": list(
                candidate_target["eef_book_translation_xyz"]
            ),
            "eef_book_quaternion_xyzw": list(
                candidate_target["eef_book_quaternion_xyzw"]
            ),
            "eef_book_transform_status": statuses["book"],
            "eef_policy_tool_translation_xyz": list(
                candidate_target["eef_policy_tool_translation_xyz"]
            ),
            "eef_policy_tool_quaternion_xyzw": list(
                candidate_target["eef_policy_tool_quaternion_xyzw"]
            ),
            "policy_tool_transform_status": statuses["policy_tool"],
        }
    )
    candidate_adapter = _parameters(candidate, "policy_observation_adapter")
    adapter.update(
        {
            "book_pose_source": "marker",
            "latch_eef_book_from_marker": False,
            "use_configured_eef_book_transform": True,
            "eef_book_translation_xyz": list(
                candidate_adapter["eef_book_translation_xyz"]
            ),
            "eef_book_quaternion_xyzw": list(
                candidate_adapter["eef_book_quaternion_xyzw"]
            ),
            "eef_book_transform_status": statuses["book"],
            "tool_offset_xyz": list(candidate_adapter["tool_offset_xyz"]),
            "tool_offset_quaternion_xyzw": list(
                candidate_adapter["tool_offset_quaternion_xyzw"]
            ),
            "policy_tool_transform_status": statuses["policy_tool"],
            "require_verified_policy_tool_transform": True,
        }
    )
    scene = _parameters(combined, "bookshelf_scene_manager")
    candidate_scene = _parameters(candidate, "bookshelf_scene_manager")
    scene.update(
        {
            "hardware_measurements_confirmed": True,
            "allow_local_insertion": False,
            "held_book_enabled": True,
            "require_held_book_pose_check": True,
            "held_book_center_tcp_xyz": list(
                candidate_scene["held_book_center_tcp_xyz"]
            ),
            "held_book_quaternion_tcp_xyzw": list(
                candidate_scene["held_book_quaternion_tcp_xyzw"]
            ),
        }
    )
    return combined, statuses


def _parameters(document: dict, node_name: str) -> dict:
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing ROS parameters for {node_name}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"invalid ROS parameters for {node_name}")
    return parameters


def _assert_parameter_transform(
    parameters: dict,
    translation_key: str,
    quaternion_key: str,
    expected: np.ndarray,
) -> None:
    actual = make_transform(
        parameters.get(translation_key), parameters.get(quaternion_key)
    )
    error = invert_transform(expected) @ actual
    translation_m, rotation_deg = _transform_error(error)
    if translation_m > 1.0e-12 or rotation_deg > 1.0e-6:
        raise ValueError(
            f"candidate {translation_key}/{quaternion_key} differs from bundle"
        )


def _payload_transform(payload: dict) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("bundle transform payload must be a mapping")
    translation = payload.get("translation_xyz_m", payload.get("translation_xyz"))
    return make_transform(translation, payload.get("quaternion_xyzw"))


def _transform_error(transform: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(transform[:3, 3]))
    cosine = float(np.clip((np.trace(transform[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return translation_m, math.degrees(math.acos(cosine))


def _candidate_id(source_hashes: dict) -> str:
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_file(path, label: str) -> Path:
    if path is None or not str(path).strip():
        raise ValueError(f"{label} path is missing")
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ValueError(f"{label} is missing: {value}")
    return value


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict:
    path = _required_file(path, "YAML input")
    text = path.read_text(encoding="utf-8")
    if "&id" in text or "*id" in text:
        raise ValueError(f"YAML aliases are not permitted: {path}")
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _write_new_directory_atomically(
    output_directory: Path, files: dict[str, str]
) -> None:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.parent / (
        f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        for name, text in files.items():
            (staging / name).write_text(text, encoding="utf-8")
        os.replace(staging, output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _default_paths() -> tuple[Path, Path]:
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError as error:
        raise ValueError(
            "ament_index_python is unavailable; pass --template-dir and "
            "--scene-template explicitly"
        ) from error
    shadow_share = Path(get_package_share_directory("bookshelf_shadow_ros"))
    guarded_share = Path(
        get_package_share_directory("bookshelf_guarded_control_ros")
    )
    return (
        shadow_share / "config",
        guarded_share / "config" / "bookshelf_scene_physical.yaml",
    )


def main(args=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote one reviewed stationary A/B/C calibration bundle into a "
            "single synchronized ROS parameter file. This command is offline "
            "and creates no ROS, planning, controller, or execution interface."
        )
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--candidate-config")
    parser.add_argument(
        "--shadow-replay-pipeline-report",
        required=True,
        help=(
            "Passing stationary_shadow_replay_pipeline_report.json bound to "
            "the same calibration bundle and candidate YAML."
        ),
    )
    parser.add_argument("--template-dir")
    parser.add_argument("--scene-template")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--slot-approval-token", required=True)
    parser.add_argument("--book-approval-token", required=True)
    parser.add_argument("--policy-tool-approval-token", required=True)
    parser.add_argument("--scene-approval-token", required=True)
    parsed = parser.parse_args(args)
    try:
        bundle_path = _required_file(parsed.bundle, "calibration bundle")
        bundle = _load_json(bundle_path)
        candidate_config = parsed.candidate_config or bundle.get("outputs", {}).get(
            "unapproved_parameter_candidate"
        )
        if not candidate_config:
            raise ValueError(
                "bundle does not declare an unapproved parameter candidate"
            )
        if parsed.template_dir and parsed.scene_template:
            template_directory = Path(parsed.template_dir)
            scene_template = Path(parsed.scene_template)
        elif parsed.template_dir or parsed.scene_template:
            raise ValueError(
                "pass both --template-dir and --scene-template, or neither"
            )
        else:
            template_directory, scene_template = _default_paths()
        provenance = promote_stationary_calibration_bundle(
            bundle_path,
            candidate_config,
            template_directory,
            scene_template,
            parsed.output_dir,
            shadow_replay_pipeline_report_path=(
                parsed.shadow_replay_pipeline_report
            ),
            reviewer=parsed.reviewer,
            slot_approval_token=parsed.slot_approval_token,
            book_approval_token=parsed.book_approval_token,
            policy_tool_approval_token=parsed.policy_tool_approval_token,
            scene_approval_token=parsed.scene_approval_token,
        )
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Trial configuration: {provenance['trial_config']}")
    print(
        "Provenance: "
        + str(Path(provenance["trial_config"]).with_suffix(".provenance.json"))
    )
    print(f"Candidate ID: {provenance['candidate_id']}")
    print("All source hashes and transform checks: PASS")
    print("Local insertion allowed: False")
    print("Execution authorized: False")
    print("Hardware commanded: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
