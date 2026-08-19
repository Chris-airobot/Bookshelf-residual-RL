import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from bookshelf_shadow_ros.calibrated_preinsert_target_math import transform_to_dict
from bookshelf_shadow_ros.physical_experiment_preflight import (
    validate_provenance,
    validate_trial_configuration,
)
from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.policy_tool_candidate_check import (
    SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
)
from bookshelf_shadow_ros.stationary_calibration_promotion import (
    BOOK_APPROVAL_TOKEN,
    POLICY_TOOL_APPROVAL_TOKEN,
    SCENE_APPROVAL_TOKEN,
    promote_stationary_calibration_bundle,
)
from bookshelf_shadow_ros.stationary_capture_bundle import sha256_file
from bookshelf_shadow_ros.static_slot_capture import APPROVAL_TOKEN


PACKAGE_ROOT = Path(__file__).parents[1]
TEMPLATE_DIRECTORY = PACKAGE_ROOT / "config"
SCENE_TEMPLATE = (
    PACKAGE_ROOT.parents[0]
    / "bookshelf_guarded_control_ros"
    / "config"
    / "bookshelf_scene_physical.yaml"
)


def _candidate_id(source_hashes):
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_inputs(tmp_path: Path):
    view_a = tmp_path / "view_a.json"
    view_b = tmp_path / "view_b.json"
    book = tmp_path / "book.json"
    context = tmp_path / "context.json"
    for path, label in (
        (view_a, "view_a"),
        (view_b, "view_b"),
        (book, "book"),
        (context, "context"),
    ):
        _write_json(path, {"source": label})

    cross_view = tmp_path / "cross_view.json"
    cross_payload = {
        "views": {
            "view_a": {"report": {"path": str(view_a)}},
            "view_b": {"report": {"path": str(view_b)}},
        }
    }
    _write_json(cross_view, cross_payload)
    source_hashes = {
        "view_a_report_sha256": sha256_file(view_a),
        "view_b_report_sha256": sha256_file(view_b),
        "cross_view_slot_report_sha256": sha256_file(cross_view),
        "book_report_sha256": sha256_file(book),
        "eef_tcp_context_sha256": sha256_file(context),
    }
    candidate_id = _candidate_id(source_hashes)

    transform_eef_tcp = make_transform([0.0, 0.0, 0.172])
    transform_tcp_book = make_transform([0.008, -0.010, -0.048])
    transform_eef_book = transform_eef_tcp @ transform_tcp_book
    transform_book_policy_tool = make_transform(
        SIM_NOMINAL_BOOK_TOOL_TRANSLATION,
        SIM_NOMINAL_BOOK_TOOL_QUATERNION,
    )
    transform_eef_policy_tool = transform_eef_book @ transform_book_policy_tool
    book_status = f"measured_stationary_bag_candidate_{candidate_id}"
    tool_status = f"derived_unverified_stationary_bag_candidate_{candidate_id}"

    candidate_document = {
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
                "book_pose_source": "marker",
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
                "held_book_center_tcp_xyz": transform_tcp_book[:3, 3].tolist(),
                "held_book_quaternion_tcp_xyzw": transform_to_dict(
                    transform_tcp_book
                )["quaternion_xyzw"],
            }
        },
    }
    candidate_path = tmp_path / "stationary_calibration_candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate_document, sort_keys=False), encoding="utf-8"
    )
    slot = {
        "translation_xyz": [0.855, 0.084, 0.171],
        "quaternion_xyzw": [0.0, 0.0, 0.04, 0.9991996797437437],
        "width_m": 0.0378,
        "confidence": 0.86,
        "transform_status": "captured_rgbd_static_unapproved",
    }
    bundle = {
        "schema_version": 1,
        "kind": "bookshelf_stationary_capture_calibration_bundle_candidate",
        "candidate_id": candidate_id,
        "candidate_valid": True,
        "candidate_selected": False,
        "source_hashes": source_hashes,
        "slot": slot,
        "book_calibration": {
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
            "round_trip_translation_error_m": 0.0,
            "round_trip_rotation_error_deg": 0.0,
            "verified": False,
        },
        "safety": {
            "shadow_only": True,
            "plan_requested": False,
            "execution_authorized": False,
            "hardware_commanded": False,
            "active_configuration_modified": False,
            "human_approval_required": True,
        },
        "outputs": {
            "cross_view_slot_candidate": str(cross_view),
            "book_calibration_summary": str(book),
            "eef_tcp_context": str(context),
            "unapproved_parameter_candidate": str(candidate_path),
        },
        "output_hashes": {
            "unapproved_parameter_candidate_sha256": sha256_file(candidate_path)
        },
    }
    bundle_path = tmp_path / "bundle.json"
    _write_json(bundle_path, bundle)
    return bundle_path, candidate_path, (view_a, view_b, cross_view, book, context)


def _write_shadow_evidence(tmp_path: Path, bundle_path: Path, candidate_path: Path):
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    candidate_id = bundle["candidate_id"]
    runtime_adapter = tmp_path / "stationary_shadow_adapter.yaml"
    runtime_adapter.write_text("policy_observation_adapter: {}\n", encoding="utf-8")
    marker_report = tmp_path / "marker_book_calibration_summary.json"
    _write_json(
        marker_report,
        {
            "schema_version": 1,
            "calibration_valid": True,
            "read_only": True,
            "hardware_commanded": False,
        },
    )
    safety = {
        "shadow_only": True,
        "plan_requested": False,
        "execution_authorized": False,
        "hardware_commanded": False,
        "active_configuration_modified": False,
        "candidate_selected": False,
    }
    observation_report = tmp_path / "stationary_shadow_replay_report.json"
    _write_json(
        observation_report,
        {
            "schema_version": 1,
            "kind": "bookshelf_stationary_shadow_replay_audit",
            "candidate_id": candidate_id,
            "passed": True,
            "reason": None,
            "observation_pipeline": {
                "valid_samples": 40,
                "minimum_valid_samples": 30,
                "book_pose_sources": {"marker": 40},
                "slot_pose_sources": {"configured_static": 40},
                "failure_reasons": [],
                "policy_diagnostics": {
                    "messages": 45,
                    "adapter_policy_observation_mismatches": 0,
                    "normalized_observation": {"yaw_err": {"mean": 0.0}},
                },
            },
            "safety": safety,
        },
    )
    pipeline_report = tmp_path / "stationary_shadow_replay_pipeline_report.json"
    _write_json(
        pipeline_report,
        {
            "schema_version": 1,
            "kind": "bookshelf_stationary_shadow_replay_pipeline",
            "candidate_id": candidate_id,
            "passed": True,
            "calibration_bundle": {
                "path": str(bundle_path.resolve()),
                "sha256": sha256_file(bundle_path),
            },
            "calibration_source": {
                "candidate_path": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
            },
            "runtime_adapter": {
                "path": str(runtime_adapter),
                "sha256": sha256_file(runtime_adapter),
                "book_pose_source": "marker",
                "slot_pose_source": "configured_static",
                "policy_tool_candidate_used_for_diagnostics_only": True,
            },
            "observation_audit": {
                "path": str(observation_report),
                "sha256": sha256_file(observation_report),
                "passed": True,
            },
            "marker_detection": {
                "path": str(marker_report),
                "sha256": sha256_file(marker_report),
                "calibration_valid": True,
            },
            "safety": safety,
        },
    )
    return pipeline_report, observation_report, marker_report, runtime_adapter


def _promote(tmp_path, **overrides):
    bundle, candidate, source_paths = _write_inputs(tmp_path)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)
    values = {
        "reviewer": "test-reviewer",
        "slot_approval_token": APPROVAL_TOKEN,
        "book_approval_token": BOOK_APPROVAL_TOKEN,
        "policy_tool_approval_token": POLICY_TOOL_APPROVAL_TOKEN,
        "scene_approval_token": SCENE_APPROVAL_TOKEN,
    }
    values.update(overrides)
    output = tmp_path / "approved"
    provenance = promote_stationary_calibration_bundle(
        bundle,
        candidate,
        TEMPLATE_DIRECTORY,
        SCENE_TEMPLATE,
        output,
        shadow_replay_pipeline_report_path=shadow,
        **values,
    )
    return output, provenance, source_paths


def test_atomic_promotion_writes_one_synchronized_fail_closed_config(tmp_path):
    output, provenance, source_paths = _promote(tmp_path)
    config_path = output / "trial_static_slot.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = document["calibrated_preinsert_target"]["ros__parameters"]
    adapter = document["policy_observation_adapter"]["ros__parameters"]
    scene = document["bookshelf_scene_manager"]["ros__parameters"]

    assert target["eef_book_translation_xyz"] == adapter[
        "eef_book_translation_xyz"
    ]
    assert target["eef_policy_tool_translation_xyz"] == adapter["tool_offset_xyz"]
    assert target["eef_book_transform_status"].startswith(
        "measured_stationary_bag_human_approved_"
    )
    assert target["policy_tool_transform_status"].startswith(
        "verified_stationary_bag_policy_tool_"
    )
    assert target["policy_tool_transform_status"] == adapter[
        "policy_tool_transform_status"
    ]
    assert adapter["require_verified_policy_tool_transform"] is True
    assert adapter["book_pose_source"] == "marker"
    assert adapter["latch_eef_book_from_marker"] is False
    assert adapter["use_configured_eef_book_transform"] is True
    assert scene["hardware_measurements_confirmed"] is True
    assert scene["allow_local_insertion"] is False
    assert scene["held_book_center_tcp_xyz"] == pytest.approx(
        [0.008, -0.010, -0.048]
    )
    assert provenance["execution_authorized"] is False
    assert provenance["hardware_commanded"] is False
    assert provenance["validation"]["all_checks_passed"] is True
    assert validate_trial_configuration(config_path)["width_m"] == pytest.approx(
        0.0378
    )
    validate_provenance(config_path)
    assert all(path.is_file() for path in source_paths)


def test_promotion_migrates_matching_fixed_source_with_shadow_evidence(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    document = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    document["policy_observation_adapter"]["ros__parameters"][
        "book_pose_source"
    ] = "eef_fixed"
    candidate.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["output_hashes"]["unapproved_parameter_candidate_sha256"] = sha256_file(
        candidate
    )
    _write_json(bundle, payload)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)

    provenance = promote_stationary_calibration_bundle(
        bundle,
        candidate,
        TEMPLATE_DIRECTORY,
        SCENE_TEMPLATE,
        tmp_path / "approved",
        shadow_replay_pipeline_report_path=shadow,
        reviewer="reviewer",
        slot_approval_token=APPROVAL_TOKEN,
        book_approval_token=BOOK_APPROVAL_TOKEN,
        policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
        scene_approval_token=SCENE_APPROVAL_TOKEN,
    )

    assert provenance["runtime_source_migration"] == {
        "candidate_book_pose_source": "eef_fixed",
        "promoted_book_pose_source": "marker",
        "authorized_by_shadow_replay": True,
    }
    approved = yaml.safe_load(
        (tmp_path / "approved" / "trial_static_slot.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert approved["policy_observation_adapter"]["ros__parameters"][
        "book_pose_source"
    ] == "marker"


def test_promotion_rejects_shadow_candidate_id_mismatch(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)
    payload = json.loads(shadow.read_text(encoding="utf-8"))
    payload["candidate_id"] = "different"
    _write_json(shadow, payload)

    with pytest.raises(ValueError, match="candidate ID"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            tmp_path / "approved",
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token=BOOK_APPROVAL_TOKEN,
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )


def test_promotion_rejects_non_marker_shadow_samples(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    shadow, observation, _, _ = _write_shadow_evidence(
        tmp_path, bundle, candidate
    )
    observation_payload = json.loads(observation.read_text(encoding="utf-8"))
    observation_payload["observation_pipeline"]["book_pose_sources"] = {
        "configured_eef_book": 40
    }
    _write_json(observation, observation_payload)
    shadow_payload = json.loads(shadow.read_text(encoding="utf-8"))
    shadow_payload["observation_audit"]["sha256"] = sha256_file(observation)
    _write_json(shadow, shadow_payload)

    with pytest.raises(ValueError, match="marker poses for every book sample"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            tmp_path / "approved",
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token=BOOK_APPROVAL_TOKEN,
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )


def test_promotion_rejects_tampered_shadow_runtime_adapter(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    shadow, _, _, runtime_adapter = _write_shadow_evidence(
        tmp_path, bundle, candidate
    )
    runtime_adapter.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime-adapter hash mismatch"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            tmp_path / "approved",
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token=BOOK_APPROVAL_TOKEN,
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )


def test_promotion_rejects_unsafe_shadow_report(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)
    payload = json.loads(shadow.read_text(encoding="utf-8"))
    payload["safety"]["hardware_commanded"] = True
    _write_json(shadow, payload)

    with pytest.raises(ValueError, match="safety declaration"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            tmp_path / "approved",
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token=BOOK_APPROVAL_TOKEN,
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )


def test_promotion_rejects_any_missing_approval_without_writing(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)
    output = tmp_path / "approved"

    with pytest.raises(ValueError, match="book projection approval"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            output,
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token="NO",
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )

    assert not output.exists()


def test_promotion_rejects_tampered_source_without_writing(tmp_path):
    bundle, candidate, source_paths = _write_inputs(tmp_path)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)
    source_paths[0].write_text("tampered", encoding="utf-8")
    output = tmp_path / "approved"

    with pytest.raises(ValueError, match="source hash mismatch"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            output,
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token=BOOK_APPROVAL_TOKEN,
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )

    assert not output.exists()


def test_promotion_refuses_to_overwrite_existing_output(tmp_path):
    bundle, candidate, _ = _write_inputs(tmp_path)
    shadow, _, _, _ = _write_shadow_evidence(tmp_path, bundle, candidate)
    output = tmp_path / "approved"
    output.mkdir()

    with pytest.raises(ValueError, match="refusing to overwrite"):
        promote_stationary_calibration_bundle(
            bundle,
            candidate,
            TEMPLATE_DIRECTORY,
            SCENE_TEMPLATE,
            output,
            shadow_replay_pipeline_report_path=shadow,
            reviewer="reviewer",
            slot_approval_token=APPROVAL_TOKEN,
            book_approval_token=BOOK_APPROVAL_TOKEN,
            policy_tool_approval_token=POLICY_TOOL_APPROVAL_TOKEN,
            scene_approval_token=SCENE_APPROVAL_TOKEN,
        )


def test_promotion_module_has_no_robot_or_planning_interface():
    source = (
        PACKAGE_ROOT
        / "bookshelf_shadow_ros"
        / "stationary_calibration_promotion.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "apply_planning_scene",
        "rclpy.init",
    ):
        assert forbidden not in source
