"""Fail-closed validation for the unified physical shadow rehearsal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


APPROVAL_TOKEN = "VISUALLY_APPROVED_STATIC_SLOT"


def validate_shadow_rehearsal_assets(
    approved_config,
    policy_bundle,
    activation_envelope,
) -> dict:
    """Validate immutable rehearsal inputs without creating ROS interfaces."""

    config_path = _required_file(approved_config, "approved configuration")
    policy_path = _required_file(policy_bundle, "policy bundle")
    envelope_path = _required_file(
        activation_envelope, "policy activation envelope"
    )
    provenance_path = config_path.with_suffix(".provenance.json")
    provenance_path = _required_file(provenance_path, "configuration provenance")

    document = _load_yaml(config_path)
    provenance = _load_json(provenance_path)
    envelope = _load_json(envelope_path)

    _validate_provenance(config_path, provenance)
    slot = _parameters(document, "static_slot_environment_check")
    target = _parameters(document, "calibrated_preinsert_target")
    adapter = _parameters(document, "policy_observation_adapter")
    scene = _parameters(document, "bookshelf_scene_manager")

    slot_statuses = {
        str(slot.get("static_slot_transform_status", "")),
        str(target.get("static_slot_transform_status", "")),
        str(adapter.get("static_slot_transform_status", "")),
    }
    if len(slot_statuses) != 1 or not next(iter(slot_statuses)).startswith(
        "captured_rgbd_static_human_approved_"
    ):
        raise ValueError("approved slot status is missing or inconsistent")
    if adapter.get("slot_pose_source") != "configured_static":
        raise ValueError("rehearsal must use the frozen configured slot")
    if adapter.get("allow_configured_static_slot") is not True:
        raise ValueError("configured static slot is not enabled")
    if adapter.get("book_pose_source") != "marker":
        raise ValueError("rehearsal must use continuous marker book poses")
    if adapter.get("latch_eef_book_from_marker") is not False:
        raise ValueError("rehearsal must not latch one marker observation")
    if adapter.get("use_configured_eef_book_transform") is not True:
        raise ValueError("fixed grasp reference is missing")
    if adapter.get("require_verified_policy_tool_transform") is not True:
        raise ValueError("policy-tool transform is not required to be verified")
    if not str(adapter.get("policy_tool_transform_status", "")).startswith(
        "verified_stationary_bag_policy_tool_"
    ):
        raise ValueError("policy-tool transform status is not approved")

    required_scene = {
        "hardware_measurements_confirmed": True,
        "allow_local_insertion": False,
        "held_book_enabled": True,
        "require_held_book_pose_check": True,
    }
    wrong_scene = {
        key: scene.get(key)
        for key, expected in required_scene.items()
        if scene.get(key) is not expected
    }
    if wrong_scene:
        raise ValueError(f"scene is not rehearsal-safe: {wrong_scene}")

    labels = envelope.get("labels")
    lower = envelope.get("lower")
    upper = envelope.get("upper")
    if not isinstance(labels, list) or len(labels) != 12:
        raise ValueError("activation envelope must contain 12 labels")
    if not isinstance(lower, list) or not isinstance(upper, list):
        raise ValueError("activation envelope bounds are missing")
    if len(lower) != len(labels) or len(upper) != len(labels):
        raise ValueError("activation envelope bounds do not match its labels")
    if any(float(low) > float(high) for low, high in zip(lower, upper)):
        raise ValueError("activation envelope contains inverted bounds")
    if policy_path.suffix != ".npz":
        raise ValueError("policy bundle must be the portable .npz actor")

    return {
        "candidate_id": str(provenance.get("candidate_id", "")),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "policy_bundle": str(policy_path),
        "policy_bundle_sha256": _sha256_file(policy_path),
        "activation_envelope": str(envelope_path),
        "activation_envelope_sha256": _sha256_file(envelope_path),
        "slot_width_m": float(slot["static_slot_width_m"]),
        "book_pose_source": "marker",
        "slot_pose_source": "configured_static",
        "allow_local_insertion": False,
        "execution_authorized": False,
        "hardware_commanded": False,
    }


def _validate_provenance(config_path: Path, provenance: dict) -> None:
    if provenance.get("human_approval_recorded") is not True:
        raise ValueError("configuration provenance has no human approval")
    if provenance.get("approval_token") != APPROVAL_TOKEN:
        raise ValueError("configuration provenance approval token is invalid")
    if provenance.get("hardware_commanded") is not False:
        raise ValueError("configuration provenance is not hardware-safe")
    if provenance.get("execution_authorized") is not False:
        raise ValueError("configuration provenance authorizes execution")
    if provenance.get("trial_config_sha256") != _sha256_file(config_path):
        raise ValueError("approved configuration hash differs from provenance")
    candidate = _required_file(
        provenance.get("candidate_report"), "approved candidate report"
    )
    if provenance.get("candidate_report_sha256") != _sha256_file(candidate):
        raise ValueError("approved candidate report hash differs from provenance")


def _parameters(document: dict, node_name: str) -> dict:
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"approved configuration is missing {node_name}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"invalid parameters for {node_name}")
    return parameters


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
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
