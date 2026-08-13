"""Fail-closed software and live-ROS preflight for a bookshelf trial."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from .static_slot_capture import APPROVAL_TOKEN


CRITICAL_NODE_NAMES = {
    "/camera",
    "/rgbd_slot_detector",
    "/static_slot_environment_check",
    "/guarded_policy_tool_executor",
    "/policy_tool_plan_checker",
}
PROHIBITED_EXECUTION_NODES = {
    "/guarded_policy_tool_executor",
    "/policy_to_robot_node",
    "/cartesian_action_executor_node",
    "/action_executor_node",
}
REQUIRED_HARDWARE_NODES = {
    "/camera",
    "/controller_manager",
    "/move_group",
    "/robot_state_publisher",
}
REQUIRED_TOPICS = {
    "/camera/color/image_raw",
    "/camera/depth/image_rect_raw",
    "/camera/aligned_depth_to_color/image_raw",
    "/joint_states",
    "/tf",
    "/tf_static",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameters(document: dict, node_name: str) -> dict:
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing ROS parameters for {node_name}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"invalid ROS parameters for {node_name}")
    return parameters


def validate_trial_configuration(config_path: Path) -> dict:
    text = config_path.read_text(encoding="utf-8")
    if "&id" in text or "*id" in text:
        raise ValueError("trial configuration contains ROS-incompatible YAML aliases")
    document = yaml.safe_load(text)
    check = _parameters(document, "static_slot_environment_check")
    target = _parameters(document, "calibrated_preinsert_target")
    adapter = _parameters(document, "policy_observation_adapter")

    translations = [
        check["static_slot_translation_xyz"],
        target["static_slot_translation_xyz"],
        adapter["configured_static_slot_translation_xyz"],
    ]
    quaternions = [
        check["static_slot_quaternion_xyzw"],
        target["static_slot_quaternion_xyzw"],
        adapter["configured_static_slot_quaternion_xyzw"],
    ]
    widths = [
        check["static_slot_width_m"],
        target["static_slot_width_m"],
        adapter["configured_static_slot_width_m"],
    ]
    statuses = [
        check["static_slot_transform_status"],
        target["static_slot_transform_status"],
        adapter["static_slot_transform_status"],
    ]
    if not all(np.allclose(value, translations[0], atol=1.0e-12) for value in translations):
        raise ValueError("slot translations differ between trial nodes")
    if not all(np.allclose(value, quaternions[0], atol=1.0e-12) for value in quaternions):
        raise ValueError("slot orientations differ between trial nodes")
    if not all(np.isclose(value, widths[0], atol=1.0e-12) for value in widths):
        raise ValueError("slot widths differ between trial nodes")
    if len(set(statuses)) != 1 or not statuses[0].startswith(
        "captured_rgbd_static_human_approved_"
    ):
        raise ValueError("slot transform status is inconsistent or unapproved")

    return {
        "translation_xyz": list(map(float, translations[0])),
        "quaternion_xyzw": list(map(float, quaternions[0])),
        "width_m": float(widths[0]),
        "confidence": float(target["static_slot_confidence"]),
        "transform_status": statuses[0],
    }


def validate_provenance(config_path: Path) -> dict:
    path = config_path.with_suffix(".provenance.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("human_approval_recorded") is not True:
        raise ValueError("provenance does not record human approval")
    if document.get("approval_token") != APPROVAL_TOKEN:
        raise ValueError("provenance approval token is invalid")
    if document.get("hardware_commanded") is not False:
        raise ValueError("provenance does not prove hardware_commanded=false")
    candidate = Path(document["candidate_report"]).expanduser()
    if not candidate.is_file():
        raise ValueError(f"candidate report is missing: {candidate}")
    actual_hash = sha256_file(candidate)
    if actual_hash != document.get("candidate_report_sha256"):
        raise ValueError("candidate report SHA256 does not match provenance")
    return {
        "path": str(path),
        "candidate_report": str(candidate),
        "candidate_report_sha256": actual_hash,
    }


def validate_policy_assets(bundle_path: Path, envelope_path: Path) -> dict:
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    labels = envelope.get("labels")
    lower = envelope.get("lower")
    upper = envelope.get("upper")
    if not isinstance(labels, list) or len(labels) != 12:
        raise ValueError("activation envelope must contain 12 labels")
    if len(lower) != len(labels) or len(upper) != len(labels):
        raise ValueError("activation envelope bounds do not match its labels")
    if not np.all(np.asarray(lower, dtype=float) <= np.asarray(upper, dtype=float)):
        raise ValueError("activation envelope has inverted bounds")
    return {
        "policy_bundle": str(bundle_path),
        "policy_bundle_sha256": sha256_file(bundle_path),
        "activation_envelope": str(envelope_path),
        "activation_envelope_sha256": sha256_file(envelope_path),
        "activation_envelope_source_sha256": envelope.get("metadata", {}).get(
            "source_sha256"
        ),
    }


def validate_fail_closed_executor(config_path: Path, bundle_sha256: str) -> dict:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = _parameters(document, "guarded_policy_tool_executor")
    required = {
        "dry_run": True,
        "allow_execution": False,
        "planning_scene_complete": False,
        "approval_token": "DISABLED",
    }
    wrong = {
        name: {"expected": expected, "actual": parameters.get(name)}
        for name, expected in required.items()
        if parameters.get(name) != expected
    }
    if wrong:
        raise ValueError(f"committed executor is not fail-closed: {wrong}")
    expected_hash = str(parameters.get("expected_bundle_sha256", ""))
    if expected_hash != bundle_sha256:
        raise ValueError("policy bundle SHA256 differs from executor expectation")
    return {
        "path": str(config_path),
        "dry_run": True,
        "allow_execution": False,
        "planning_scene_complete": False,
        "approval_token": "DISABLED",
        "execution_authorized": False,
    }


def _run(command: list[str], timeout_s: float) -> tuple[int | None, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
        return None, (stdout or "") + (stderr or "")


def validate_live_ros(
    *,
    timeout_s: float,
    require_frozen_check: bool,
    expected_shadow_prefix: str | None = None,
    expected_guarded_prefix: str | None = None,
) -> dict:
    package_prefixes = {}
    for package, expected in (
        ("bookshelf_shadow_ros", expected_shadow_prefix),
        ("bookshelf_guarded_control_ros", expected_guarded_prefix),
    ):
        code, output = _run(["ros2", "pkg", "prefix", package], timeout_s)
        prefix = output.strip().splitlines()[0] if output.strip() else ""
        if code != 0 or not prefix:
            raise ValueError(f"ROS package is unavailable: {package}")
        if expected and Path(prefix).resolve() != Path(expected).expanduser().resolve():
            raise ValueError(
                f"{package} resolves to {prefix}, expected {expected}"
            )
        package_prefixes[package] = prefix

    _, node_output = _run(["ros2", "node", "list"], timeout_s)
    nodes = [line.strip() for line in node_output.splitlines() if line.startswith("/")]
    counts = {name: nodes.count(name) for name in CRITICAL_NODE_NAMES}
    duplicates = {name: count for name, count in counts.items() if count > 1}
    missing_nodes = sorted(REQUIRED_HARDWARE_NODES - set(nodes))
    prohibited = sorted(PROHIBITED_EXECUTION_NODES & set(nodes))
    if duplicates:
        raise ValueError(f"duplicate critical ROS nodes: {duplicates}")
    if missing_nodes:
        raise ValueError(f"required hardware nodes are missing: {missing_nodes}")
    if prohibited:
        raise ValueError(f"robot execution nodes are running: {prohibited}")
    if require_frozen_check and counts["/static_slot_environment_check"] != 1:
        raise ValueError("exactly one frozen static-slot check node is required")

    _, topic_output = _run(["ros2", "topic", "list"], timeout_s)
    topics = {line.strip() for line in topic_output.splitlines() if line.startswith("/")}
    missing_topics = sorted(REQUIRED_TOPICS - topics)
    if missing_topics:
        raise ValueError(f"required ROS topics are missing: {missing_topics}")

    probes = {}
    for name, field in (
        ("/camera/color/image_raw", "header"),
        ("/camera/depth/image_rect_raw", "header"),
        ("/camera/aligned_depth_to_color/image_raw", "header"),
        ("/joint_states", "name"),
    ):
        code, output = _run(
            ["ros2", "topic", "echo", "--once", name, "--field", field],
            timeout_s,
        )
        if code != 0 or "---" not in output:
            raise ValueError(f"no live message received from {name}")
        probes[name] = "message_received"

    _, tf_output = _run(
        [
            "ros2",
            "run",
            "tf2_ros",
            "tf2_echo",
            "link_base",
            "camera_color_optical_frame",
        ],
        timeout_s,
    )
    if "Translation:" not in tf_output:
        raise ValueError("TF link_base <- camera_color_optical_frame is unavailable")
    probes["link_base<-camera_color_optical_frame"] = "transform_received"

    frozen_check = "not_required"
    if require_frozen_check:
        code, output = _run(
            [
                "ros2",
                "topic",
                "echo",
                "--once",
                "/bookshelf_environment/static_slot_check_passed",
                "--field",
                "data",
            ],
            timeout_s,
        )
        if code != 0 or "true" not in output.lower():
            raise ValueError("frozen static-slot live check is not passing")
        frozen_check = "passed"

    return {
        "package_prefixes": package_prefixes,
        "critical_node_counts": counts,
        "prohibited_execution_nodes": prohibited,
        "probes": probes,
        "frozen_slot_check": frozen_check,
    }


def _record_check(checks: list[dict], name: str, function, *args, **kwargs):
    try:
        details = function(*args, **kwargs)
        checks.append({"name": name, "passed": True, "details": details})
        return details
    except Exception as error:
        checks.append({"name": name, "passed": False, "reason": str(error)})
        return None


def main(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read-only bookshelf software preflight. It creates no ROS publishers, "
            "planners, controllers, gripper clients, or execution clients."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--trial-slot-config", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--activation-envelope", type=Path, required=True)
    parser.add_argument("--executor-config", type=Path)
    parser.add_argument("--check-ros", action="store_true")
    parser.add_argument("--require-frozen-check", action="store_true")
    parser.add_argument("--topic-timeout-s", type=float, default=6.0)
    parser.add_argument("--expected-shadow-prefix")
    parser.add_argument("--expected-guarded-prefix")
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(args)

    repository = parsed.repository.expanduser().resolve()
    slot_config = parsed.trial_slot_config.expanduser().resolve()
    bundle = parsed.policy_bundle.expanduser().resolve()
    envelope = parsed.activation_envelope.expanduser().resolve()
    executor = (
        parsed.executor_config.expanduser().resolve()
        if parsed.executor_config
        else repository
        / "ros2/bookshelf_guarded_control_ros/config/guarded_policy_tool_executor.yaml"
    )
    checks: list[dict[str, Any]] = []
    slot = _record_check(
        checks, "trial_slot_configuration", validate_trial_configuration, slot_config
    )
    _record_check(checks, "trial_slot_provenance", validate_provenance, slot_config)
    assets = _record_check(
        checks, "policy_assets", validate_policy_assets, bundle, envelope
    )
    if assets:
        _record_check(
            checks,
            "committed_executor_fail_closed",
            validate_fail_closed_executor,
            executor,
            assets["policy_bundle_sha256"],
        )
    if parsed.check_ros:
        _record_check(
            checks,
            "live_ros",
            validate_live_ros,
            timeout_s=parsed.topic_timeout_s,
            require_frozen_check=parsed.require_frozen_check,
            expected_shadow_prefix=parsed.expected_shadow_prefix,
            expected_guarded_prefix=parsed.expected_guarded_prefix,
        )

    passed = bool(checks) and all(check["passed"] for check in checks)
    report = {
        "schema_version": 1,
        "kind": "bookshelf_physical_experiment_software_preflight",
        "generated_at": datetime.now().astimezone().isoformat(),
        "passed": passed,
        "software_ready": passed,
        "execution_authorized": False,
        "hardware_commanded": False,
        "repository": str(repository),
        "slot": slot,
        "checks": checks,
        "limitations": [
            "Software readiness does not authorize robot execution.",
            "This preflight does not check collision geometry, IK, reachability, or contact.",
            "A separately reviewed physical executor configuration is still required for execution.",
        ],
    }
    output = parsed.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report: {output}")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'}: {check['name']}")
        if not check["passed"]:
            print(f"  {check['reason']}")
    print(f"Software ready: {passed}")
    print("Execution authorized: False")
    print("Hardware commanded: False")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
