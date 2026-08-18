#!/usr/bin/env python3
"""Replay stationary A/B/C bags into unapproved calibration candidates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from .stationary_capture_bundle import (
    RAW_REPLAY_TOPICS,
    build_cross_view_slot_candidate,
    build_stationary_calibration_bundle,
    inspect_capture_run,
    sha256_file,
)


EXECUTION_NODE_FRAGMENTS = (
    "guarded_policy_tool_executor",
    "guarded_preinsert_executor",
    "policy_to_robot",
    "cartesian_action_executor",
    "action_executor",
)

PIPELINE_NODE_NAMES = (
    "/rgbd_slot_detector",
    "/static_slot_capture",
    "/marker_book_calibration",
    "/eef_tcp_context_capture",
)


def slot_launch_command(
    output_dir,
    repository_path,
    target_samples,
    minimum_confidence=0.60,
    *,
    roi_x_min=0.12,
    roi_x_max=0.88,
    minimum_slot_width_m=0.020,
    maximum_slot_width_m=0.090,
    use_latest_tf=True,
) -> list[str]:
    if not 0.0 <= float(minimum_confidence) <= 1.0:
        raise ValueError("slot minimum confidence must be in [0, 1]")
    _validate_slot_search_constraints(
        roi_x_min=roi_x_min,
        roi_x_max=roi_x_max,
        minimum_slot_width_m=minimum_slot_width_m,
        maximum_slot_width_m=maximum_slot_width_m,
    )
    return [
        "ros2",
        "launch",
        "bookshelf_shadow_ros",
        "static_slot_capture.launch.py",
        f"output_dir:={Path(output_dir)}",
        f"repository_path:={Path(repository_path)}",
        f"target_samples:={int(target_samples)}",
        f"minimum_confidence:={float(minimum_confidence)}",
        f"detector_roi_x_min:={float(roi_x_min)}",
        f"detector_roi_x_max:={float(roi_x_max)}",
        f"detector_minimum_slot_width_m:={float(minimum_slot_width_m)}",
        f"detector_maximum_slot_width_m:={float(maximum_slot_width_m)}",
        f"capture_use_latest_tf:={str(bool(use_latest_tf)).lower()}",
        "use_sim_time:=true",
    ]


def _validate_slot_search_constraints(
    *,
    roi_x_min,
    roi_x_max,
    minimum_slot_width_m,
    maximum_slot_width_m,
):
    roi_x_min = float(roi_x_min)
    roi_x_max = float(roi_x_max)
    minimum_slot_width_m = float(minimum_slot_width_m)
    maximum_slot_width_m = float(maximum_slot_width_m)
    if not 0.0 <= roi_x_min < roi_x_max <= 1.0:
        raise ValueError("slot detector ROI must satisfy 0 <= min < max <= 1")
    if not 0.0 < minimum_slot_width_m < maximum_slot_width_m:
        raise ValueError("slot width limits must satisfy 0 < min < max")


def book_launch_command(
    output_dir, target_samples, mount_yaml=None
) -> list[str]:
    command = [
        "ros2",
        "launch",
        "bookshelf_shadow_ros",
        "marker_book_bag_calibration.launch.py",
        f"output_dir:={Path(output_dir)}",
        f"target_samples:={int(target_samples)}",
        "enable_rviz:=false",
        "enable_frame_audit:=true",
        "capture_eef_tcp_context:=true",
        "use_sim_time:=true",
    ]
    if mount_yaml:
        command.append(f"mount_yaml:={Path(mount_yaml)}")
    return command


def bag_play_command(bag_directory, *, rate: float) -> list[str]:
    if float(rate) <= 0.0:
        raise ValueError("bag replay rate must be positive")
    return [
        "ros2",
        "bag",
        "play",
        str(Path(bag_directory)),
        "--clock",
        "30",
        "--disable-keyboard-controls",
        "--rate",
        str(float(rate)),
        "--topics",
        *RAW_REPLAY_TOPICS,
    ]


def process_stationary_captures(
    *,
    view_a_run,
    view_b_run,
    book_run,
    output_dir,
    repository_path,
    mount_yaml=None,
    slot_target_samples: int = 120,
    slot_minimum_confidence: float = 0.60,
    view_a_roi_x_min: float = 0.12,
    view_a_roi_x_max: float = 0.88,
    view_b_roi_x_min: float = 0.12,
    view_b_roi_x_max: float = 0.88,
    view_a_minimum_slot_width_m: float = 0.020,
    view_a_maximum_slot_width_m: float = 0.090,
    view_b_minimum_slot_width_m: float = 0.020,
    view_b_maximum_slot_width_m: float = 0.090,
    book_target_samples: int = 250,
    replay_rate: float = 1.0,
    maximum_translation_disagreement_m: float = 0.010,
    maximum_rotation_disagreement_deg: float = 5.0,
    maximum_rotation_sanity_disagreement_deg: float = 15.0,
    maximum_width_disagreement_m: float = 0.005,
    hash_bag_files: bool = True,
) -> dict:
    """Run all read-only replays and assemble the candidate bundle."""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    _assert_ros_graph_is_safe()
    source_captures = {
        "view_a": inspect_capture_run(
            view_a_run,
            role="view_a",
            expected_condition="no_book",
            hash_bag_files=hash_bag_files,
        ),
        "view_b": inspect_capture_run(
            view_b_run,
            role="view_b",
            expected_condition="no_book",
            hash_bag_files=hash_bag_files,
        ),
        "book_attached": inspect_capture_run(
            book_run,
            role="book_attached",
            expected_condition="book_attached",
            hash_bag_files=hash_bag_files,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    slot_detector_constraints = {
        "tf_lookup_mode": "latest_available_stationary",
        "view_a": {
            "roi_x_normalized": [view_a_roi_x_min, view_a_roi_x_max],
            "slot_width_m": [
                view_a_minimum_slot_width_m,
                view_a_maximum_slot_width_m,
            ],
        },
        "view_b": {
            "roi_x_normalized": [view_b_roi_x_min, view_b_roi_x_max],
            "slot_width_m": [
                view_b_minimum_slot_width_m,
                view_b_maximum_slot_width_m,
            ],
        },
    }
    for label in ("view_a", "view_b"):
        constraints = slot_detector_constraints[label]
        _validate_slot_search_constraints(
            roi_x_min=constraints["roi_x_normalized"][0],
            roi_x_max=constraints["roi_x_normalized"][1],
            minimum_slot_width_m=constraints["slot_width_m"][0],
            maximum_slot_width_m=constraints["slot_width_m"][1],
        )
    _write_json(output_dir / "capture_input_audit.json", {
        "schema_version": 1,
        "kind": "bookshelf_stationary_capture_input_audit",
        "generated_at": datetime.now().astimezone().isoformat(),
        "validated": True,
        "hardware_commanded": False,
        "source_captures": source_captures,
        "slot_detector_constraints": slot_detector_constraints,
    })

    stages = (
        (
            "view_a",
            source_captures["view_a"],
            slot_launch_command(
                output_dir / "view_a",
                repository_path,
                slot_target_samples,
                slot_minimum_confidence,
                roi_x_min=view_a_roi_x_min,
                roi_x_max=view_a_roi_x_max,
                minimum_slot_width_m=view_a_minimum_slot_width_m,
                maximum_slot_width_m=view_a_maximum_slot_width_m,
            ),
            output_dir / "view_a" / "static_slot_capture_candidate.json",
        ),
        (
            "view_b",
            source_captures["view_b"],
            slot_launch_command(
                output_dir / "view_b",
                repository_path,
                slot_target_samples,
                slot_minimum_confidence,
                roi_x_min=view_b_roi_x_min,
                roi_x_max=view_b_roi_x_max,
                minimum_slot_width_m=view_b_minimum_slot_width_m,
                maximum_slot_width_m=view_b_maximum_slot_width_m,
            ),
            output_dir / "view_b" / "static_slot_capture_candidate.json",
        ),
        (
            "book_attached",
            source_captures["book_attached"],
            book_launch_command(
                output_dir / "book", book_target_samples, mount_yaml
            ),
            output_dir / "book" / "marker_book_calibration_summary.json",
        ),
    )
    stage_records = {}
    for name, capture, launch_command, expected_report in stages:
        print(f"===== PROCESSING {name.upper()} =====", flush=True)
        stage_records[name] = _run_replay_stage(
            name=name,
            launch_command=launch_command,
            bag_directory=capture["bag_directory"],
            bag_duration_s=float(capture["duration_s"]),
            replay_rate=replay_rate,
            expected_report=expected_report,
            log_directory=expected_report.parent,
        )

    view_a_report_path = Path(stage_records["view_a"]["report"])
    view_b_report_path = Path(stage_records["view_b"]["report"])
    book_report_path = Path(stage_records["book_attached"]["report"])
    context_path = output_dir / "book" / "eef_tcp_context.json"
    view_a_report = _load_json(view_a_report_path)
    view_b_report = _load_json(view_b_report_path)
    book_report = _load_json(book_report_path)
    eef_tcp_context = _load_json(context_path)

    view_a_report_provenance = {
        "path": str(view_a_report_path),
        "sha256": sha256_file(view_a_report_path),
    }
    view_b_report_provenance = {
        "path": str(view_b_report_path),
        "sha256": sha256_file(view_b_report_path),
    }
    slot_report = build_cross_view_slot_candidate(
        view_a_report,
        view_b_report,
        view_a_provenance=view_a_report_provenance,
        view_b_provenance=view_b_report_provenance,
        maximum_translation_disagreement_m=(
            maximum_translation_disagreement_m
        ),
        maximum_rotation_disagreement_deg=(
            maximum_rotation_disagreement_deg
        ),
        maximum_rotation_sanity_disagreement_deg=(
            maximum_rotation_sanity_disagreement_deg
        ),
        maximum_width_disagreement_m=maximum_width_disagreement_m,
    )
    slot_report_path = output_dir / "static_slot_cross_view_candidate.json"
    _write_json(slot_report_path, slot_report)
    if not slot_report["valid"]:
        raise ValueError(f"A/B slot validation failed: {slot_report['reason']}")

    source_hashes = {
        "view_a_report_sha256": view_a_report_provenance["sha256"],
        "view_b_report_sha256": view_b_report_provenance["sha256"],
        "cross_view_slot_report_sha256": sha256_file(slot_report_path),
        "book_report_sha256": sha256_file(book_report_path),
        "eef_tcp_context_sha256": sha256_file(context_path),
    }
    bundle, candidate_parameters = build_stationary_calibration_bundle(
        slot_report,
        book_report,
        eef_tcp_context,
        capture_provenance=source_captures,
        source_hashes=source_hashes,
    )
    bundle["pipeline_stages"] = stage_records
    candidate_path = output_dir / "stationary_calibration_candidate.yaml"
    candidate_path.write_text(
        "# UNAPPROVED CANDIDATE. Review and promote through separate gates.\n"
        + yaml.safe_dump(candidate_parameters, sort_keys=False),
        encoding="utf-8",
    )
    bundle["outputs"] = {
        "cross_view_slot_candidate": str(slot_report_path),
        "book_calibration_summary": str(book_report_path),
        "eef_tcp_context": str(context_path),
        "unapproved_parameter_candidate": str(candidate_path),
    }
    bundle["output_hashes"] = {
        "unapproved_parameter_candidate_sha256": sha256_file(candidate_path),
    }
    bundle_path = output_dir / "stationary_calibration_bundle_candidate.json"
    _write_json(bundle_path, bundle)
    print(f"Cross-view slot candidate: {slot_report_path}")
    print(f"Calibration bundle candidate: {bundle_path}")
    print(f"Parameter candidate: {candidate_path}")
    print("Candidate valid: True")
    print("Candidate selected: False")
    print("Execution authorized: False")
    print("Hardware commanded: False")
    return bundle


def _run_replay_stage(
    *,
    name: str,
    launch_command: list[str],
    bag_directory: str,
    bag_duration_s: float,
    replay_rate: float,
    expected_report: Path,
    log_directory: Path,
) -> dict:
    log_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    ros_log_dir = log_directory / "ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    launch_log_path = log_directory / "launch.log"
    playback_log_path = log_directory / "bag_play.log"
    launch_process = None
    playback_process = None
    started_at = datetime.now().astimezone().isoformat()
    try:
        with launch_log_path.open("w", encoding="utf-8") as launch_log:
            launch_process = subprocess.Popen(
                launch_command,
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
            )
        time.sleep(2.0)
        if launch_process.poll() is not None:
            raise RuntimeError(
                f"{name} launch exited before replay; inspect {launch_log_path}"
            )
        play_command = bag_play_command(bag_directory, rate=replay_rate)
        with playback_log_path.open("w", encoding="utf-8") as playback_log:
            playback_process = subprocess.Popen(
                play_command,
                stdout=playback_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
            )
            timeout_s = max(float(bag_duration_s) / float(replay_rate) + 45.0, 60.0)
            try:
                playback_returncode = playback_process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"{name} bag replay exceeded {timeout_s:.1f}s"
                ) from error
        if playback_returncode != 0:
            raise RuntimeError(
                f"{name} bag replay failed; inspect {playback_log_path}"
            )
        time.sleep(3.0)
    finally:
        _stop_process_group(playback_process)
        _stop_process_group(launch_process)

    if not expected_report.is_file():
        raise RuntimeError(
            f"{name} did not write {expected_report}; inspect {launch_log_path}"
        )
    report = _load_json(expected_report)
    if name in ("view_a", "view_b") and report.get("valid") is not True:
        raise RuntimeError(f"{name} slot capture is invalid: {report.get('reason')}")
    if name == "book_attached" and report.get("calibration_valid") is not True:
        raise RuntimeError("book-attached calibration is invalid")
    return {
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "launch_command": launch_command,
        "bag_play_command": bag_play_command(bag_directory, rate=replay_rate),
        "report": str(expected_report),
        "report_sha256": sha256_file(expected_report),
        "launch_log": str(launch_log_path),
        "bag_play_log": str(playback_log_path),
    }


def _stop_process_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2.0)


def _assert_ros_graph_is_safe():
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("could not inspect the ROS graph") from error
    if result.returncode != 0:
        raise RuntimeError(f"ros2 node list failed: {result.stderr.strip()}")
    nodes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    blocked = [
        node
        for node in nodes
        if any(fragment in node for fragment in EXECUTION_NODE_FRAGMENTS)
    ]
    conflicts = [node for node in nodes if node in PIPELINE_NODE_NAMES]
    if blocked:
        raise RuntimeError("execution-capable ROS nodes are active: " + ", ".join(blocked))
    if conflicts:
        raise RuntimeError("pipeline node names are already active: " + ", ".join(conflicts))


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay stationary no-book A/B and attached-book C captures into "
            "unapproved, fail-closed calibration candidates."
        )
    )
    parser.add_argument("--view-a-run", required=True)
    parser.add_argument("--view-b-run", required=True)
    parser.add_argument("--book-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--mount-yaml")
    parser.add_argument("--slot-target-samples", type=int, default=120)
    parser.add_argument("--slot-minimum-confidence", type=float, default=0.60)
    parser.add_argument("--view-a-roi-x-min", type=float, default=0.12)
    parser.add_argument("--view-a-roi-x-max", type=float, default=0.88)
    parser.add_argument("--view-b-roi-x-min", type=float, default=0.12)
    parser.add_argument("--view-b-roi-x-max", type=float, default=0.88)
    parser.add_argument(
        "--view-a-minimum-slot-width-m", type=float, default=0.020
    )
    parser.add_argument(
        "--view-a-maximum-slot-width-m", type=float, default=0.090
    )
    parser.add_argument(
        "--view-b-minimum-slot-width-m", type=float, default=0.020
    )
    parser.add_argument(
        "--view-b-maximum-slot-width-m", type=float, default=0.090
    )
    parser.add_argument("--book-target-samples", type=int, default=250)
    parser.add_argument("--replay-rate", type=float, default=1.0)
    parser.add_argument("--maximum-translation-disagreement-m", type=float, default=0.010)
    parser.add_argument("--maximum-rotation-disagreement-deg", type=float, default=5.0)
    parser.add_argument(
        "--maximum-rotation-sanity-disagreement-deg",
        type=float,
        default=15.0,
        help=(
            "Hard stop for gross View B orientation mismatch; smaller "
            "differences above the diagnostic threshold remain warnings."
        ),
    )
    parser.add_argument("--maximum-width-disagreement-m", type=float, default=0.005)
    parser.add_argument(
        "--skip-bag-hashes",
        action="store_true",
        help="Skip large data-file hashes for a quick smoke run.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_was_nonempty = output_dir.exists() and any(output_dir.iterdir())
    try:
        process_stationary_captures(
            view_a_run=args.view_a_run,
            view_b_run=args.view_b_run,
            book_run=args.book_run,
            output_dir=output_dir,
            repository_path=args.repository,
            mount_yaml=args.mount_yaml,
            slot_target_samples=args.slot_target_samples,
            slot_minimum_confidence=args.slot_minimum_confidence,
            view_a_roi_x_min=args.view_a_roi_x_min,
            view_a_roi_x_max=args.view_a_roi_x_max,
            view_b_roi_x_min=args.view_b_roi_x_min,
            view_b_roi_x_max=args.view_b_roi_x_max,
            view_a_minimum_slot_width_m=args.view_a_minimum_slot_width_m,
            view_a_maximum_slot_width_m=args.view_a_maximum_slot_width_m,
            view_b_minimum_slot_width_m=args.view_b_minimum_slot_width_m,
            view_b_maximum_slot_width_m=args.view_b_maximum_slot_width_m,
            book_target_samples=args.book_target_samples,
            replay_rate=args.replay_rate,
            maximum_translation_disagreement_m=(
                args.maximum_translation_disagreement_m
            ),
            maximum_rotation_disagreement_deg=(
                args.maximum_rotation_disagreement_deg
            ),
            maximum_rotation_sanity_disagreement_deg=(
                args.maximum_rotation_sanity_disagreement_deg
            ),
            maximum_width_disagreement_m=args.maximum_width_disagreement_m,
            hash_bag_files=not args.skip_bag_hashes,
        )
    except Exception as error:
        print(f"FAIL: {error}")
        if not output_was_nonempty:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "pipeline_failure.json"
            _write_json(failure_path, {
                "schema_version": 1,
                "kind": "bookshelf_stationary_capture_pipeline_failure",
                "generated_at": datetime.now().astimezone().isoformat(),
                "reason": str(error),
                "candidate_valid": False,
                "candidate_selected": False,
                "execution_authorized": False,
                "hardware_commanded": False,
                "active_configuration_modified": False,
            })
            print(f"Failure report: {failure_path}")
        else:
            print("Existing output directory was left unchanged.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
