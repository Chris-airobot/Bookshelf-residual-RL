#!/usr/bin/env python3
"""Offline audit of real Bookshelf policy logs and matching full rosbags."""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import yaml


RELEASE_THRESHOLD = 0.5
HELD_GRIPPER_SEMANTIC = 0.009838026859259968
MIN_FRESH_MARKER_SAMPLES = 20
SLOT_TRANSLATION_TOLERANCE_M = 0.001
SLOT_ROTATION_TOLERANCE_DEG = 0.5


def _timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def _matrix(document, key):
    return np.asarray(document[key]["matrix"], dtype=np.float64)


def rotation_error_degrees(left, right):
    relative = np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]
    return math.degrees(math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def classify_log(events, calculated_count):
    counts = Counter(events)
    valid_insert = calculated_count >= 2
    release = counts["release_requested"] > 0
    push = counts["push_started"] > 0 or counts["push_policy_step"] > 0
    complete = counts["episode_complete"] > 0
    if complete:
        kind = "complete_episode"
    elif release:
        kind = "partial_after_release"
    elif valid_insert:
        kind = "partial_insert"
    else:
        kind = "debug_one_step"
    return {
        "classification": kind,
        "valid_insert": valid_insert,
        "release": release,
        "push": push,
        "episode_complete": complete,
        "partial_or_debug": not complete,
    }


def contamination_label(slot_recoverable, slot_stale, marker_applicable, marker_stale):
    if not slot_recoverable or not marker_applicable:
        return "UNKNOWN"
    if slot_stale and marker_stale:
        return "BOTH"
    if slot_stale:
        return "SLOT_STALE"
    if marker_stale:
        return "MARKER_STALE"
    return "CLEAN"


def _read_log(path):
    rows = []
    parse_errors = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    calculated = [row for row in rows if row.get("event") == "calculated"]
    times = [_timestamp(row["timestamp"]) for row in rows if row.get("timestamp")]
    result = classify_log([row.get("event") for row in rows], len(calculated))
    result.update({
        "path": str(path),
        "run": path.parent.name,
        "line_count": len(rows),
        "parse_errors": parse_errors,
        "insert_decisions": len(calculated),
        "event_counts": dict(Counter(row.get("event") for row in rows)),
        "start_time": min(times) if times else None,
        "end_time": max(times) if times else None,
        "rows": rows,
        "calculated": calculated,
    })
    return result


def _bag_inventory(root):
    result = []
    for metadata_path in sorted((root / "full_real_bags").glob("*/metadata.yaml")):
        document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))["rosbag2_bagfile_information"]
        start = document["starting_time"]["nanoseconds_since_epoch"] / 1.0e9
        end = start + document["duration"]["nanoseconds"] / 1.0e9
        storage = document.get("relative_file_paths", [])
        if storage:
            db_path = metadata_path.parent / storage[0]
        else:
            db_path = next(metadata_path.parent.glob("*.db3"))
        result.append({"path": db_path, "directory": metadata_path.parent, "start": start, "end": end})
    return result


def _pair_bag(log, bags):
    if log["start_time"] is None:
        return None
    overlapping = [bag for bag in bags if log["end_time"] >= bag["start"] and log["start_time"] <= bag["end"]]
    return min(overlapping, key=lambda bag: abs(log["start_time"] - bag["start"])) if overlapping else None


def _deserialize_support():
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    return deserialize_message, get_message


def _topic(con, name):
    row = con.execute("SELECT id,type FROM topics WHERE name=?", (name,)).fetchone()
    return row if row else (None, None)


def _topic_rows(con, topic_id, start, end):
    if topic_id is None:
        return []
    return con.execute(
        "SELECT timestamp,data FROM messages INDEXED BY timestamp_idx "
        "WHERE timestamp BETWEEN ? AND ? AND topic_id=? ORDER BY timestamp",
        (int(start * 1.0e9), int(end * 1.0e9), topic_id),
    ).fetchall()


def inspect_bag(bag):
    deserialize_message, get_message = _deserialize_support()
    con = sqlite3.connect(f"file:{bag['path']}?mode=ro", uri=True)
    status_id, status_type = _topic(con, "/bookshelf_simple/status")
    status_rows = []
    if status_id:
        message_type = get_message(status_type)
        for timestamp, data in _topic_rows(con, status_id, bag["start"], bag["end"]):
            message = deserialize_message(data, message_type)
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            status_rows.append((timestamp / 1.0e9, payload))
    frozen_events = [(stamp, value) for stamp, value in status_rows if value.get("phase") == "slot_frozen"]
    frozen_slot = None
    if frozen_events:
        freeze_stamp = frozen_events[-1][0]
        slot_id, slot_type = _topic(con, "/bookshelf_simple/slot_pose_base")
        if slot_id:
            row = con.execute(
                "SELECT timestamp,data FROM messages INDEXED BY timestamp_idx "
                "WHERE timestamp<=? AND topic_id=? ORDER BY timestamp DESC LIMIT 1",
                (int((freeze_stamp + 0.1) * 1.0e9), slot_id),
            ).fetchone()
            if row:
                pose = deserialize_message(row[1], get_message(slot_type))
                p, q = pose.pose.position, pose.pose.orientation
                frozen_slot = {
                    "timestamp": row[0] / 1.0e9,
                    "translation_xyz": [p.x, p.y, p.z],
                    "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                }

    capture = None
    pg_id, pg_type = _topic(con, "/bookshelf_simple/per_grasp_status")
    if pg_id:
        pg_rows = []
        message_type = get_message(pg_type)
        for timestamp, data in _topic_rows(con, pg_id, bag["start"], bag["end"]):
            message = deserialize_message(data, message_type)
            try:
                pg_rows.append((timestamp / 1.0e9, json.loads(message.data)))
            except json.JSONDecodeError:
                pass
        starts = [stamp for stamp, value in pg_rows if value.get("source") == "collecting"]
        finishes = [(stamp, value) for stamp, value in pg_rows if value.get("source") in ("per_grasp", "fixed_fallback")]
        if starts and finishes:
            capture_start = starts[-1]
            capture_end, report = next(((stamp, value) for stamp, value in finishes if stamp >= capture_start), finishes[-1])
            tf_id, tf_type = _topic(con, "/tf")
            unique_stamps = set()
            newest_before = None
            if tf_id:
                tf_message_type = get_message(tf_type)
                for _, data in _topic_rows(con, tf_id, capture_start - 10.0, capture_end + 0.1):
                    message = deserialize_message(data, tf_message_type)
                    for transform in message.transforms:
                        if transform.child_frame_id != "target_book_center":
                            continue
                        stamp = transform.header.stamp.sec + transform.header.stamp.nanosec / 1.0e9
                        if stamp < capture_start:
                            newest_before = stamp
                        elif stamp <= capture_end + 0.1:
                            unique_stamps.add(stamp)
            newest_age = None if newest_before is None else capture_start - newest_before
            marker_stale = len(unique_stamps) < MIN_FRESH_MARKER_SAMPLES
            capture = {
                "start": capture_start,
                "end": capture_end,
                "source": report.get("source"),
                "reported_samples": report.get("sample_count"),
                "accepted_samples": report.get("accepted_count"),
                "unique_marker_updates": len(unique_stamps),
                "newest_marker_age_at_start_s": newest_age,
                "marker_stale": marker_stale,
                "diagnostics": report,
            }
    con.close()
    return {"frozen_slot": frozen_slot, "per_grasp_capture": capture}


def _load_geometry(config_path, repository):
    sys.path.insert(0, str(repository / "ros2" / "bookshelf_simple_experiment_ros"))
    from bookshelf_simple_experiment_ros.policy_observation_math import ObservationScales
    from bookshelf_simple_experiment_ros.policy_tool_math import make_transform
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = lambda name: document[name]["ros__parameters"]
    target = params("calibrated_preinsert_target")
    adapter = params("policy_observation_adapter")
    return {
        "eef_tool": make_transform(target["eef_policy_tool_translation_xyz"], target["eef_policy_tool_quaternion_xyzw"]),
        "book_size": tuple(float(value) for value in target["book_size_xyz"]),
        "slot_depth": float(target["slot_depth_m"]),
        "scales": ObservationScales(
            rear_to_mouth=float(adapter["rear_to_mouth_obs_scale"]),
            front_to_back=float(adapter["front_to_back_obs_scale"]),
            lateral=float(adapter["lat_err_obs_scale"]),
            vertical=float(adapter["z_err_obs_scale"]),
            yaw=math.radians(float(adapter["yaw_err_obs_scale_deg"])),
            tool_to_book=float(adapter["tool_to_book_obs_scale"]),
        ),
    }


def replay(calculated, frozen_slot, geometry, actor_path):
    from bookshelf_simple_experiment_ros.policy_observation_math import compute_policy_observation, invert_transform
    from bookshelf_simple_experiment_ros.policy_tool_math import make_transform
    from bookshelf_simple_experiment_ros.residual_policy_math import (
        NumpyActorBundle, combine_motion_delta, compute_policy_nominal_delta, scale_residual_action,
    )
    slot = make_transform(frozen_slot["translation_xyz"], frozen_slot["quaternion_xyzw"])
    actor = NumpyActorBundle(actor_path)
    results = []
    for row in calculated:
        if not all(key in row for key in ("T_base_book", "T_base_eef")):
            continue
        base_book = _matrix(row, "T_base_book")
        base_tool = _matrix(row, "T_base_eef") @ geometry["eef_tool"]
        raw, observation = compute_policy_observation(
            invert_transform(slot) @ base_book,
            invert_transform(slot) @ base_tool,
            book_size=geometry["book_size"], slot_depth=geometry["slot_depth"],
            mode_observation=0.0, gripper_open=HELD_GRIPPER_SEMANTIC,
            scales=geometry["scales"],
        )
        normalized, actor_mean, action = actor.predict(observation)
        nominal = compute_policy_nominal_delta(raw)
        residual = scale_residual_action(action)
        final = combine_motion_delta(nominal, residual)
        results.append({
            "step_index": row.get("step_index"),
            "raw_observation": raw.tolist(),
            "policy_observation": observation.tolist(),
            "normalized_observation": normalized.tolist(),
            "actor_mean": actor_mean.tolist(),
            "action": action.tolist(),
            "nominal_delta": nominal.tolist(),
            "final_delta": final.tolist(),
        })
    crossing = next((item for item in results if item["action"][5] > RELEASE_THRESHOLD), None)
    return {
        "states_replayed": len(results),
        "release_crossing": crossing,
        "release_crossed": crossing is not None,
        "maximum_release_actor_mean": max((item["actor_mean"][5] for item in results), default=None),
        "final": results[-1] if results else None,
        "takeover": results[0] if results else None,
    }


def _serializable_run(log):
    omitted = {"rows", "calculated"}
    return {key: value for key, value in log.items() if key not in omitted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path("/home/riot/BookshelfFiles/experiment_logs"))
    parser.add_argument("--actor", type=Path, default=Path("/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz"))
    parser.add_argument("--approved-config", type=Path, default=Path("/home/riot/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--takeover-states-output", type=Path, required=True)
    parser.add_argument("--closed-loop-results", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    geometry = _load_geometry(args.approved_config, repository)
    bags = _bag_inventory(args.experiment_root)
    logs = [_read_log(path) for path in sorted(args.experiment_root.rglob("policy_step.jsonl"))]
    bag_cache = {}
    trusted_states = []
    output_runs = []
    for log in logs:
        bag = _pair_bag(log, bags)
        bag_evidence = None
        if bag:
            key = str(bag["path"])
            if key not in bag_cache:
                bag_cache[key] = inspect_bag(bag)
            bag_evidence = bag_cache[key]
        calculated = log["calculated"]
        policy_slot = _matrix(calculated[0], "T_base_slot") if calculated and "T_base_slot" in calculated[0] else None
        frozen = bag_evidence["frozen_slot"] if bag_evidence else None
        slot_error = None
        slot_stale = False
        if policy_slot is not None and frozen:
            from bookshelf_simple_experiment_ros.policy_tool_math import make_transform
            accepted = make_transform(frozen["translation_xyz"], frozen["quaternion_xyzw"])
            slot_error = {
                "translation_mm": float(np.linalg.norm(policy_slot[:3, 3] - accepted[:3, 3]) * 1000.0),
                "rotation_deg": rotation_error_degrees(policy_slot, accepted),
            }
            slot_stale = slot_error["translation_mm"] > SLOT_TRANSLATION_TOLERANCE_M * 1000.0 or slot_error["rotation_deg"] > SLOT_ROTATION_TOLERANCE_DEG
        source = calculated[0].get("eef_book_transform_source") if calculated else None
        marker_applicable = source == "per_grasp"
        capture = bag_evidence["per_grasp_capture"] if bag_evidence else None
        marker_stale = bool(capture and capture["marker_stale"])
        contamination = contamination_label(frozen is not None, slot_stale, marker_applicable and capture is not None, marker_stale)
        replay_result = None
        if frozen and calculated:
            replay_result = replay(calculated, frozen, geometry, args.actor)
        first = calculated[0] if calculated else None
        last = calculated[-1] if calculated else None
        release_event = next((row for row in log["rows"] if row.get("event") == "release_requested"), None)
        failure_rows = [row for row in log["rows"] if row.get("event") in ("failed", "rollout_failed") or row.get("reason") and "fail" in str(row.get("reason")).lower()]
        failures = [
            {
                "event": row.get("event"), "timestamp": row.get("timestamp"),
                "reason": row.get("reason"), "servo_result": row.get("servo_result"),
            }
            for row in failure_rows
        ]
        summary = _serializable_run(log)
        summary.update({
            "bag": str(bag["directory"]) if bag else None,
            "accepted_frozen_slot": frozen,
            "policy_slot": first.get("T_base_slot") if first else None,
            "slot_error": slot_error,
            "eef_book_transform_source": source,
            "eef_book_used": first.get("T_eef_book_used") if first else None,
            "per_grasp_capture": capture,
            "measured_gripper_takeover": first.get("measured_gripper_open") if first else None,
            "semantic_gripper_takeover": first.get("policy_gripper_open", first.get("raw_observation", [None] * 10)[9]) if first else None,
            "takeover_observation": first.get("raw_observation") if first else None,
            "release_observation": last.get("raw_observation") if release_event and last else None,
            "release_action": release_event.get("release_action") if release_event else None,
            "insert_progression_m": (float(last["raw_observation"][1]) - float(first["raw_observation"][1])) if first and last else None,
            "push_complete": log["event_counts"].get("push_complete", 0) > 0,
            "failure_records": failures,
            "contamination": contamination,
            "corrected_replay": replay_result,
        })
        trustworthy = bool(
            log["valid_insert"] and frozen and marker_applicable and capture
            and not marker_stale and replay_result and replay_result["takeover"]
        )
        if trustworthy:
            trusted_states.append({
                "run": log["run"],
                "policy_log": str(log["path"]),
                "bag": str(bag["directory"]),
                "contamination": contamination,
                "raw_observation": replay_result["takeover"]["raw_observation"],
                "accepted_frozen_slot": frozen,
                "eef_book_used": first.get("T_eef_book_used"),
            })
        output_runs.append(summary)

    counts = Counter(run["contamination"] for run in output_runs)
    usable = [run for run in output_runs if run["valid_insert"]]
    corrected = [run for run in usable if run["corrected_replay"]]
    closed_loop = None
    if args.closed_loop_results:
        raw_closed_loop = json.loads(args.closed_loop_results.read_text(encoding="utf-8"))
        cases = raw_closed_loop.get("results", [])
        closed_loop = {
            "source": str(args.closed_loop_results),
            "case_count": len(cases),
            "insert_progressed": sum(bool(case.get("progressed_forward")) for case in cases),
            "entered_slot": sum(bool(case.get("entered_slot")) for case in cases),
            "released": sum(bool(case.get("released")) for case in cases),
            "final_success": sum(bool(case.get("success")) for case in cases),
            "post_push_success": sum(bool(case.get("post_push_success")) for case in cases),
            "cases": cases,
        }
    usable_slot_stale = sum(bool(run.get("slot_error") and (
        run["slot_error"]["translation_mm"] > SLOT_TRANSLATION_TOLERANCE_M * 1000.0
        or run["slot_error"]["rotation_deg"] > SLOT_ROTATION_TOLERANCE_DEG
    )) for run in usable)
    usable_marker_stale = sum(bool(run.get("per_grasp_capture") and run["per_grasp_capture"]["marker_stale"]) for run in usable)
    # Two trustworthy real states cannot establish repeatable real-world robustness.
    recommendation = "INSUFFICIENT_EVIDENCE" if len(trusted_states) < 3 else (
        "RETRAIN_JULY" if any(run["corrected_replay"]["release_crossed"] for run in corrected) else "KEEP_JULY"
    )
    audit = {
        "schema_version": 1,
        "actor": str(args.actor),
        "actor_sha256": __import__("hashlib").sha256(args.actor.read_bytes()).hexdigest(),
        "held_gripper_semantic": HELD_GRIPPER_SEMANTIC,
        "summary": {
            "total_logs": len(output_runs),
            "usable_insert_runs": len(usable),
            "complete_episodes": sum(run["episode_complete"] for run in output_runs),
            "partial_or_debug_runs": sum(run["partial_or_debug"] for run in output_runs),
            "contamination_counts_all_logs": dict(counts),
            "usable_runs_with_slot_stale_evidence": usable_slot_stale,
            "usable_runs_with_marker_stale_evidence": usable_marker_stale,
            "usable_runs_marked_clean": sum(run["contamination"] == "CLEAN" for run in usable),
            "corrected_replay_runs": len(corrected),
            "corrected_release_crossings": sum(run["corrected_replay"]["release_crossed"] for run in corrected),
            "trustworthy_takeover_states": len(trusted_states),
            "recommendation": recommendation,
        },
        "closed_loop_isaac": closed_loop,
        "runs": output_runs,
    }
    states_document = {
        "schema_version": 1,
        "description": "Bag-backed corrected real takeover states; no observation values were fabricated.",
        "states": trusted_states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.takeover_states_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.takeover_states_output.write_text(json.dumps(states_document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2))
    print(f"audit={args.output.resolve()}")
    print(f"takeover_states={args.takeover_states_output.resolve()}")


if __name__ == "__main__":
    main()
