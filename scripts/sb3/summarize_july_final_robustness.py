#!/usr/bin/env python3
"""Aggregate the diagnostic-only July joint-sensitivity and Isaac results."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values):
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "p95": _percentile(values, 95.0),
        "max": max(values),
    }


def _offline_summary(csv_path):
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    result = {}
    for profile in ("NORMAL", "STRESS"):
        selected = [row for row in rows if row["profile"] == profile]
        releases = [float(row["actor_mean_5"]) for row in selected]
        actions = [
            [float(row[f"action_{index}"]) for index in range(6)]
            for row in selected
        ]
        worst_index = max(range(len(selected)), key=lambda index: releases[index])
        worst = selected[worst_index]
        result[profile] = {
            "samples": len(selected),
            "release_crossing_count": sum(value > 0.5 for value in releases),
            "release_crossing_fraction": mean(value > 0.5 for value in releases),
            "release_raw_distribution": _distribution(releases),
            "x_forward_fraction": mean(action[0] > 0.0 for action in actions),
            "x_reverse_fraction": mean(action[0] < 0.0 for action in actions),
            "x_clipped_fraction": mean(abs(action[0]) >= 1.0 - 1e-7 for action in actions),
            "motion_clipped_fraction_xyz_yaw_pitch": [
                mean(abs(action[index]) >= 1.0 - 1e-7 for action in actions)
                for index in range(5)
            ],
            "worst_release_sample": {
                "run": worst["run"],
                "state": worst["state"],
                "sample": int(worst["sample"]),
                "release_raw": releases[worst_index],
                "action": actions[worst_index],
                "perturbation": {
                    key: float(worst[key])
                    for key in (
                        "rear_to_mouth_m", "lateral_m", "z_m", "yaw_rad",
                        "tilt_1_rad", "tilt_2_rad", "tool_delta_x_m",
                        "tool_delta_y_m", "tool_delta_z_m", "gripper_open",
                    )
                },
            },
        }
    return result


def _closed_loop_summary(document):
    result = {}
    for profile in ("NORMAL", "STRESS"):
        episodes = [
            row for row in document["results"]
            if row["case_metadata"]["profile"] == profile
        ]
        count = len(episodes)
        releases = [row["release"] for row in episodes if row["release"]]
        retreat = [row["maximum_initial_retreat_m"] for row in episodes]
        error_envelopes = {}
        for key in ("lateral", "z", "yaw"):
            values = [
                max(abs(row["insert_ranges"][key][0]), abs(row["insert_ranges"][key][1]))
                for row in episodes
            ]
            error_envelopes[key] = _distribution(values)
        insert_failures = []
        for row in episodes:
            if row["entered_slot"] and row["correct_release"]:
                continue
            insert_history = [item for item in row["history"] if item["mode"] == 0]
            insert_failures.append({
                "case": row["case"],
                "case_name": row["case_name"],
                "source_run": row["case_metadata"]["source_run"],
                "perturbation": row["case_metadata"]["perturbation"],
                "progressed_forward": row["progressed_forward"],
                "entered_slot": row["entered_slot"],
                "released": row["released"],
                "failure_code": row["failure_code"],
                "steps": row["steps"],
                "maximum_initial_retreat_m": row["maximum_initial_retreat_m"],
                "insert_ranges": row["insert_ranges"],
                "last_insert_state": insert_history[-1] if insert_history else None,
            })
        result[profile] = {
            "episodes": count,
            "progressed_forward_count": sum(row["progressed_forward"] for row in episodes),
            "progressed_forward_rate": mean(row["progressed_forward"] for row in episodes),
            "entered_slot_count": sum(row["entered_slot"] for row in episodes),
            "insert_success_rate": mean(row["entered_slot"] for row in episodes),
            "released_count": sum(row["released"] for row in episodes),
            "correct_release_count": sum(row["correct_release"] for row in episodes),
            "correct_release_rate": mean(row["correct_release"] for row in episodes),
            "premature_release_count": sum(row["premature_release"] for row in episodes),
            "premature_release_rate": mean(row["premature_release"] for row in episodes),
            "never_release_count": sum(row["never_released"] for row in episodes),
            "never_release_rate": mean(row["never_released"] for row in episodes),
            "push_entered_count": sum(row["push_entered"] for row in episodes),
            "push_entered_rate": mean(row["push_entered"] for row in episodes),
            "post_push_final_success_count": sum(row["post_push_success"] for row in episodes),
            "post_push_final_success_rate": mean(row["post_push_success"] for row in episodes),
            "downstream_failure_after_correct_release_count": sum(
                row["correct_release"] and not row["success"] for row in episodes
            ),
            "release_rear_to_mouth_m": _distribution(
                [row["rear_to_mouth"] for row in releases]
            ),
            "release_insertion_depth_m": _distribution(
                [row["insertion_depth"] for row in releases]
            ),
            "maximum_initial_retreat_m": _distribution(retreat),
            "insert_error_envelope": error_envelopes,
            "insert_failures": insert_failures,
        }
    differences = [row["difference"] for row in document["results"]]
    result["initialization_fidelity"] = {
        "max_absolute_difference_by_observation_dimension": [
            max(abs(row[index]) for row in differences) for index in range(12)
        ],
        "note": "front_to_back differs because the Isaac shelf is shallower; prior direct sensitivity proved this clipped dimension has zero actor effect in these states.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-csv", type=Path, required=True)
    parser.add_argument("--offline-config", type=Path, required=True)
    parser.add_argument("--isaac-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    offline_config = json.loads(args.offline_config.read_text(encoding="utf-8"))
    isaac = json.loads(args.isaac_results.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "policy": "July-8 residual PPO",
        "offline_joint_sensitivity": _offline_summary(args.offline_csv),
        "joint_perturbation_ranges": {
            "normal": {
                "rear_to_mouth_m": 0.003,
                "lateral_m": 0.002,
                "z_m": 0.002,
                "yaw_tilt_rad": math.radians(1.0),
                "gripper": 0.009838026859259968,
                "tool_delta_half_ranges_m": offline_config["tool_delta_normal_half_ranges_m"],
            },
            "stress": {
                "rear_to_mouth_m": 0.003,
                "lateral_m": 0.004,
                "z_m": 0.004,
                "yaw_tilt_rad": math.radians(2.0),
                "offline_gripper_range": [0.009838026859259968, 0.39],
                "closed_loop_gripper": 0.009838026859259968,
                "tool_delta_half_ranges_m": offline_config["tool_delta_stress_half_ranges_m"],
            },
            "tool_delta_observed_min_m": offline_config["tool_delta_observed_min_m"],
            "tool_delta_observed_max_m": offline_config["tool_delta_observed_max_m"],
        },
        "closed_loop_isaac": _closed_loop_summary(isaac),
        "prior_evidence": {
            "corrected_historical_release_crossings": "0/4",
            "corrected_real_state_isaac": "A and C both entered/released; C failed later in PUSH/final",
            "earlier_exact_and_small_perturbation_isaac": "11/11 entered and released; 6/10 perturbed final successes, with four PUSH/final timeouts",
            "front_to_back_22mm_actor_effect": 0.0,
            "one_at_a_time_realistic_release_crossings": 0,
        },
        "decision": {
            "recommendation": "KEEP_JULY",
            "confidence": "medium",
            "qualification": "Three of 50 NORMAL Isaac cases stalled before entry/release; zero of 30 STRESS cases did. This is a small but real tail concern, not evidence of premature release. Correct deployment geometry and fresh per-grasp calibration should be validated on hardware before reconsidering retraining.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
