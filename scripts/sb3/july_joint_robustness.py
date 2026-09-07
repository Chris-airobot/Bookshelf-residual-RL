#!/usr/bin/env python3
"""CPU-only joint sensitivity and deterministic Isaac case generation for July PPO."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


HELD_GRIPPER = 0.009838026859259968
RUNS = ("simple_policy_20260904_003219", "simple_policy_20260904_005654")


def _matrix(row, key):
    return np.asarray(row[key]["matrix"], dtype=np.float64)


def _uniform(rng, magnitude):
    return float(rng.uniform(-magnitude, magnitude))


def generate_perturbation(rng, profile, tool_half_ranges):
    stress = profile == "STRESS"
    spatial = 0.004 if stress else 0.002
    angle = math.radians(2.0 if stress else 1.0)
    tool_multiplier = 2.0 if stress else 1.0
    return {
        "rear_to_mouth_m": _uniform(rng, 0.003),
        "lateral_m": _uniform(rng, spatial),
        "z_m": _uniform(rng, spatial),
        "yaw_rad": _uniform(rng, angle),
        "tilt_1_rad": _uniform(rng, angle),
        "tilt_2_rad": _uniform(rng, angle),
        "tool_delta_x_m": _uniform(rng, tool_half_ranges[0] * tool_multiplier),
        "tool_delta_y_m": _uniform(rng, tool_half_ranges[1] * tool_multiplier),
        "tool_delta_z_m": _uniform(rng, tool_half_ranges[2] * tool_multiplier),
        "gripper_open": float(rng.uniform(HELD_GRIPPER, 0.39)) if stress else HELD_GRIPPER,
    }


def apply_perturbation(raw, perturbation):
    value = np.asarray(raw, dtype=np.float64).copy()
    value[1] += perturbation["rear_to_mouth_m"]
    value[3] += perturbation["lateral_m"]
    value[4] += perturbation["z_m"]
    value[5] += perturbation["yaw_rad"]
    value[6] += perturbation["tool_delta_x_m"]
    value[7] += perturbation["tool_delta_y_m"]
    value[8] += perturbation["tool_delta_z_m"]
    value[9] = perturbation["gripper_open"]
    value[10] = math.sin(math.asin(float(np.clip(value[10], -1.0, 1.0))) + perturbation["tilt_1_rad"])
    value[11] = math.sin(math.asin(float(np.clip(value[11], -1.0, 1.0))) + perturbation["tilt_2_rad"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-state", type=int, default=1000)
    parser.add_argument("--normal-isaac-cases", type=int, default=50)
    parser.add_argument("--stress-isaac-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "ros2" / "bookshelf_simple_experiment_ros"))
    from bookshelf_simple_experiment_ros.policy_observation_math import compute_policy_observation, invert_transform, ObservationScales
    from bookshelf_simple_experiment_ros.policy_tool_math import make_transform
    from bookshelf_simple_experiment_ros.residual_policy_math import NumpyActorBundle

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path("/home/riot/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml").read_text())
    params = lambda name: config[name]["ros__parameters"]
    target, adapter = params("calibrated_preinsert_target"), params("policy_observation_adapter")
    scales = ObservationScales(
        float(adapter["rear_to_mouth_obs_scale"]), float(adapter["front_to_back_obs_scale"]),
        float(adapter["lat_err_obs_scale"]), float(adapter["z_err_obs_scale"]),
        math.radians(float(adapter["yaw_err_obs_scale_deg"])), float(adapter["tool_to_book_obs_scale"]),
    )
    divisors = np.array([1, scales.rear_to_mouth, scales.front_to_back, scales.lateral, scales.vertical, scales.yaw, scales.tool_to_book, scales.tool_to_book, scales.tool_to_book, 1, 1, 1], dtype=np.float32)
    eef_tool = make_transform(target["eef_policy_tool_translation_xyz"], target["eef_policy_tool_quaternion_xyzw"])
    actor = NumpyActorBundle(args.actor)
    audit_runs = {run["run"]: run for run in audit["runs"] if run["run"] in RUNS}

    corrected = {}
    for name, metadata in audit_runs.items():
        rows = [json.loads(line) for line in Path(metadata["path"]).open() if json.loads(line).get("event") == "calculated"]
        slot = make_transform(metadata["accepted_frozen_slot"]["translation_xyz"], metadata["accepted_frozen_slot"]["quaternion_xyzw"])
        values = []
        for row in rows:
            base_book = _matrix(row, "T_base_book")
            base_tool = _matrix(row, "T_base_eef") @ eef_tool
            raw, _ = compute_policy_observation(
                invert_transform(slot) @ base_book, invert_transform(slot) @ base_tool,
                book_size=tuple(target["book_size_xyz"]), slot_depth=float(target["slot_depth_m"]),
                mode_observation=0.0, gripper_open=HELD_GRIPPER, scales=scales,
            )
            values.append(raw.astype(np.float64))
        corrected[name] = values

    pooled_tools = np.asarray([raw[6:9] for values in corrected.values() for raw in values])
    tool_spans = pooled_tools.max(axis=0) - pooled_tools.min(axis=0)
    tool_half_ranges = 0.5 * tool_spans
    representative = []
    for name, values in corrected.items():
        halfway = 0.5 * (values[0][1] + values[-1][1])
        midpoint = min(range(len(values)), key=lambda index: abs(values[index][1] - halfway))
        representative.extend([(name, "mid", midpoint, values[midpoint]), (name, "near_release", len(values) - 1, values[-1])])

    rows_out, summary = [], []
    rng = np.random.default_rng(args.seed)
    for profile in ("NORMAL", "STRESS"):
        for run, state, index, baseline in representative:
            actions, means, perturbations = [], [], []
            for sample in range(args.samples_per_state):
                perturbation = generate_perturbation(rng, profile, tool_half_ranges)
                raw = apply_perturbation(baseline, perturbation)
                observation = np.clip(raw.astype(np.float32) / divisors, -1.0, 1.0)
                observation[0], observation[9] = raw[0], np.clip(raw[9], 0.0, 1.0)
                _, actor_mean, action = actor.predict(observation)
                actions.append(action); means.append(actor_mean); perturbations.append(perturbation)
                row = {"profile": profile, "run": run, "state": state, "state_index": index, "sample": sample, **perturbation}
                row.update({f"raw_{i}": float(value) for i, value in enumerate(raw)})
                row.update({f"actor_mean_{i}": float(value) for i, value in enumerate(actor_mean)})
                row.update({f"action_{i}": float(value) for i, value in enumerate(action)})
                rows_out.append(row)
            actions, means = np.asarray(actions), np.asarray(means)
            release = means[:, 5]
            worst_index = int(np.argmax(release))
            summary.append({
                "profile": profile, "run": run, "state": state, "state_index": index,
                "samples": args.samples_per_state,
                "release_crossing_fraction": float(np.mean(release > 0.5)),
                "release_distribution": {"min": float(np.min(release)), "median": float(np.median(release)), "p95": float(np.percentile(release, 95)), "max": float(np.max(release))},
                "x_forward_fraction": float(np.mean(actions[:, 0] > 0)),
                "x_reverse_fraction": float(np.mean(actions[:, 0] < 0)),
                "x_clipped_fraction": float(np.mean(np.isclose(np.abs(actions[:, 0]), 1.0))),
                "motion_clipped_fraction_by_dimension": [float(value) for value in np.mean(np.isclose(np.abs(actions[:, :5]), 1.0), axis=0)],
                "worst_case": {"release_raw": float(release[worst_index]), "action": actions[worst_index].tolist(), "perturbation": perturbations[worst_index]},
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "joint_offline_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows_out[0].keys()); writer.writeheader(); writer.writerows(rows_out)

    # Closed-loop cases keep semantic gripper, since deployment maps held INSERT to it.
    isaac_rng = np.random.default_rng(args.seed + 1)
    isaac_states = []
    counters = {"NORMAL": args.normal_isaac_cases, "STRESS": args.stress_isaac_cases}
    for profile, count in counters.items():
        for case_index in range(count):
            run, _, _, baseline = representative[(case_index % 2) * 2]  # alternate A/C takeover-derived geometry
            perturbation = generate_perturbation(isaac_rng, profile, tool_half_ranges)
            perturbation["gripper_open"] = HELD_GRIPPER
            raw = apply_perturbation(corrected[run][0], perturbation)
            isaac_states.append({
                "case_name": f"{profile.lower()}_{case_index:03d}_{run[-6:]}", "profile": profile,
                "source_run": run, "perturbation": perturbation, "raw_observation": raw.tolist(),
            })
    (args.output_dir / "isaac_joint_cases.json").write_text(json.dumps({"schema_version": 1, "states": isaac_states, "tool_delta_normal_half_ranges_m": tool_half_ranges.tolist(), "tool_delta_stress_half_ranges_m": tool_spans.tolist()}, indent=2) + "\n")
    result = {"seed": args.seed, "samples_per_state_profile": args.samples_per_state, "tool_delta_observed_min_m": pooled_tools.min(axis=0).tolist(), "tool_delta_observed_max_m": pooled_tools.max(axis=0).tolist(), "tool_delta_normal_half_ranges_m": tool_half_ranges.tolist(), "tool_delta_stress_half_ranges_m": tool_spans.tolist(), "summaries": summary}
    (args.output_dir / "joint_offline_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"samples": len(rows_out), "isaac_cases": len(isaac_states), "tool_normal_half_ranges_mm": (1000 * tool_half_ranges).tolist(), "release_crossings": sum(item["release_crossing_fraction"] * item["samples"] for item in summary)}, indent=2))


if __name__ == "__main__":
    main()
