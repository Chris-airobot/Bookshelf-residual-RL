#!/usr/bin/env python3
"""Audit and summarize paired bookshelf model-selection evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PROFILES = (
    "nominal", "physical_like", "gripper_only", "observation_moderate",
    "combined_moderate", "strong_tail", "combined_moderate_actuation",
)


def mean(values):
    return statistics.fmean(values) if values else math.nan


def median(values):
    return statistics.median(values) if values else math.nan


def rate(values):
    return mean([float(value) for value in values])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    labels = [line.split("\t", 1)[0] for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = []
    scenario_hash_by_profile = {}
    for label in labels:
        for profile in PROFILES:
            directory = args.root / label / profile
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            coverage = summary.get("frozen_scenario_bank_coverage") or {}
            if not coverage.get("complete") or int(summary["episode_count"]) != 512:
                raise RuntimeError(f"incomplete paired evaluation: {directory}")
            bank_hash = summary["metadata"]["frozen_scenario_bank"]["scenario_sha256"]
            previous = scenario_hash_by_profile.setdefault(profile, bank_hash)
            if bank_hash != previous:
                raise RuntimeError(f"scenario bank mismatch for {profile}: {directory}")
            with (directory / "episodes.csv").open(newline="", encoding="utf-8") as stream:
                episodes = {int(row["episode_index"]): row for row in csv.DictReader(stream)}
            with (directory / "policy_robustness_metrics.csv").open(newline="", encoding="utf-8") as stream:
                robust = list(csv.DictReader(stream))
            released = [row for row in robust if row["failure_to_release"] == "0"]
            rear = [float(row["rear_to_mouth_at_release_m"]) for row in released]
            front = [float(row["front_to_back_at_release_m"]) for row in released]
            depth = [float(row["insertion_depth_at_release_m"]) for row in released]
            actions = [float(row["release_action"]) for row in released]
            result = {
                "policy": label,
                "profile": profile,
                "episodes": len(robust),
                "success_rate": rate(int(row["success"]) for row in robust),
                "premature_release_rate": rate(
                    int(row["premature_release"] or 0) for row in robust
                ),
                "never_release_rate": rate(int(row["failure_to_release"]) for row in robust),
                "rear_to_mouth_release_mean_m": mean(rear),
                "rear_to_mouth_release_median_m": median(rear),
                "front_to_back_release_mean_m": mean(front),
                "front_to_back_release_median_m": median(front),
                "insertion_depth_release_mean_m": mean(depth),
                "insertion_depth_release_median_m": median(depth),
                "post_push_success_rate": rate(int(row["post_push_success"]) for row in robust),
                "episode_reward_mean": mean(float(episodes[int(row["episode_index"])]["episode_reward"]) for row in robust),
                "release_action_mean": mean(actions),
                "release_action_margin_mean": mean([value - 0.5 for value in actions]),
                "released_rear_outside_5mm_rate": rate(value < -0.005 for value in rear),
                "scenario_bank_sha256": bank_hash,
            }
            rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detailed = args.output.with_suffix(".profiles.csv")
    with detailed.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    by_key = {(row["policy"], row["profile"]): row for row in rows}
    compact = []
    for label in labels:
        nominal = by_key[label, "nominal"]
        physical = by_key[label, "physical_like"]
        combined = by_key[label, "combined_moderate"]
        compact.append({
            "policy": label,
            "nominal_success": nominal["success_rate"],
            "physical_like_success": physical["success_rate"],
            "combined_success": combined["success_rate"],
            "combined_premature_release": combined["premature_release_rate"],
            "combined_never_release": combined["never_release_rate"],
            "combined_median_rear_to_mouth_m": combined["rear_to_mouth_release_median_m"],
            "combined_post_push_success": combined["post_push_success_rate"],
        })
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact[0]))
        writer.writeheader(); writer.writerows(compact)
    print(f"Wrote {len(compact)} policies to {args.output}; paired profiles to {detailed}")


if __name__ == "__main__":
    main()
