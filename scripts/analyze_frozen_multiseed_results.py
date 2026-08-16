#!/usr/bin/env python3
"""Create paper-ready statistics from aligned frozen-bank policy evaluations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


_RUN_NAME = re.compile(r"^(?P<method>.+)_seed(?P<seed>-?\d+)$")
_NORMAL_95 = 1.959963984540054
_T_975 = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118488,
    7: 2.3646242510,
    8: 2.3060041350,
    9: 2.2621571629,
    10: 2.2281388520,
}


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one sample")
    proportion = successes / total
    z2 = _NORMAL_95**2
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        _NORMAL_95
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return 100.0 * (center - radius), 100.0 * (center + radius)


def _mean_t_interval(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    critical = _T_975.get(len(values) - 1, _NORMAL_95)
    radius = critical * stdev / math.sqrt(len(values))
    return max(0.0, mean - radius), min(100.0, mean + radius)


def _exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail + 1)
    ]
    maximum = max(log_terms)
    log_probability = maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))
    log_two_sided = min(0.0, math.log(2.0) + log_probability)
    return math.exp(log_two_sided) if log_two_sided > -745.0 else 0.0


def _resolve_episodes_path(summary_path: Path, summary: dict[str, Any]) -> Path:
    value = summary.get("episodes_csv")
    if not value:
        raise ValueError(f"Missing episodes_csv: {summary_path}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def load_run(summary_path: str | Path, *, method: str, seed: int | None) -> dict[str, Any]:
    path = Path(summary_path).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    episode_count = int(summary.get("episode_count", -1))
    scenario_hash = summary.get("scenario_sha256")
    coverage = summary.get("frozen_scenario_bank_coverage") or {}
    if episode_count <= 0 or not scenario_hash:
        raise ValueError(f"Invalid frozen evaluation summary: {path}")
    if not summary.get("scenario_trace_complete", False) or not coverage.get("complete", False):
        raise ValueError(f"Incomplete frozen evaluation: {path}")

    episodes_path = _resolve_episodes_path(path, summary)
    outcomes: dict[int, str] = {}
    with episodes_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"scenario_bank_index", "outcome"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing episode columns {sorted(missing)}: {episodes_path}")
        for row in reader:
            index = int(float(row["scenario_bank_index"]))
            if index in outcomes:
                raise ValueError(f"Duplicate scenario_bank_index={index}: {episodes_path}")
            outcomes[index] = row["outcome"]

    expected_indices = set(range(episode_count))
    if set(outcomes) != expected_indices:
        raise ValueError(f"Scenario indices are incomplete or unexpected: {episodes_path}")
    success_count = sum(value == "success" for value in outcomes.values())
    lower, upper = _wilson_interval(success_count, episode_count)
    return {
        "method": method,
        "seed": seed,
        "summary": str(path),
        "episodes_csv": str(episodes_path),
        "scenario_sha256": scenario_hash,
        "episode_count": episode_count,
        "success": success_count,
        "success_pct": 100.0 * success_count / episode_count,
        "success_wilson95_pct": [lower, upper],
        "outcome_counts": summary.get("outcomes", {}),
        "outcomes_by_scenario": outcomes,
        "checkpoint_sha256": summary.get("metadata", {}).get("checkpoint_sha256"),
    }


def discover_seeded_runs(results_root: str | Path) -> list[dict[str, Any]]:
    root = Path(results_root).resolve()
    runs = []
    for summary_path in sorted(root.glob("*_seed*/summary.json")):
        match = _RUN_NAME.match(summary_path.parent.name)
        if match is None:
            continue
        runs.append(
            load_run(
                summary_path,
                method=match.group("method"),
                seed=int(match.group("seed")),
            )
        )
    if not runs:
        raise ValueError(f"No seeded frozen evaluations found in {root}")
    return runs


def _validate_alignment(runs: list[dict[str, Any]]) -> tuple[str, int]:
    hashes = {run["scenario_sha256"] for run in runs}
    counts = {run["episode_count"] for run in runs}
    if len(hashes) != 1 or len(counts) != 1:
        raise ValueError("Frozen evaluations are not aligned to one scenario bank")
    return next(iter(hashes)), next(iter(counts))


def _paired_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_outcomes = left["outcomes_by_scenario"]
    right_outcomes = right["outcomes_by_scenario"]
    both_success = left_only = right_only = both_fail = 0
    for index in left_outcomes:
        left_success = left_outcomes[index] == "success"
        right_success = right_outcomes[index] == "success"
        if left_success and right_success:
            both_success += 1
        elif left_success:
            left_only += 1
        elif right_success:
            right_only += 1
        else:
            both_fail += 1
    return {
        "left": f"{left['method']}_seed{left['seed']}",
        "right": f"{right['method']}_seed{right['seed']}",
        "left_success_pct": left["success_pct"],
        "right_success_pct": right["success_pct"],
        "right_minus_left_percentage_points": right["success_pct"] - left["success_pct"],
        "both_success": both_success,
        "left_only_success": left_only,
        "right_only_success": right_only,
        "both_fail": both_fail,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(left_only, right_only),
    }


def analyze_runs(
    seeded_runs: list[dict[str, Any]], baseline_run: dict[str, Any] | None = None
) -> dict[str, Any]:
    all_runs = list(seeded_runs)
    if baseline_run is not None:
        all_runs.append(baseline_run)
    scenario_hash, scenario_count = _validate_alignment(all_runs)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in seeded_runs:
        grouped.setdefault(run["method"], []).append(run)
    method_summaries = []
    consensus = []
    for method, runs in sorted(grouped.items()):
        runs.sort(key=lambda value: value["seed"])
        rates = [run["success_pct"] for run in runs]
        interval = _mean_t_interval(rates)
        method_summaries.append(
            {
                "method": method,
                "training_seed_count": len(runs),
                "training_seeds": [run["seed"] for run in runs],
                "mean_success_pct": statistics.mean(rates),
                "sample_stdev_success_percentage_points": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                "mean_t95_pct": list(interval) if interval is not None else None,
                "minimum_success_pct": min(rates),
                "maximum_success_pct": max(rates),
            }
        )
        successes_per_scenario = [
            sum(run["outcomes_by_scenario"][index] == "success" for run in runs)
            for index in range(scenario_count)
        ]
        consensus.append(
            {
                "method": method,
                "all_seeds_succeed": sum(value == len(runs) for value in successes_per_scenario),
                "all_seeds_fail": sum(value == 0 for value in successes_per_scenario),
                "mixed_seed_outcomes": sum(0 < value < len(runs) for value in successes_per_scenario),
            }
        )

    paired = []
    for left_method, right_method in itertools.combinations(sorted(grouped), 2):
        left_by_seed = {run["seed"]: run for run in grouped[left_method]}
        right_by_seed = {run["seed"]: run for run in grouped[right_method]}
        for seed in sorted(set(left_by_seed).intersection(right_by_seed)):
            paired.append(_paired_comparison(left_by_seed[seed], right_by_seed[seed]))
    if baseline_run is not None:
        for run in sorted(seeded_runs, key=lambda value: (value["method"], value["seed"])):
            paired.append(_paired_comparison(baseline_run, run))

    serializable_runs = [
        {key: value for key, value in run.items() if key != "outcomes_by_scenario"}
        for run in all_runs
    ]
    return {
        "schema_version": 1,
        "scenario_sha256": scenario_hash,
        "scenario_count": scenario_count,
        "runs": serializable_runs,
        "method_summaries": method_summaries,
        "seed_consensus": consensus,
        "paired_comparisons": paired,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "multiseed_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output / "run_summary.csv",
        report["runs"],
        ["method", "seed", "episode_count", "success", "success_pct", "checkpoint_sha256", "summary"],
    )
    _write_csv(
        output / "paired_comparisons.csv",
        report["paired_comparisons"],
        [
            "left",
            "right",
            "left_success_pct",
            "right_success_pct",
            "right_minus_left_percentage_points",
            "both_success",
            "left_only_success",
            "right_only_success",
            "both_fail",
            "mcnemar_exact_two_sided_p",
        ],
    )

    lines = [
        "# Frozen Multi-Seed Evaluation",
        "",
        f"Scenario count: {report['scenario_count']}",
        f"Scenario SHA256: `{report['scenario_sha256']}`",
        "",
        "## Method Summary",
        "",
        "| Method | Seeds | Mean success | Sample SD | Range | Training-seed 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["method_summaries"]:
        interval = row["mean_t95_pct"]
        interval_text = "n/a" if interval is None else f"{interval[0]:.2f}-{interval[1]:.2f}%"
        lines.append(
            f"| {row['method']} | {row['training_seed_count']} | {row['mean_success_pct']:.2f}% | "
            f"{row['sample_stdev_success_percentage_points']:.2f} pp | "
            f"{row['minimum_success_pct']:.2f}-{row['maximum_success_pct']:.2f}% | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "## Individual Runs",
            "",
            "| Method | Seed | Success | Wilson 95% CI | Drop | Timeout |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["runs"]:
        lower, upper = row["success_wilson95_pct"]
        outcomes = row["outcome_counts"]
        lines.append(
            f"| {row['method']} | {row['seed'] if row['seed'] is not None else 'n/a'} | "
            f"{row['success']}/{row['episode_count']} ({row['success_pct']:.2f}%) | "
            f"{lower:.2f}-{upper:.2f}% | {outcomes.get('drop', 0)} | {outcomes.get('timeout', 0)} |"
        )
    (output / "paper_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--baseline-name", default="nominal_only")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    runs = discover_seeded_runs(args.results_root)
    baseline = None
    if args.baseline_summary is not None:
        baseline = load_run(args.baseline_summary, method=args.baseline_name, seed=None)
    report = analyze_runs(runs, baseline)
    output_dir = args.output_dir or (args.results_root / "analysis")
    write_reports(report, output_dir)
    for method in report["method_summaries"]:
        print(
            f"{method['method']:16s} mean={method['mean_success_pct']:.2f}% "
            f"sd={method['sample_stdev_success_percentage_points']:.2f} pp "
            f"seeds={method['training_seeds']}"
        )
    print(f"Reports: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
