import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/analyze_frozen_multiseed_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_frozen_multiseed_results_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def _write_run(root, name, outcomes, scenario_hash="bank-hash"):
    directory = root / name
    directory.mkdir()
    episodes = directory / "episodes.csv"
    with episodes.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_bank_index", "outcome"])
        writer.writeheader()
        for index, outcome in enumerate(outcomes):
            writer.writerow({"scenario_bank_index": index, "outcome": outcome})
    summary = directory / "summary.json"
    counts = {value: outcomes.count(value) for value in set(outcomes)}
    summary.write_text(
        json.dumps(
            {
                "episode_count": len(outcomes),
                "episodes_csv": str(episodes),
                "scenario_sha256": scenario_hash,
                "scenario_trace_complete": True,
                "frozen_scenario_bank_coverage": {"complete": True},
                "outcomes": counts,
                "metadata": {"checkpoint_sha256": f"checkpoint-{name}"},
            }
        )
    )
    return summary


def test_analysis_reports_seed_statistics_and_paired_transitions(tmp_path):
    _write_run(tmp_path, "ppo_only_seed42", ["drop", "drop", "drop", "drop"])
    _write_run(tmp_path, "ppo_only_seed123", ["drop", "drop", "drop", "drop"])
    _write_run(tmp_path, "residual_ppo_seed42", ["success", "success", "drop", "success"])
    _write_run(tmp_path, "residual_ppo_seed123", ["success", "drop", "success", "success"])
    baseline = _write_run(tmp_path, "nominal", ["success", "drop", "drop", "drop"])

    runs = ANALYSIS.discover_seeded_runs(tmp_path)
    baseline_run = ANALYSIS.load_run(baseline, method="nominal_only", seed=None)
    report = ANALYSIS.analyze_runs(runs, baseline_run)

    methods = {row["method"]: row for row in report["method_summaries"]}
    assert methods["ppo_only"]["mean_success_pct"] == 0.0
    assert methods["residual_ppo"]["mean_success_pct"] == 75.0
    comparison = next(
        row
        for row in report["paired_comparisons"]
        if row["left"] == "ppo_only_seed42" and row["right"] == "residual_ppo_seed42"
    )
    assert comparison["right_only_success"] == 3
    assert comparison["left_only_success"] == 0
    assert comparison["right_minus_left_percentage_points"] == 75.0


def test_analysis_rejects_misaligned_banks(tmp_path):
    first = _write_run(tmp_path, "ppo_only_seed42", ["drop", "drop"], "bank-a")
    second = _write_run(tmp_path, "residual_ppo_seed42", ["success", "drop"], "bank-b")
    runs = [
        ANALYSIS.load_run(first, method="ppo_only", seed=42),
        ANALYSIS.load_run(second, method="residual_ppo", seed=42),
    ]
    with pytest.raises(ValueError, match="not aligned"):
        ANALYSIS.analyze_runs(runs)


def test_analysis_rejects_duplicate_scenario_indices(tmp_path):
    summary = _write_run(tmp_path, "ppo_only_seed42", ["drop", "drop"])
    episodes = summary.parent / "episodes.csv"
    episodes.write_text("scenario_bank_index,outcome\n0,drop\n0,drop\n")
    with pytest.raises(ValueError, match="Duplicate"):
        ANALYSIS.load_run(summary, method="ppo_only", seed=42)
