import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/run_frozen_clearance_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_frozen_clearance_sweep_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def _model_pair(root, method, seed):
    directory = root / f"{method}_seed{seed}"
    directory.mkdir(parents=True)
    (directory / "model.zip").write_bytes(b"model")
    (directory / "model_vecnormalize.pkl").write_bytes(b"stats")


def test_build_runs_contains_nominal_and_seeded_policy_pairs(tmp_path):
    for method in ("ppo_only", "residual_ppo"):
        for seed in (42, 123):
            _model_pair(tmp_path, method, seed)
    runs = SWEEP.build_evaluation_runs(
        tmp_path,
        seeds=[42, 123],
        methods=["nominal_only", "ppo_only", "residual_ppo"],
    )
    assert [run["name"] for run in runs] == [
        "nominal_only",
        "ppo_only_seed42",
        "ppo_only_seed123",
        "residual_ppo_seed42",
        "residual_ppo_seed123",
    ]
    assert runs[0]["checkpoint"] is None
    assert runs[-1]["task"] == "Bookshelf-Residual-Direct-v0"


def test_build_runs_fails_closed_on_missing_normalization(tmp_path):
    directory = tmp_path / "ppo_only_seed42"
    directory.mkdir()
    (directory / "model.zip").write_bytes(b"model")
    with pytest.raises(FileNotFoundError, match="Missing checkpoint pair"):
        SWEEP.build_evaluation_runs(tmp_path, [42], ["ppo_only"])


def test_completed_summary_must_match_full_bank(tmp_path):
    bank = {"scenario_count": 2, "scenario_sha256": "correct"}
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "episode_count": 2,
                "scenario_sha256": "correct",
                "scenario_trace_complete": True,
                "frozen_scenario_bank_coverage": {"complete": True},
                "metadata": {
                    "frozen_scenario_bank": {
                        "scenario_count": 2,
                        "scenario_sha256": "correct",
                    }
                },
            }
        )
    )
    assert SWEEP.completed_summary_matches_bank(summary, bank) is True
    bank["scenario_sha256"] = "different"
    assert SWEEP.completed_summary_matches_bank(summary, bank) is False


def test_clearance_labels_are_stable():
    assert SWEEP.clearance_label(0.001) == "clearance_1p0mm"
    assert SWEEP.clearance_label(0.0035) == "clearance_3p5mm"


def test_clearance_aggregation_uses_training_seeds_as_replicates():
    rows = [
        {"clearance_m": 0.003, "method": "residual_ppo", "success_pct": 80.0},
        {"clearance_m": 0.003, "method": "residual_ppo", "success_pct": 100.0},
        {"clearance_m": 0.003, "method": "nominal_only", "success_pct": 25.0},
    ]
    aggregate = {
        (row["method"], row["clearance_m"]): row
        for row in SWEEP.aggregate_clearance_rows(rows)
    }
    residual = aggregate[("residual_ppo", 0.003)]
    assert residual["training_seed_count"] == 2
    assert residual["mean_success_pct"] == 90.0
    assert residual["sample_stdev_success_percentage_points"] == pytest.approx(14.1421356)
    assert aggregate[("nominal_only", 0.003)]["sample_stdev_success_percentage_points"] == 0.0


def test_startup_failure_retries_when_no_summary_exists(tmp_path, monkeypatch):
    summary = tmp_path / "summary.json"
    bank = {"scenario_count": 2, "scenario_sha256": "bank"}
    calls = []

    def fake_stream(command, log_path):
        calls.append(log_path)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)
        summary.write_text(
            json.dumps(
                {
                    "episode_count": 2,
                    "scenario_trace_complete": True,
                    "frozen_scenario_bank_coverage": {"complete": True},
                    "metadata": {
                        "frozen_scenario_bank": {
                            "scenario_count": 2,
                            "scenario_sha256": "bank",
                        }
                    },
                }
            )
        )

    monkeypatch.setattr(SWEEP, "_stream_command", fake_stream)
    monkeypatch.setattr(SWEEP.time, "sleep", lambda _: None)
    SWEEP._run_evaluation_with_retries(
        command=["fake"],
        log_path=tmp_path / "run.log",
        summary_path=summary,
        bank=bank,
        retries=2,
        retry_delay_s=30.0,
    )
    assert [path.name for path in calls] == ["run.log", "run.retry1.log"]


def test_successful_process_without_complete_summary_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(SWEEP, "_stream_command", lambda command, log_path: None)
    with pytest.raises(ValueError, match="without a complete summary"):
        SWEEP._run_evaluation_with_retries(
            command=["fake"],
            log_path=tmp_path / "run.log",
            summary_path=tmp_path / "summary.json",
            bank={"scenario_count": 2, "scenario_sha256": "bank"},
            retries=0,
            retry_delay_s=0.0,
        )
