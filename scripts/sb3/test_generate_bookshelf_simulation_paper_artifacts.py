import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/generate_bookshelf_simulation_paper_artifacts.py"
SPEC = importlib.util.spec_from_file_location("paper_artifacts_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARTIFACTS)


def test_method_series_is_sorted_and_keeps_seed_sd():
    rows = [
        {
            "clearance_mm": "5.0",
            "method": "residual_ppo",
            "mean_success_pct": "96.0",
            "sample_stdev_success_percentage_points": "2.0",
        },
        {
            "clearance_mm": "1.0",
            "method": "residual_ppo",
            "mean_success_pct": "10.0",
            "sample_stdev_success_percentage_points": "4.0",
        },
    ]
    assert ARTIFACTS.build_method_series(rows, x_field="clearance_mm") == {
        "residual_ppo": {
            "x": [1.0, 5.0],
            "mean": [10.0, 96.0],
            "sd": [4.0, 2.0],
        }
    }


def test_pivot_requires_every_method():
    rows = [
        {
            "offset_scale": "1.0",
            "offset_regime": "in_distribution",
            "method": "residual_ppo",
            "mean_success_pct": "90.0",
            "sample_stdev_success_percentage_points": "3.0",
        }
    ]
    with pytest.raises(ValueError, match="missing methods"):
        ARTIFACTS._pivot_sweep_rows(
            rows, x_field="offset_scale", regime_field="offset_regime"
        )


def test_main_table_uses_seeds_as_replicates_and_wilson_for_nominal():
    analysis = {
        "scenario_count": 2000,
        "runs": [
            {
                "method": "nominal_only",
                "seed": None,
                "episode_count": 2000,
                "success_pct": 28.85,
                "success_wilson95_pct": [26.91, 30.87],
            }
        ],
        "method_summaries": [
            {
                "method": "ppo_only",
                "training_seeds": [42, 123, 2026],
                "mean_success_pct": 0.0,
                "sample_stdev_success_percentage_points": 0.0,
                "minimum_success_pct": 0.0,
                "maximum_success_pct": 0.0,
                "mean_t95_pct": [0.0, 0.0],
            },
            {
                "method": "residual_ppo",
                "training_seeds": [42, 123, 2026],
                "mean_success_pct": 90.93,
                "sample_stdev_success_percentage_points": 2.8,
                "minimum_success_pct": 89.05,
                "maximum_success_pct": 94.15,
                "mean_t95_pct": [83.98, 97.89],
            },
        ],
    }
    rows = ARTIFACTS.build_main_3mm_rows(analysis)
    assert rows[0]["interval_type"] == "Wilson episode CI"
    assert rows[2]["training_seeds"] == "42,123,2026"
    assert rows[2]["sample_stdev_pp"] == 2.8


def test_paired_table_extracts_nominal_residual_effects():
    analysis = {
        "paired_comparisons": [
            {
                "left": "nominal_only_seedNone",
                "right": "residual_ppo_seed42",
                "left_success_pct": 28.85,
                "right_success_pct": 89.6,
                "right_minus_left_percentage_points": 60.75,
                "left_only_success": 61,
                "right_only_success": 1276,
                "both_success": 516,
                "both_fail": 147,
                "mcnemar_exact_two_sided_p": 1.7e-296,
            }
        ]
    }
    row = ARTIFACTS.build_nominal_residual_paired_rows(analysis)[0]
    assert row["residual_seed"] == 42
    assert row["gain_percentage_points"] == 60.75


def test_p_value_formatter_handles_underflow():
    assert ARTIFACTS._format_p(0.0) == r"$<10^{-300}$"
    assert "10^{-5}" in ARTIFACTS._format_p(2.0e-5)
