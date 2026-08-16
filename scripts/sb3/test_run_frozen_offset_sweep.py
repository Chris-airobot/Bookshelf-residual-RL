import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "run_frozen_offset_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_frozen_offset_sweep_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def _bank(scale, seed=42):
    noise = SWEEP.BANK_GENERATOR.scaled_training_reset_noise(scale)
    return SWEEP.BANK_GENERATOR.MODULE.generate_frozen_scenario_bank(
        scenario_count=12,
        seed=seed,
        slot_clearance_min=0.003,
        slot_clearance_max=0.003,
        slot_pitch=0.034,
        row_book_count=10,
        side_book_merge_probability=0.35,
        arm_joint_noise=noise[0],
        grasp_x_jitter=noise[1],
        grasp_y_jitter=noise[2],
        grasp_z_jitter=noise[3],
        grasp_yaw_jitter=noise[4],
    )


def test_training_noise_scale_matches_final_curriculum_boundary():
    assert SWEEP.BANK_GENERATOR.scaled_training_reset_noise(1.0) == (
        math.radians(3.0),
        0.008,
        0.006,
        0.003,
        math.radians(8.0),
    )
    half = SWEEP.BANK_GENERATOR.scaled_training_reset_noise(0.5)
    assert half[1:4] == (0.004, 0.003, 0.0015)
    with pytest.raises(ValueError, match="non-negative"):
        SWEEP.BANK_GENERATOR.scaled_training_reset_noise(-0.1)


def test_legacy_generator_modes_remain_compatible():
    assert SWEEP.BANK_GENERATOR.resolve_reset_noise(
        old_reset_noise=False, reset_noise_scale=None
    ) == SWEEP.BANK_GENERATOR.CURRENT_RESET_NOISE
    assert SWEEP.BANK_GENERATOR.resolve_reset_noise(
        old_reset_noise=True, reset_noise_scale=None
    ) == SWEEP.BANK_GENERATOR.FINAL_TRAINING_RESET_NOISE
    with pytest.raises(ValueError, match="mutually exclusive"):
        SWEEP.BANK_GENERATOR.resolve_reset_noise(
            old_reset_noise=True, reset_noise_scale=1.0
        )


def test_generated_offset_banks_are_exactly_paired_and_scaled():
    banks = {scale: _bank(scale) for scale in (0.0, 0.5, 1.0, 1.5)}
    SWEEP.validate_paired_offset_banks(banks)
    assert [
        scenario["missing_book_index"] for scenario in banks[0.0]["scenarios"]
    ] == [scenario["missing_book_index"] for scenario in banks[1.5]["scenarios"]]


def test_pair_validation_rejects_non_noise_drift():
    banks = {0.5: _bank(0.5), 1.0: _bank(1.0)}
    original = banks[1.0]["scenarios"][3]["missing_book_index"]
    banks[1.0]["scenarios"][3]["missing_book_index"] = (original + 1) % 10
    with pytest.raises(ValueError, match="non-noise fields differ"):
        SWEEP.validate_paired_offset_banks(banks)


def test_offset_labels_and_regimes_are_explicit():
    assert SWEEP.offset_scale_label(0.0) == "offset_0p00x"
    assert SWEEP.offset_scale_label(1.25) == "offset_1p25x"
    assert SWEEP.offset_regime(1.0) == "in_distribution"
    assert SWEEP.offset_regime(1.01) == "out_of_distribution"


def test_offset_aggregation_uses_training_seeds_as_replicates():
    rows = [
        {"offset_scale": 1.25, "method": "residual_ppo", "success_pct": 70.0},
        {"offset_scale": 1.25, "method": "residual_ppo", "success_pct": 90.0},
        {"offset_scale": 1.25, "method": "nominal_only", "success_pct": 20.0},
    ]
    aggregate = {
        (row["method"], row["offset_scale"]): row
        for row in SWEEP.aggregate_offset_rows(rows)
    }
    residual = aggregate[("residual_ppo", 1.25)]
    assert residual["offset_regime"] == "out_of_distribution"
    assert residual["mean_success_pct"] == 80.0
    assert residual["sample_stdev_success_percentage_points"] == pytest.approx(
        14.1421356
    )


def test_summary_writer_emits_machine_and_paper_tables(tmp_path):
    rows = []
    for scale in (1.0, 1.25):
        for method, rates in (
            ("nominal_only", [25.0]),
            ("ppo_only", [0.0, 0.0]),
            ("residual_ppo", [80.0, 90.0]),
        ):
            for seed_index, rate in enumerate(rates):
                rows.append(
                    {
                        "offset_scale": scale,
                        "offset_regime": SWEEP.offset_regime(scale),
                        "clearance_m": 0.003,
                        "method": method,
                        "training_seed": None if method == "nominal_only" else seed_index,
                        "success": int(rate),
                        "episodes": 100,
                        "success_pct": rate,
                        "drop": 100 - int(rate),
                        "timeout": 0,
                        "scenario_sha256": "hash",
                        "checkpoint_sha256": None,
                        "summary": "summary.json",
                    }
                )
    SWEEP._write_offset_summary(tmp_path, rows)
    markdown = (tmp_path / "offset_paper_results.md").read_text()
    latex = (tmp_path / "offset_paper_results.tex").read_text()
    assert "1.25x | out_of_distribution" in markdown
    assert "Residual PPO" in latex
    assert "1.25$\\times$ & OOD" in latex
