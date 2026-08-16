#!/usr/bin/env python3
"""Run paired frozen-bank evaluations across initial-offset severities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_eval_scenario_bank as BANK_GENERATOR  # noqa: E402
import run_frozen_clearance_sweep as CLEARANCE_SWEEP  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OFFSET_SCALES = (0.0, 0.5, 1.0, 1.25, 1.5)
DEFAULT_SEEDS = CLEARANCE_SWEEP.DEFAULT_SEEDS
NOISE_FIELDS = (
    *[f"joint_noise_{index}" for index in range(1, 8)],
    "grasp_jitter_x",
    "grasp_jitter_y",
    "grasp_jitter_z",
    "grasp_jitter_yaw",
)


def offset_scale_label(scale: float) -> str:
    return f"offset_{float(scale):.2f}x".replace(".", "p")


def offset_regime(scale: float) -> str:
    return "in_distribution" if float(scale) <= 1.0 else "out_of_distribution"


def expected_generation_noise(scale: float) -> dict[str, float]:
    values = BANK_GENERATOR.scaled_training_reset_noise(scale)
    return dict(zip(BANK_GENERATOR.RESET_NOISE_KEYS, values))


def _validate_bank_configuration(
    bank: dict[str, Any], *, scale: float, clearance: float, scenario_count: int
) -> None:
    if int(bank.get("scenario_count", -1)) != int(scenario_count):
        raise ValueError("Bank scenario count does not match the offset sweep")
    clearances = {float(scenario["slot_clearance"]) for scenario in bank.get("scenarios", [])}
    if len(clearances) != 1 or not math.isclose(
        next(iter(clearances)), clearance, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("Bank clearance does not match the offset sweep")
    generation = bank.get("source", {}).get("generation") or {}
    for key, expected in expected_generation_noise(scale).items():
        observed = generation.get(key)
        if observed is None or not math.isclose(
            float(observed), expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"Bank reset-noise mismatch for scale={scale:g}, {key}: "
                f"expected {expected}, observed {observed}"
            )


def validate_paired_offset_banks(banks_by_scale: dict[float, dict[str, Any]]) -> None:
    """Prove banks share scenarios and differ only by proportional reset noise."""

    if not banks_by_scale:
        raise ValueError("No offset banks were provided")
    ordered = sorted((float(scale), bank) for scale, bank in banks_by_scale.items())
    reference_scale, reference_bank = next(
        ((scale, bank) for scale, bank in ordered if scale > 0.0), ordered[0]
    )
    reference_scenarios = reference_bank["scenarios"]
    for scale, bank in ordered:
        scenarios = bank["scenarios"]
        if len(scenarios) != len(reference_scenarios):
            raise ValueError("Offset banks have different scenario counts")
        for index, (reference, candidate) in enumerate(zip(reference_scenarios, scenarios)):
            reference_fixed = {
                key: value for key, value in reference.items() if key not in NOISE_FIELDS
            }
            candidate_fixed = {
                key: value for key, value in candidate.items() if key not in NOISE_FIELDS
            }
            if candidate_fixed != reference_fixed:
                raise ValueError(
                    f"Offset banks are not paired at scenario index {index}: "
                    "non-noise fields differ"
                )
            for field in NOISE_FIELDS:
                value = float(candidate[field])
                if scale == 0.0:
                    if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
                        raise ValueError(f"Zero-scale bank contains nonzero {field}")
                    continue
                reference_value = float(reference[field])
                expected = reference_value * scale / reference_scale
                # Canonical scenario values are rounded to 10 decimal places.
                if not math.isclose(value, expected, rel_tol=1.0e-10, abs_tol=5.0e-10):
                    raise ValueError(
                        f"Offset banks are not proportionally paired at scenario {index}, {field}"
                    )


def _load_or_generate_bank(
    *,
    bank_path: Path,
    offset_scale: float,
    clearance: float,
    scenario_count: int,
    bank_seed: int,
    dry_run: bool,
) -> dict[str, Any] | None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/generate_eval_scenario_bank.py"),
        str(bank_path),
        "--scenarios",
        str(scenario_count),
        "--seed",
        str(bank_seed),
        "--slot-clearance",
        str(clearance),
        "--reset-noise-scale",
        str(offset_scale),
    ]
    if dry_run:
        print(f"BANK: {shlex.join(command)}")
        return None
    if not bank_path.is_file():
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    _validate_bank_configuration(
        bank,
        scale=offset_scale,
        clearance=clearance,
        scenario_count=scenario_count,
    )
    return bank


def aggregate_offset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((float(row["offset_scale"]), row["method"]), []).append(row)
    aggregate = []
    for (scale, method), method_rows in sorted(grouped.items()):
        rates = [float(row["success_pct"]) for row in method_rows]
        aggregate.append(
            {
                "offset_scale": scale,
                "offset_regime": offset_regime(scale),
                "method": method,
                "training_seed_count": len(method_rows),
                "mean_success_pct": statistics.mean(rates),
                "sample_stdev_success_percentage_points": (
                    statistics.stdev(rates) if len(rates) > 1 else 0.0
                ),
                "minimum_success_pct": min(rates),
                "maximum_success_pct": max(rates),
            }
        )
    return aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_offset_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    (output_root / "offset_sweep_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_root / "offset_sweep_summary.csv",
        rows,
        [
            "offset_scale",
            "offset_regime",
            "clearance_m",
            "method",
            "training_seed",
            "success",
            "episodes",
            "success_pct",
            "drop",
            "timeout",
            "scenario_sha256",
            "checkpoint_sha256",
            "summary",
        ],
    )
    aggregate = aggregate_offset_rows(rows)
    (output_root / "offset_method_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_root / "offset_method_summary.csv",
        aggregate,
        list(aggregate[0]),
    )

    methods = sorted({row["method"] for row in aggregate})
    by_key = {(row["offset_scale"], row["method"]): row for row in aggregate}
    lines = [
        "# Initial-Offset Robustness Sweep",
        "",
        "Offset scale 1.0x is the maximum final-training reset randomization.",
        "Values above 1.0x are out of distribution.",
        "",
        "| Offset scale | Regime | " + " | ".join(methods) + " |",
        "|---:|---|" + "---:|" * len(methods),
    ]
    for scale in sorted({row["offset_scale"] for row in aggregate}):
        values = []
        for method in methods:
            row = by_key[(scale, method)]
            if row["training_seed_count"] > 1:
                values.append(
                    f"{row['mean_success_pct']:.2f} +/- "
                    f"{row['sample_stdev_success_percentage_points']:.2f}%"
                )
            else:
                values.append(f"{row['mean_success_pct']:.2f}%")
        lines.append(
            f"| {scale:.2f}x | {offset_regime(scale)} | " + " | ".join(values) + " |"
        )
    (output_root / "offset_paper_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    latex_names = {
        "nominal_only": "Nominal only",
        "ppo_only": "PPO only",
        "residual_ppo": "Residual PPO",
    }
    latex = [
        r"\begin{tabular}{ll" + "r" * len(methods) + "}",
        r"\toprule",
        "Offset & Regime & "
        + " & ".join(latex_names.get(method, method.replace("_", r"\_")) for method in methods)
        + r" \\",
        r"\midrule",
    ]
    for scale in sorted({row["offset_scale"] for row in aggregate}):
        values = []
        for method in methods:
            row = by_key[(scale, method)]
            if row["training_seed_count"] > 1:
                values.append(
                    f"{row['mean_success_pct']:.2f} $\\pm$ "
                    f"{row['sample_stdev_success_percentage_points']:.2f}"
                )
            else:
                values.append(f"{row['mean_success_pct']:.2f}")
        regime = "ID" if scale <= 1.0 else "OOD"
        latex.append(f"{scale:.2f}$\\times$ & {regime} & " + " & ".join(values) + r" \\")
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Values are success percentages; uncertainty is sample SD across training seeds.",
        ]
    )
    (output_root / "offset_paper_results.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def _run_plotter(output_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/plot_frozen_offset_sweep.py"),
            str(output_root / "offset_method_summary.csv"),
            "--output-stem",
            str(output_root / "offset_robustness"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def run_sweep(args: argparse.Namespace) -> None:
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    panda_usd = args.panda_usd.expanduser().resolve()
    isaac_python = args.isaac_python.expanduser().resolve()
    if not panda_usd.is_file():
        raise FileNotFoundError(f"Franka USD is missing: {panda_usd}")
    if not isaac_python.is_file():
        raise FileNotFoundError(f"Isaac Python launcher is missing: {isaac_python}")
    runs = CLEARANCE_SWEEP.build_evaluation_runs(model_root, args.seeds, args.methods)
    output_root.mkdir(parents=True, exist_ok=True)

    training_noise = dict(
        zip(
            BANK_GENERATOR.RESET_NOISE_KEYS,
            BANK_GENERATOR.FINAL_TRAINING_RESET_NOISE,
        )
    )
    manifest = {
        "schema_version": 1,
        "kind": "bookshelf_frozen_initial_offset_sweep",
        "offset_scales": args.offset_scales,
        "training_boundary_scale": 1.0,
        "training_reset_noise_maxima": training_noise,
        "slot_clearance_m": args.slot_clearance,
        "scenario_count_per_scale": args.scenarios,
        "bank_seed": args.bank_seed,
        "evaluation_seed": args.evaluation_seed,
        "num_envs": args.num_envs,
        "pairing": "same seed and normalized random draws across offset scales",
        "inter_run_delay_s": args.inter_run_delay_s,
        "startup_retries": args.startup_retries,
        "retry_delay_s": args.retry_delay_s,
        "runs": runs,
    }
    (output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    banks_by_scale: dict[float, dict[str, Any]] = {}
    for scale in args.offset_scales:
        bank = _load_or_generate_bank(
            bank_path=output_root / offset_scale_label(scale) / "frozen_bank.json",
            offset_scale=scale,
            clearance=args.slot_clearance,
            scenario_count=args.scenarios,
            bank_seed=args.bank_seed,
            dry_run=args.dry_run,
        )
        if bank is not None:
            banks_by_scale[scale] = bank
    if not args.dry_run:
        validate_paired_offset_banks(banks_by_scale)
        print("PASS: offset banks are scenario-paired and proportionally scaled")

    summary_rows = []
    for scale in args.offset_scales:
        label = offset_scale_label(scale)
        scale_root = output_root / label
        bank_path = scale_root / "frozen_bank.json"
        bank = banks_by_scale.get(scale)
        summaries = []
        for run in runs:
            run_output = scale_root / run["name"]
            summary_path = run_output / "summary.json"
            command = CLEARANCE_SWEEP._play_command(
                isaac_python=isaac_python,
                panda_usd=panda_usd,
                bank_path=bank_path,
                output_dir=run_output,
                run=run,
                clearance=args.slot_clearance,
                num_envs=args.num_envs,
                evaluation_seed=args.evaluation_seed,
            )
            if args.dry_run:
                print(f"RUN: {shlex.join(command)}")
                continue
            assert bank is not None
            if CLEARANCE_SWEEP.completed_summary_matches_bank(summary_path, bank):
                print(f"SKIP: complete {label}/{run['name']}")
            elif summary_path.exists():
                raise ValueError(
                    f"Existing summary does not match the requested bank: {summary_path}"
                )
            else:
                print(f"RUN: {label}/{run['name']}")
                CLEARANCE_SWEEP._run_evaluation_with_retries(
                    command=command,
                    log_path=scale_root / f"{run['name']}.log",
                    summary_path=summary_path,
                    bank=bank,
                    retries=args.startup_retries,
                    retry_delay_s=args.retry_delay_s,
                )
                if args.inter_run_delay_s > 0.0:
                    print(
                        f"COOLDOWN: {args.inter_run_delay_s:.1f}s before the next Isaac launch"
                    )
                    CLEARANCE_SWEEP.time.sleep(args.inter_run_delay_s)
            summaries.append(summary_path)

        if args.dry_run:
            continue
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/audit_frozen_scenario_replays.py"),
                "--output",
                str(scale_root / "frozen_replay_audit.json"),
                str(bank_path),
                *[str(path) for path in summaries],
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        if any(run["method"] == "nominal_only" for run in runs):
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/analyze_frozen_multiseed_results.py"),
                    str(scale_root),
                    "--baseline-summary",
                    str(scale_root / "nominal_only/summary.json"),
                    "--output-dir",
                    str(scale_root / "analysis"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
        for run, summary_path in zip(runs, summaries):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            outcomes = summary.get("outcomes", {})
            episodes = int(summary["episode_count"])
            success = int(outcomes.get("success", 0))
            summary_rows.append(
                {
                    "offset_scale": scale,
                    "offset_regime": offset_regime(scale),
                    "clearance_m": args.slot_clearance,
                    "method": run["method"],
                    "training_seed": run["seed"],
                    "success": success,
                    "episodes": episodes,
                    "success_pct": 100.0 * success / episodes,
                    "drop": int(outcomes.get("drop", 0)),
                    "timeout": int(outcomes.get("timeout", 0)),
                    "scenario_sha256": summary["scenario_sha256"],
                    "checkpoint_sha256": summary.get("metadata", {}).get(
                        "checkpoint_sha256"
                    ),
                    "summary": str(summary_path),
                }
            )
        _write_offset_summary(output_root, summary_rows)

    if args.dry_run:
        print(f"DRY RUN: {len(args.offset_scales)} offset scales x {len(runs)} runs")
        return
    if not args.skip_plot:
        _run_plotter(output_root)
    print(f"Sweep summary: {output_root / 'offset_sweep_summary.csv'}")
    print(f"Method summary: {output_root / 'offset_method_summary.csv'}")
    print(f"Paper table: {output_root / 'offset_paper_results.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panda-usd", type=Path, required=True)
    parser.add_argument(
        "--isaac-python", type=Path, default=Path.home() / "isaacsim/python.sh"
    )
    parser.add_argument(
        "--offset-scales", type=float, nargs="+", default=list(DEFAULT_OFFSET_SCALES)
    )
    parser.add_argument("--slot-clearance", type=float, default=0.003)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["nominal_only", "ppo_only", "residual_ppo"],
    )
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--bank-seed", type=int, default=20260812)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--inter-run-delay-s", type=float, default=15.0)
    parser.add_argument("--startup-retries", type=int, default=2)
    parser.add_argument("--retry-delay-s", type=float, default=30.0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0.0 for value in args.offset_scales):
        parser.error("all offset scales must be finite and non-negative")
    if len(set(args.offset_scales)) != len(args.offset_scales):
        parser.error("offset scales must be unique")
    args.offset_scales = sorted(args.offset_scales)
    if args.slot_clearance <= 0.0:
        parser.error("--slot-clearance must be positive")
    if args.scenarios <= 0 or args.num_envs <= 0:
        parser.error("--scenarios and --num-envs must be positive")
    if args.inter_run_delay_s < 0.0 or args.startup_retries < 0 or args.retry_delay_s < 0.0:
        parser.error("retry counts and delays must be non-negative")
    run_sweep(args)


if __name__ == "__main__":
    main()
