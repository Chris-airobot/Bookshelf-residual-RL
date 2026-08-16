#!/usr/bin/env python3
"""Run resumable multi-seed frozen-bank evaluations across slot clearances."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEARANCES = (0.001, 0.002, 0.003, 0.004, 0.005)
DEFAULT_SEEDS = (42, 123, 2026)


def clearance_label(clearance: float) -> str:
    millimetres = 1000.0 * float(clearance)
    return f"clearance_{millimetres:.1f}mm".replace(".", "p")


def build_evaluation_runs(
    model_root: str | Path,
    seeds: list[int] | tuple[int, ...],
    methods: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    root = Path(model_root).expanduser().resolve()
    runs = []
    if "nominal_only" in methods:
        runs.append(
            {
                "name": "nominal_only",
                "method": "nominal_only",
                "seed": None,
                "task": "Bookshelf-Residual-Direct-v0",
                "checkpoint": None,
            }
        )
    task_by_method = {
        "ppo_only": "Bookshelf-PPO-Direct-v0",
        "residual_ppo": "Bookshelf-Residual-Direct-v0",
    }
    unknown = set(methods).difference({"nominal_only", *task_by_method})
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    for method in methods:
        if method not in task_by_method:
            continue
        for seed in seeds:
            directory = root / f"{method}_seed{seed}"
            checkpoint = directory / "model.zip"
            vecnormalize = directory / "model_vecnormalize.pkl"
            if not checkpoint.is_file() or not vecnormalize.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint pair for {method} seed={seed}: {directory}"
                )
            runs.append(
                {
                    "name": f"{method}_seed{seed}",
                    "method": method,
                    "seed": int(seed),
                    "task": task_by_method[method],
                    "checkpoint": str(checkpoint),
                }
            )
    return runs


def completed_summary_matches_bank(summary_path: str | Path, bank: dict[str, Any]) -> bool:
    path = Path(summary_path)
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    coverage = summary.get("frozen_scenario_bank_coverage") or {}
    bank_metadata = summary.get("metadata", {}).get("frozen_scenario_bank") or {}
    return bool(
        summary.get("scenario_trace_complete", False)
        and coverage.get("complete", False)
        and int(summary.get("episode_count", -1)) == int(bank["scenario_count"])
        and int(bank_metadata.get("scenario_count", -1)) == int(bank["scenario_count"])
        and bank_metadata.get("scenario_sha256") == bank["scenario_sha256"]
    )


def _stream_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _run_evaluation_with_retries(
    *,
    command: list[str],
    log_path: Path,
    summary_path: Path,
    bank: dict[str, Any],
    retries: int,
    retry_delay_s: float,
) -> None:
    for attempt in range(retries + 1):
        attempt_log = log_path if attempt == 0 else log_path.with_name(
            f"{log_path.stem}.retry{attempt}{log_path.suffix}"
        )
        try:
            _stream_command(command, attempt_log)
        except subprocess.CalledProcessError:
            if completed_summary_matches_bank(summary_path, bank):
                print(f"PASS: complete summary exists despite process shutdown error: {summary_path}")
                return
            if summary_path.exists():
                raise ValueError(
                    f"Failed run left a summary that does not match its bank: {summary_path}"
                )
            if attempt >= retries:
                raise
            print(
                f"RETRY: process failed before producing a summary; "
                f"attempt {attempt + 2}/{retries + 1} after {retry_delay_s:.1f}s"
            )
            time.sleep(retry_delay_s)
            continue
        if not completed_summary_matches_bank(summary_path, bank):
            raise ValueError(f"Evaluation exited successfully without a complete summary: {summary_path}")
        return


def _load_or_generate_bank(
    *,
    bank_path: Path,
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
    ]
    if dry_run:
        print(f"BANK: {shlex.join(command)}")
        return None
    if not bank_path.is_file():
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if int(bank.get("scenario_count", -1)) != scenario_count:
        raise ValueError(f"Bank scenario count mismatch: {bank_path}")
    clearances = {float(item["slot_clearance"]) for item in bank.get("scenarios", [])}
    if len(clearances) != 1 or not abs(next(iter(clearances)) - clearance) < 1.0e-9:
        raise ValueError(f"Bank clearance mismatch: {bank_path}")
    return bank


def _play_command(
    *,
    isaac_python: Path,
    panda_usd: Path,
    bank_path: Path,
    output_dir: Path,
    run: dict[str, Any],
    clearance: float,
    num_envs: int,
    evaluation_seed: int,
) -> list[str]:
    command = [
        "env",
        f"PYTHONPATH={REPO_ROOT / 'source/bookshelf'}",
        f"BOOKSHELF_PANDA_USD_PATH={panda_usd}",
        str(isaac_python),
        str(REPO_ROOT / "scripts/sb3/play.py"),
        "--task",
        run["task"],
        "--num_envs",
        str(num_envs),
        "--headless",
        "--seed",
        str(evaluation_seed),
        "--eval_slot_clearance",
        str(clearance),
        "--eval_scenario_bank",
        str(bank_path),
        "--eval_output_dir",
        str(output_dir),
    ]
    if run["checkpoint"] is None:
        command.append("--eval_nominal_only")
    else:
        command.extend(["--checkpoint", run["checkpoint"]])
    return command


def aggregate_clearance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((float(row["clearance_m"]), row["method"]), []).append(row)
    aggregate = []
    for (clearance, method), method_rows in sorted(grouped.items()):
        rates = [float(row["success_pct"]) for row in method_rows]
        aggregate.append(
            {
                "clearance_m": clearance,
                "clearance_mm": 1000.0 * clearance,
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


def _write_sweep_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    (output_root / "clearance_sweep_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "clearance_m",
        "clearance_mm",
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
    ]
    with (output_root / "clearance_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_clearance_rows(rows)
    (output_root / "clearance_method_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "clearance_method_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)


def run_sweep(args: argparse.Namespace) -> None:
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    panda_usd = args.panda_usd.expanduser().resolve()
    isaac_python = args.isaac_python.expanduser().resolve()
    if not panda_usd.is_file():
        raise FileNotFoundError(f"Franka USD is missing: {panda_usd}")
    if not isaac_python.is_file():
        raise FileNotFoundError(f"Isaac Python launcher is missing: {isaac_python}")
    runs = build_evaluation_runs(model_root, args.seeds, args.methods)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "clearances_m": args.clearances,
        "scenario_count_per_clearance": args.scenarios,
        "bank_seed": args.bank_seed,
        "evaluation_seed": args.evaluation_seed,
        "num_envs": args.num_envs,
        "inter_run_delay_s": args.inter_run_delay_s,
        "startup_retries": args.startup_retries,
        "retry_delay_s": args.retry_delay_s,
        "runs": runs,
    }
    (output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_rows = []
    for clearance in args.clearances:
        label = clearance_label(clearance)
        clearance_root = output_root / label
        bank_path = clearance_root / "frozen_bank.json"
        bank = _load_or_generate_bank(
            bank_path=bank_path,
            clearance=clearance,
            scenario_count=args.scenarios,
            bank_seed=args.bank_seed,
            dry_run=args.dry_run,
        )
        summaries = []
        for run in runs:
            run_output = clearance_root / run["name"]
            summary_path = run_output / "summary.json"
            command = _play_command(
                isaac_python=isaac_python,
                panda_usd=panda_usd,
                bank_path=bank_path,
                output_dir=run_output,
                run=run,
                clearance=clearance,
                num_envs=args.num_envs,
                evaluation_seed=args.evaluation_seed,
            )
            if args.dry_run:
                print(f"RUN: {shlex.join(command)}")
                continue
            assert bank is not None
            if completed_summary_matches_bank(summary_path, bank):
                print(f"SKIP: complete {label}/{run['name']}")
            elif summary_path.exists():
                raise ValueError(
                    f"Existing summary does not match the requested bank: {summary_path}"
                )
            else:
                print(f"RUN: {label}/{run['name']}")
                _run_evaluation_with_retries(
                    command=command,
                    log_path=clearance_root / f"{run['name']}.log",
                    summary_path=summary_path,
                    bank=bank,
                    retries=args.startup_retries,
                    retry_delay_s=args.retry_delay_s,
                )
                if args.inter_run_delay_s > 0.0:
                    print(f"COOLDOWN: {args.inter_run_delay_s:.1f}s before the next Isaac launch")
                    time.sleep(args.inter_run_delay_s)
            summaries.append(summary_path)

        if args.dry_run:
            continue
        audit_path = clearance_root / "frozen_replay_audit.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/audit_frozen_scenario_replays.py"),
                "--output",
                str(audit_path),
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
                    str(clearance_root),
                    "--baseline-summary",
                    str(clearance_root / "nominal_only/summary.json"),
                    "--output-dir",
                    str(clearance_root / "analysis"),
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
                    "clearance_m": clearance,
                    "clearance_mm": 1000.0 * clearance,
                    "method": run["method"],
                    "training_seed": run["seed"],
                    "success": success,
                    "episodes": episodes,
                    "success_pct": 100.0 * success / episodes,
                    "drop": int(outcomes.get("drop", 0)),
                    "timeout": int(outcomes.get("timeout", 0)),
                    "scenario_sha256": summary["scenario_sha256"],
                    "checkpoint_sha256": summary.get("metadata", {}).get("checkpoint_sha256"),
                    "summary": str(summary_path),
                }
            )
        _write_sweep_summary(output_root, summary_rows)

    if args.dry_run:
        print(f"DRY RUN: {len(args.clearances)} clearances x {len(runs)} runs")
    else:
        print(f"Sweep summary: {output_root / 'clearance_sweep_summary.csv'}")
        print(f"Method summary: {output_root / 'clearance_method_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panda-usd", type=Path, required=True)
    parser.add_argument(
        "--isaac-python", type=Path, default=Path.home() / "isaacsim/python.sh"
    )
    parser.add_argument("--clearances", type=float, nargs="+", default=list(DEFAULT_CLEARANCES))
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.clearances):
        parser.error("all clearances must be positive")
    if args.scenarios <= 0 or args.num_envs <= 0:
        parser.error("--scenarios and --num-envs must be positive")
    if args.inter_run_delay_s < 0.0 or args.startup_retries < 0 or args.retry_delay_s < 0.0:
        parser.error("retry counts and delays must be non-negative")
    run_sweep(args)


if __name__ == "__main__":
    main()
