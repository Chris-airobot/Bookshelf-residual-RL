#!/usr/bin/env python3
"""Generate a hardware-free validation report for the bookshelf shadow policy."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from bookshelf_shadow_ros.offline_validation import (
    audit_shadow_source_tree,
    case_as_dict,
    controller_config_parity,
    evaluate_shadow_case,
    make_pose_transform,
    perturb_transform,
)
from bookshelf_shadow_ros.policy_observation_math import validate_detector_measurement
from bookshelf_shadow_ros.policy_shadow_math import (
    NumpyActorBundle,
    validate_shadow_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    REPOSITORY_ROOT
    / "data"
    / "policy_exports"
    / "bookshelf_residual_2026-07-08_shadow_actor.npz"
)
DEFAULT_ENV_CFG = (
    REPOSITORY_ROOT
    / "source"
    / "bookshelf"
    / "bookshelf"
    / "tasks"
    / "direct"
    / "bookshelf"
    / "bookshelf_residual_env_cfg.py"
)
DEFAULT_SHADOW_SOURCE = (
    REPOSITORY_ROOT
    / "ros2"
    / "bookshelf_shadow_ros"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic synthetic geometry, policy-response, calibration "
            "sensitivity, fail-closed, parity, and source safety checks."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--env-cfg", type=Path, default=DEFAULT_ENV_CFG)
    parser.add_argument("--shadow-source", type=Path, default=DEFAULT_SHADOW_SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory. Defaults to logs/offline_shadow_validation/<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sensitivity-samples",
        type=int,
        default=1000,
        help="Monte Carlo samples per calibration-error level.",
    )
    return parser.parse_args()


def _flatten_case(prefix: str, result) -> dict:
    row = {}
    groups = (
        ("raw", result.raw_metrics),
        ("obs", result.observation),
        ("normalized", result.normalized_observation),
        ("actor", result.actor_mean),
        ("action", result.policy_action),
        ("nominal", result.nominal_delta),
        ("residual", result.residual_delta),
        ("final", result.final_delta),
    )
    for group, values in groups:
        for index, value in enumerate(values):
            row[f"{prefix}{group}_{index}"] = float(value)
    row[f"{prefix}release_requested"] = int(result.release_requested)
    return row


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _response_sweep(bundle, base_book, tool):
    definitions = (
        ("lateral_mm", np.linspace(-12.0, 12.0, 49)),
        ("vertical_mm", np.linspace(-12.0, 12.0, 49)),
        ("yaw_deg", np.linspace(-10.0, 10.0, 41)),
        ("pitch_deg", np.linspace(-8.0, 8.0, 33)),
        ("insertion_mm", np.linspace(-30.0, 30.0, 61)),
    )
    rows = []
    for parameter, values in definitions:
        for value in values:
            translation = [0.0, 0.0, 0.0]
            rotation = [0.0, 0.0, 0.0]
            if parameter == "lateral_mm":
                translation[1] = value * 0.001
            elif parameter == "vertical_mm":
                translation[2] = value * 0.001
            elif parameter == "yaw_deg":
                rotation[2] = math.radians(value)
            elif parameter == "pitch_deg":
                rotation[1] = math.radians(value)
            elif parameter == "insertion_mm":
                translation[0] = value * 0.001
            book = perturb_transform(
                base_book,
                translation_xyz=translation,
                rotation_rpy=rotation,
            )
            result = evaluate_shadow_case(bundle, book, tool)
            row = {"parameter": parameter, "value": float(value)}
            row.update(_flatten_case("", result))
            rows.append(row)
    return rows


def _calibration_sensitivity(bundle, base_book, tool, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    baseline = evaluate_shadow_case(bundle, base_book, tool)
    baseline_observation = np.asarray(baseline.observation)
    baseline_residual = np.asarray(baseline.residual_delta)
    baseline_final = np.asarray(baseline.final_delta)
    baseline_forward = baseline.nominal_delta[0] > 0.0

    levels = (
        ("1mm_0.5deg", 0.001, math.radians(0.5)),
        ("3mm_1deg", 0.003, math.radians(1.0)),
        ("5mm_2deg", 0.005, math.radians(2.0)),
        ("10mm_5deg", 0.010, math.radians(5.0)),
    )
    sample_rows = []
    summaries = []
    for name, translation_bound, rotation_bound in levels:
        observation_linf = []
        residual_position_norm = []
        residual_orientation_norm = []
        final_position_norm = []
        final_orientation_norm = []
        release_flips = 0
        forward_gate_flips = 0
        for sample_index in range(samples):
            translation = rng.uniform(-translation_bound, translation_bound, size=3)
            rotation = rng.uniform(-rotation_bound, rotation_bound, size=3)
            result = evaluate_shadow_case(
                bundle,
                perturb_transform(
                    base_book,
                    translation_xyz=translation,
                    rotation_rpy=rotation,
                ),
                tool,
            )
            obs_error = float(
                np.max(np.abs(np.asarray(result.observation) - baseline_observation))
            )
            residual_difference = np.asarray(result.residual_delta) - baseline_residual
            final_difference = np.asarray(result.final_delta) - baseline_final
            residual_position_error = float(np.linalg.norm(residual_difference[:3]))
            residual_orientation_error = float(np.linalg.norm(residual_difference[3:]))
            final_position_error = float(np.linalg.norm(final_difference[:3]))
            final_orientation_error = float(np.linalg.norm(final_difference[3:]))
            release_flip = result.release_requested != baseline.release_requested
            forward_flip = (result.nominal_delta[0] > 0.0) != baseline_forward
            release_flips += int(release_flip)
            forward_gate_flips += int(forward_flip)
            sample_rows.append(
                {
                    "level": name,
                    "sample": sample_index,
                    "tx_m": float(translation[0]),
                    "ty_m": float(translation[1]),
                    "tz_m": float(translation[2]),
                    "roll_deg": math.degrees(float(rotation[0])),
                    "pitch_deg": math.degrees(float(rotation[1])),
                    "yaw_deg": math.degrees(float(rotation[2])),
                    "observation_linf": obs_error,
                    "residual_position_l2_m": residual_position_error,
                    "residual_orientation_l2_rad": residual_orientation_error,
                    "final_position_l2_m": final_position_error,
                    "final_orientation_l2_rad": final_orientation_error,
                    "release_flip": int(release_flip),
                    "forward_gate_flip": int(forward_flip),
                }
            )
            observation_linf.append(obs_error)
            residual_position_norm.append(residual_position_error)
            residual_orientation_norm.append(residual_orientation_error)
            final_position_norm.append(final_position_error)
            final_orientation_norm.append(final_orientation_error)

        summaries.append(
            {
                "level": name,
                "translation_bound_m": translation_bound,
                "rotation_bound_deg": math.degrees(rotation_bound),
                "samples": samples,
                "observation_linf_mean": float(np.mean(observation_linf)),
                "observation_linf_p95": float(np.percentile(observation_linf, 95)),
                "residual_position_l2_mean_m": float(np.mean(residual_position_norm)),
                "residual_position_l2_p95_m": float(
                    np.percentile(residual_position_norm, 95)
                ),
                "residual_orientation_l2_mean_rad": float(
                    np.mean(residual_orientation_norm)
                ),
                "residual_orientation_l2_p95_rad": float(
                    np.percentile(residual_orientation_norm, 95)
                ),
                "final_position_l2_mean_m": float(np.mean(final_position_norm)),
                "final_position_l2_p95_m": float(
                    np.percentile(final_position_norm, 95)
                ),
                "final_orientation_l2_mean_rad": float(
                    np.mean(final_orientation_norm)
                ),
                "final_orientation_l2_p95_rad": float(
                    np.percentile(final_orientation_norm, 95)
                ),
                "release_flip_rate": release_flips / samples,
                "forward_gate_flip_rate": forward_gate_flips / samples,
            }
        )
    return sample_rows, summaries


def _fault_checks():
    good = np.zeros(12, dtype=np.float32)
    timing = {
        "observation_valid": True,
        "valid_age_s": 0.01,
        "observation_age_s": 0.01,
        "raw_metrics_age_s": 0.01,
        "pair_skew_s": 0.01,
    }
    cases = {
        "valid_inputs": validate_shadow_inputs(good, good, **timing) is None,
        "invalid_flag_rejected": validate_shadow_inputs(
            good,
            good,
            **{**timing, "observation_valid": False},
        )
        is not None,
        "stale_observation_rejected": validate_shadow_inputs(
            good,
            good,
            **{**timing, "observation_age_s": 0.51},
        )
        is not None,
        "skew_rejected": validate_shadow_inputs(
            good,
            good,
            **{**timing, "pair_skew_s": 0.11},
        )
        is not None,
        "wrong_shape_rejected": validate_shadow_inputs(
            np.zeros(11),
            good,
            **timing,
        )
        is not None,
        "nan_observation_rejected": validate_shadow_inputs(
            np.full(12, np.nan),
            good,
            **timing,
        )
        is not None,
        "low_confidence_rejected": validate_detector_measurement(0.037, 0.2)
        is not None,
        "nan_confidence_rejected": validate_detector_measurement(0.037, np.nan)
        is not None,
        "bad_width_rejected": validate_detector_measurement(0.010, 0.8)
        is not None,
    }
    return {"passed": all(cases.values()), "checks": cases}


def _determinism_check(bundle, observation, repeats=100):
    reference = bundle.predict(observation)
    maximum_error = 0.0
    for _ in range(repeats):
        current = bundle.predict(observation)
        maximum_error = max(
            maximum_error,
            *(float(np.max(np.abs(left - right))) for left, right in zip(reference, current)),
        )
    return {
        "passed": maximum_error == 0.0,
        "repeats": repeats,
        "maximum_absolute_error": maximum_error,
    }


def _write_plots(output_dir: Path, response_rows: list[dict], sensitivity: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"created": False, "reason": "matplotlib is unavailable"}

    plot_specs = (
        ("lateral_mm", 1, "dy (mm)"),
        ("vertical_mm", 2, "dz (mm)"),
        ("yaw_deg", 3, "dyaw (deg)"),
        ("pitch_deg", 4, "dpitch (deg)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (parameter, motion_index, ylabel) in zip(axes.flat, plot_specs):
        selected = [row for row in response_rows if row["parameter"] == parameter]
        x = [row["value"] for row in selected]
        factors = 1000.0 if motion_index < 3 else 180.0 / math.pi
        axis.plot(x, [row[f"nominal_{motion_index}"] * factors for row in selected], label="nominal")
        axis.plot(x, [row[f"residual_{motion_index}"] * factors for row in selected], label="residual")
        axis.plot(x, [row[f"final_{motion_index}"] * factors for row in selected], label="final")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel(parameter.replace("_", " "))
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Offline Shadow Policy Response Sweeps")
    figure.tight_layout()
    response_path = output_dir / "policy_response_sweeps.png"
    figure.savefig(response_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(17, 4))
    labels = [item["level"] for item in sensitivity]
    axes[0].bar(labels, [item["observation_linf_p95"] for item in sensitivity])
    axes[0].set_ylabel("p95 observation L-inf")
    axes[1].bar(labels, [item["final_position_l2_p95_m"] * 1000.0 for item in sensitivity])
    axes[1].set_ylabel("p95 final position delta (mm)")
    axes[2].bar(
        labels,
        [
            math.degrees(item["final_orientation_l2_p95_rad"])
            for item in sensitivity
        ],
    )
    axes[2].set_ylabel("p95 final angle delta (deg)")
    axes[3].bar(labels, [item["forward_gate_flip_rate"] * 100.0 for item in sensitivity])
    axes[3].set_ylabel("forward gate flips (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Book-Frame Calibration Sensitivity")
    figure.tight_layout()
    sensitivity_path = output_dir / "calibration_sensitivity.png"
    figure.savefig(sensitivity_path, dpi=180)
    plt.close(figure)
    return {
        "created": True,
        "files": [str(response_path), str(sensitivity_path)],
    }


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPOSITORY_ROOT / "logs" / "offline_shadow_validation" / timestamp
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = NumpyActorBundle(args.bundle)
    base_book = make_pose_transform([0.078, 0.0, 0.006])
    tool = make_pose_transform([0.020, 0.0, 0.006])
    baseline = evaluate_shadow_case(bundle, base_book, tool)

    response_rows = _response_sweep(bundle, base_book, tool)
    sensitivity_rows, sensitivity_summary = _calibration_sensitivity(
        bundle,
        base_book,
        tool,
        max(args.sensitivity_samples, 1),
        args.seed,
    )
    _write_csv(output_dir / "policy_response_sweeps.csv", response_rows)
    _write_csv(output_dir / "calibration_sensitivity_samples.csv", sensitivity_rows)

    source_findings = audit_shadow_source_tree(args.shadow_source)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "hardware_used": False,
        "robot_commands_executed": False,
        "physical_calibration_verified": False,
        "policy_bundle": str(args.bundle.resolve()),
        "policy_bundle_sha256": bundle.sha256,
        "policy_metadata": bundle.metadata,
        "baseline": case_as_dict(baseline),
        "determinism": _determinism_check(bundle, baseline.observation),
        "fault_injection": _fault_checks(),
        "controller_config_parity": controller_config_parity(args.env_cfg),
        "source_safety_audit": {
            "passed": not source_findings,
            "source_root": str(args.shadow_source.resolve()),
            "findings": source_findings,
        },
        "calibration_sensitivity": sensitivity_summary,
        "limitations": [
            "The physical link_eef-to-book transform is not identified by this report.",
            "The physical shelf depth and slot target offset are not identified by this report.",
            "RGB-D detection accuracy against physical ground truth is not established.",
            "Contact dynamics and real robot execution are not exercised.",
        ],
    }
    report["plots"] = _write_plots(output_dir, response_rows, sensitivity_summary)
    report["passed"] = all(
        (
            report["determinism"]["passed"],
            report["fault_injection"]["passed"],
            report["controller_config_parity"]["passed"],
            report["source_safety_audit"]["passed"],
        )
    )

    report_path = output_dir / "validation_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Offline shadow validation: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report: {report_path}")
    print(f"Bundle SHA256: {bundle.sha256}")
    print(
        "Physical geometry remains UNVERIFIED: link_eef-to-book calibration, "
        "shelf depth/target offset, and real RGB-D ground truth."
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
