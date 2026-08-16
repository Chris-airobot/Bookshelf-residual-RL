#!/usr/bin/env python3
"""Generate final Bookshelf simulation figures, tables, and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHOD_ORDER = ("nominal_only", "ppo_only", "residual_ppo")
METHOD_LABELS = {
    "nominal_only": "Nominal only",
    "ppo_only": "PPO only",
    "residual_ppo": "Residual PPO",
}
METHOD_STYLES = {
    "nominal_only": {"color": "#555555", "marker": "s"},
    "ppo_only": {"color": "#277DA1", "marker": "^"},
    "residual_ppo": {"color": "#2A9D55", "marker": "o"},
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_frozen_audits(summary_paths: list[str | Path]) -> list[Path]:
    audit_paths = sorted(
        {Path(summary).resolve().parent.parent / "frozen_replay_audit.json" for summary in summary_paths}
    )
    if not audit_paths:
        raise ValueError("No frozen replay audits were discovered")
    for audit_path in audit_paths:
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing frozen replay audit: {audit_path}")
        audit = read_json(audit_path)
        if not audit.get("passed", False):
            raise ValueError(f"Frozen replay audit did not pass: {audit_path}")
        checks = audit.get("checks") or []
        if not checks or not all(check.get("passed", False) for check in checks):
            raise ValueError(f"Frozen replay audit contains a failed check: {audit_path}")
    return audit_paths


def build_method_series(
    rows: list[dict[str, str]], *, x_field: str, x_scale: float = 1.0
) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        method = row["method"]
        values = series.setdefault(method, {"x": [], "mean": [], "sd": []})
        values["x"].append(float(row[x_field]) * x_scale)
        values["mean"].append(float(row["mean_success_pct"]))
        values["sd"].append(
            float(row.get("sample_stdev_success_percentage_points") or 0.0)
        )
    for method, values in series.items():
        ordered = sorted(zip(values["x"], values["mean"], values["sd"]))
        values["x"], values["mean"], values["sd"] = map(list, zip(*ordered))
        if len(set(values["x"])) != len(values["x"]):
            raise ValueError(f"Duplicate {x_field} for {method}")
    return series


def _seed_points(
    rows: list[dict[str, str]], *, x_field: str, x_scale: float = 1.0
) -> dict[str, list[tuple[float, float, int]]]:
    points: dict[str, list[tuple[float, float, int]]] = {}
    for row in rows:
        seed = row.get("training_seed", "")
        if seed in ("", None):
            continue
        points.setdefault(row["method"], []).append(
            (float(row[x_field]) * x_scale, float(row["success_pct"]), int(seed))
        )
    return points


def _plot_panel(
    axis,
    *,
    series: dict[str, dict[str, list[float]]],
    seed_points: dict[str, list[tuple[float, float, int]]],
) -> None:
    for method in METHOD_ORDER:
        if method not in series:
            continue
        values = series[method]
        style = METHOD_STYLES[method]
        axis.errorbar(
            values["x"],
            values["mean"],
            yerr=values["sd"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.8,
            linewidth=1.7,
            capsize=2.4,
            capthick=0.9,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        seeds = sorted({point[2] for point in seed_points.get(method, [])})
        for x_value, success, seed in seed_points.get(method, []):
            seed_index = seeds.index(seed)
            seed_shift = 0.018 * (seed_index - 0.5 * (len(seeds) - 1))
            axis.scatter(
                x_value + seed_shift,
                success,
                s=10,
                facecolors="white",
                edgecolors=style["color"],
                linewidths=0.65,
                zorder=4,
            )
    axis.set_ylim(-2.0, 102.0)
    axis.set_yticks(range(0, 101, 20))
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render_combined_figure(
    *,
    clearance_rows: list[dict[str, str]],
    clearance_raw_rows: list[dict[str, str]],
    offset_rows: list[dict[str, str]],
    offset_raw_rows: list[dict[str, str]],
    output_stem: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clearance_series = build_method_series(
        clearance_rows, x_field="clearance_mm"
    )
    offset_series = build_method_series(offset_rows, x_field="offset_scale")
    clearance_points = _seed_points(clearance_raw_rows, x_field="clearance_mm")
    offset_points = _seed_points(offset_raw_rows, x_field="offset_scale")

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.75), sharey=True, constrained_layout=True
    )
    _plot_panel(axes[0], series=clearance_series, seed_points=clearance_points)
    _plot_panel(axes[1], series=offset_series, seed_points=offset_points)

    clearance_x = sorted(
        {value for method in clearance_series.values() for value in method["x"]}
    )
    axes[0].set_xlim(min(clearance_x) - 0.15, max(clearance_x) + 0.15)
    axes[0].set_xticks(clearance_x)
    axes[0].set_xlabel("Slot clearance (mm)")
    axes[0].set_ylabel("Insertion success (%)")
    axes[0].set_title("(a) Clearance robustness", loc="left")

    offset_x = sorted(
        {value for method in offset_series.values() for value in method["x"]}
    )
    if max(offset_x) > 1.0:
        axes[1].axvspan(1.0, max(offset_x) + 0.03, color="#ECECEC", alpha=0.75, zorder=0)
        axes[1].text(
            1.01,
            3.0,
            "outside training range",
            color="#555555",
            fontsize=7.0,
            ha="left",
            va="bottom",
        )
    axes[1].axvline(1.0, color="#777777", linestyle="--", linewidth=0.9, zorder=1)
    axes[1].set_xlim(min(offset_x) - 0.05, max(offset_x) + 0.05)
    axes[1].set_xticks(offset_x, [f"{value:g}x" for value in offset_x])
    axes[1].set_xlabel("Initial-offset severity")
    axes[1].set_title("(b) Initialization robustness", loc="left")
    axes[0].legend(frameon=False, loc="lower right")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 300}),
    ):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


def build_main_3mm_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    method_summaries = {
        row["method"]: row for row in analysis.get("method_summaries", [])
    }
    baseline = next(
        (
            run
            for run in analysis.get("runs", [])
            if run["method"] == "nominal_only" and run.get("seed") is None
        ),
        None,
    )
    if baseline is None:
        raise ValueError("The fixed 3 mm analysis has no nominal baseline")
    rows = []
    for method in METHOD_ORDER:
        if method == "nominal_only":
            rows.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "training_seeds": "n/a",
                    "episodes_per_run": baseline["episode_count"],
                    "mean_success_pct": baseline["success_pct"],
                    "sample_stdev_pp": 0.0,
                    "minimum_success_pct": baseline["success_pct"],
                    "maximum_success_pct": baseline["success_pct"],
                    "interval_type": "Wilson episode CI",
                    "interval_low_pct": baseline["success_wilson95_pct"][0],
                    "interval_high_pct": baseline["success_wilson95_pct"][1],
                }
            )
            continue
        summary = method_summaries.get(method)
        if summary is None:
            raise ValueError(f"The fixed 3 mm analysis has no {method} summary")
        interval = summary.get("mean_t95_pct")
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "training_seeds": ",".join(str(seed) for seed in summary["training_seeds"]),
                "episodes_per_run": analysis["scenario_count"],
                "mean_success_pct": summary["mean_success_pct"],
                "sample_stdev_pp": summary["sample_stdev_success_percentage_points"],
                "minimum_success_pct": summary["minimum_success_pct"],
                "maximum_success_pct": summary["maximum_success_pct"],
                "interval_type": "training-seed t CI",
                "interval_low_pct": interval[0] if interval else None,
                "interval_high_pct": interval[1] if interval else None,
            }
        )
    return rows


def build_nominal_residual_paired_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in analysis.get("paired_comparisons", []):
        if not comparison["left"].startswith("nominal_only_seed"):
            continue
        if not comparison["right"].startswith("residual_ppo_seed"):
            continue
        rows.append(
            {
                "residual_seed": int(comparison["right"].rsplit("seed", 1)[1]),
                "nominal_success_pct": comparison["left_success_pct"],
                "residual_success_pct": comparison["right_success_pct"],
                "gain_percentage_points": comparison[
                    "right_minus_left_percentage_points"
                ],
                "nominal_only_success": comparison["left_only_success"],
                "residual_only_success": comparison["right_only_success"],
                "both_success": comparison["both_success"],
                "both_fail": comparison["both_fail"],
                "mcnemar_exact_two_sided_p": comparison[
                    "mcnemar_exact_two_sided_p"
                ],
            }
        )
    if not rows:
        raise ValueError("No paired nominal-versus-residual comparisons were found")
    return sorted(rows, key=lambda row: row["residual_seed"])


def _pivot_sweep_rows(
    rows: list[dict[str, str]], *, x_field: str, regime_field: str | None = None
) -> list[dict[str, Any]]:
    by_x: dict[float, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_x.setdefault(float(row[x_field]), {})[row["method"]] = row
    output = []
    for x_value, methods in sorted(by_x.items()):
        missing = set(METHOD_ORDER).difference(methods)
        if missing:
            raise ValueError(f"Sweep row {x_value} is missing methods: {sorted(missing)}")
        result: dict[str, Any] = {x_field: x_value}
        if regime_field is not None:
            result[regime_field] = methods["residual_ppo"][regime_field]
        for method in METHOD_ORDER:
            result[f"{method}_mean_success_pct"] = float(
                methods[method]["mean_success_pct"]
            )
            result[f"{method}_sample_stdev_pp"] = float(
                methods[method].get("sample_stdev_success_percentage_points") or 0.0
            )
        output.append(result)
    return output


def _format_p(value: float) -> str:
    if value == 0.0:
        return r"$<10^{-300}$"
    if value < 0.001:
        exponent = math.floor(math.log10(value))
        coefficient = value / (10**exponent)
        return f"${coefficient:.2f}\\times10^{{{exponent}}}$"
    return f"{value:.3f}"


def write_latex_tables(
    *,
    tables_dir: Path,
    main_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    clearance_rows: list[dict[str, Any]],
    offset_rows: list[dict[str, Any]],
) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    main_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Seeds & Success (\%) & Seed SD (pp) & Seed range (\%) \\",
        r"\midrule",
    ]
    for row in main_rows:
        main_lines.append(
            f"{row['label']} & {row['training_seeds']} & {row['mean_success_pct']:.2f} & "
            f"{row['sample_stdev_pp']:.2f} & {row['minimum_success_pct']:.2f}--"
            f"{row['maximum_success_pct']:.2f} \\\\"
        )
    main_lines.extend([r"\bottomrule", r"\end{tabular}"])
    main_path = tables_dir / "main_3mm_results.tex"
    main_path.write_text("\n".join(main_lines) + "\n", encoding="utf-8")
    outputs.append(main_path)

    paired_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Seed & Nominal (\%) & Residual (\%) & Gain (pp) & McNemar $p$ \\",
        r"\midrule",
    ]
    for row in paired_rows:
        paired_lines.append(
            f"{row['residual_seed']} & {row['nominal_success_pct']:.2f} & "
            f"{row['residual_success_pct']:.2f} & {row['gain_percentage_points']:.2f} & "
            f"{_format_p(float(row['mcnemar_exact_two_sided_p']))} \\\\"
        )
    paired_lines.extend([r"\bottomrule", r"\end{tabular}"])
    paired_path = tables_dir / "paired_3mm_results.tex"
    paired_path.write_text("\n".join(paired_lines) + "\n", encoding="utf-8")
    outputs.append(paired_path)

    def robustness_table(path: Path, rows: list[dict[str, Any]], x_key: str) -> None:
        lines = [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Condition & Nominal (\%) & PPO only (\%) & Residual PPO (\%) \\",
            r"\midrule",
        ]
        for row in rows:
            condition = (
                f"{row[x_key]:.1f} mm"
                if x_key == "clearance_mm"
                else f"{row[x_key]:.2f}$\\times$"
            )
            lines.append(
                f"{condition} & {row['nominal_only_mean_success_pct']:.2f} & "
                f"{row['ppo_only_mean_success_pct']:.2f} & "
                f"{row['residual_ppo_mean_success_pct']:.2f} $\\pm$ "
                f"{row['residual_ppo_sample_stdev_pp']:.2f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    clearance_path = tables_dir / "clearance_results.tex"
    robustness_table(clearance_path, clearance_rows, "clearance_mm")
    outputs.append(clearance_path)
    offset_path = tables_dir / "offset_results.tex"
    robustness_table(offset_path, offset_rows, "offset_scale")
    outputs.append(offset_path)
    return outputs


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def generate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    fixed_path = args.fixed_analysis.expanduser().resolve()
    clearance_path = args.clearance_summary.expanduser().resolve()
    offset_path = args.offset_summary.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    repository = args.repository.expanduser().resolve()
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    fixed_analysis = read_json(fixed_path)
    clearance_method_rows = read_csv(clearance_path)
    offset_method_rows = read_csv(offset_path)
    clearance_raw_path = clearance_path.with_name("clearance_sweep_summary.csv")
    offset_raw_path = offset_path.with_name("offset_sweep_summary.csv")
    clearance_raw_rows = read_csv(clearance_raw_path)
    offset_raw_rows = read_csv(offset_raw_path)

    fixed_summary_paths = [run["summary"] for run in fixed_analysis["runs"]]
    clearance_summary_paths = [row["summary"] for row in clearance_raw_rows]
    offset_summary_paths = [row["summary"] for row in offset_raw_rows]
    audits = validate_frozen_audits(
        fixed_summary_paths + clearance_summary_paths + offset_summary_paths
    )

    main_rows = build_main_3mm_rows(fixed_analysis)
    paired_rows = build_nominal_residual_paired_rows(fixed_analysis)
    clearance_rows = _pivot_sweep_rows(
        clearance_method_rows, x_field="clearance_mm"
    )
    offset_rows = _pivot_sweep_rows(
        offset_method_rows,
        x_field="offset_scale",
        regime_field="offset_regime",
    )

    write_csv(
        tables_dir / "main_3mm_results.csv",
        main_rows,
        list(main_rows[0]),
    )
    write_csv(
        tables_dir / "paired_3mm_results.csv",
        paired_rows,
        list(paired_rows[0]),
    )
    write_csv(
        tables_dir / "clearance_results.csv",
        clearance_rows,
        list(clearance_rows[0]),
    )
    write_csv(
        tables_dir / "offset_results.csv",
        offset_rows,
        list(offset_rows[0]),
    )
    figure_paths = render_combined_figure(
        clearance_rows=clearance_method_rows,
        clearance_raw_rows=clearance_raw_rows,
        offset_rows=offset_method_rows,
        offset_raw_rows=offset_raw_rows,
        output_stem=figures_dir / "simulation_robustness",
    )
    latex_paths = write_latex_tables(
        tables_dir=tables_dir,
        main_rows=main_rows,
        paired_rows=paired_rows,
        clearance_rows=clearance_rows,
        offset_rows=offset_rows,
    )

    source_paths = [
        fixed_path,
        clearance_path,
        clearance_raw_path,
        offset_path,
        offset_raw_path,
        *audits,
    ]
    artifact_paths = [
        *figure_paths,
        *latex_paths,
        *sorted(tables_dir.glob("*.csv")),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "bookshelf_simulation_paper_artifacts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "path": str(repository),
            "branch": git_value(repository, "branch", "--show-current"),
            "commit": git_value(repository, "rev-parse", "HEAD"),
        },
        "evaluation": {
            "fixed_3mm_scenarios_per_run": fixed_analysis["scenario_count"],
            "fixed_3mm_scenario_sha256": fixed_analysis["scenario_sha256"],
            "clearance_conditions": len(clearance_rows),
            "offset_conditions": len(offset_rows),
            "training_seed_count": 3,
            "all_frozen_audits_passed": True,
        },
        "sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_paths
        ],
        "artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
        "checkpoint_sha256": sorted(
            {
                run["checkpoint_sha256"]
                for run in fixed_analysis["runs"]
                if run.get("checkpoint_sha256")
            }
        ),
        "command": [sys.executable, *sys.argv],
    }
    manifest_path = output_root / "results_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest_path, "figures": figure_paths, "tables": latex_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-analysis", type=Path, required=True)
    parser.add_argument("--clearance-summary", type=Path, required=True)
    parser.add_argument("--offset-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    outputs = generate_artifacts(args)
    print(f"Manifest: {outputs['manifest']}")
    for path in outputs["figures"]:
        print(f"Figure: {path}")
    for path in outputs["tables"]:
        print(f"Table: {path}")


if __name__ == "__main__":
    main()
