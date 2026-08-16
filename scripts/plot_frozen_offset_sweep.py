#!/usr/bin/env python3
"""Render a paper-ready success-versus-initial-offset figure."""

from __future__ import annotations

import argparse
import csv
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


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {source}")
    required = {"offset_scale", "method", "mean_success_pct"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {source}")
    return rows


def build_plot_series(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        method = row["method"]
        values = series.setdefault(method, {"x": [], "mean": [], "sd": []})
        values["x"].append(float(row["offset_scale"]))
        values["mean"].append(float(row["mean_success_pct"]))
        values["sd"].append(
            float(row.get("sample_stdev_success_percentage_points") or 0.0)
        )
    for method, values in series.items():
        ordered = sorted(zip(values["x"], values["mean"], values["sd"]))
        values["x"], values["mean"], values["sd"] = map(list, zip(*ordered))
        if len(set(values["x"])) != len(values["x"]):
            raise ValueError(f"Duplicate offset scale for {method}")
    return series


def _load_seed_points(path: Path) -> dict[str, list[tuple[float, float, int]]]:
    if not path.is_file():
        return {}
    points: dict[str, list[tuple[float, float, int]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("training_seed", "") == "":
                continue
            points.setdefault(row["method"], []).append(
                (
                    float(row["offset_scale"]),
                    float(row["success_pct"]),
                    int(row["training_seed"]),
                )
            )
    return points


def render_figure(summary_csv: Path, output_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_csv_rows(summary_csv)
    series = build_plot_series(rows)
    seed_points = _load_seed_points(summary_csv.with_name("offset_sweep_summary.csv"))
    all_x = sorted({value for method in series.values() for value in method["x"]})
    if not all_x:
        raise ValueError("No offset scales are available to plot")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(4.9, 3.25), constrained_layout=True)
    if max(all_x) > 1.0:
        axis.axvspan(1.0, max(all_x) + 0.03, color="#ECECEC", alpha=0.75, zorder=0)
        axis.text(
            1.01,
            3.0,
            "outside training range",
            color="#555555",
            fontsize=7.5,
            ha="left",
            va="bottom",
        )
    axis.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)

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
            markersize=5.0,
            linewidth=1.8,
            capsize=2.5,
            capthick=1.0,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        method_seeds = sorted({point[2] for point in seed_points.get(method, [])})
        for x_value, success, seed in seed_points.get(method, []):
            seed_index = method_seeds.index(seed)
            seed_shift = 0.018 * (seed_index - 0.5 * (len(method_seeds) - 1))
            axis.scatter(
                x_value + seed_shift,
                success,
                s=11,
                facecolors="white",
                edgecolors=style["color"],
                linewidths=0.7,
                alpha=0.9,
                zorder=4,
            )

    axis.set_xlabel("Initial-offset severity relative to training maximum")
    axis.set_ylabel("Insertion success (%)")
    axis.set_xlim(min(all_x) - 0.05, max(all_x) + 0.05)
    axis.set_ylim(-2.0, 102.0)
    axis.set_xticks(all_x, [f"{value:g}x" for value in all_x])
    axis.set_yticks(range(0, 101, 20))
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    for path in render_figure(args.summary_csv.resolve(), args.output_stem.resolve()):
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
