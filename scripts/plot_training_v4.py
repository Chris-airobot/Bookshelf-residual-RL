#!/usr/bin/env python3
"""Plot presentation metrics from Bookshelf-Direct-v4 CSV training logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT = Path("logs/sb3/Bookshelf-Direct-v4")


def _latest_run(root: Path) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "training_summary.csv").exists() and (path / "episode_metrics.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No v4 runs with CSV metrics found under {root}")
    return max(candidates, key=lambda p: (p / "training_summary.csv").stat().st_mtime)


def _style_axis(ax, title: str, xlabel: str = "Training steps", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _plot_success(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["global_step"], summary["success_rate"] * 100.0, linewidth=2.2, color="#1f77b4")
    ax.set_ylim(0, 100)
    _style_axis(ax, "Rolling Success Rate", ylabel="Success rate (%)")
    _save(fig, out_dir / "success_rate.png")


def _plot_return_length(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    axes[0].plot(summary["global_step"], summary["mean_episode_return"], linewidth=2.0, color="#2ca02c")
    _style_axis(axes[0], "Episode Return", ylabel="Mean return")
    axes[1].plot(summary["global_step"], summary["mean_episode_length"], linewidth=2.0, color="#9467bd")
    _style_axis(axes[1], "Episode Length", ylabel="Mean steps")
    _save(fig, out_dir / "return_and_length.png")


def _plot_pose_errors(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.4), sharex=True)
    axes[0].plot(summary["global_step"], summary["mean_final_lat_err"] * 1000.0, linewidth=2.0, color="#d62728")
    _style_axis(axes[0], "Final Lateral Error", ylabel="mm")
    axes[1].plot(summary["global_step"], summary["mean_final_z_err"] * 1000.0, linewidth=2.0, color="#ff7f0e")
    _style_axis(axes[1], "Final Height Error", ylabel="mm")
    axes[2].plot(summary["global_step"], summary["mean_final_yaw_err_deg"], linewidth=2.0, color="#17becf")
    _style_axis(axes[2], "Final Yaw Error", ylabel="degrees")
    _save(fig, out_dir / "final_pose_errors.png")


def _plot_depth(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        summary["global_step"],
        summary["mean_final_rear_to_mouth"] * 1000.0,
        linewidth=2.0,
        label="rear_to_mouth",
        color="#1f77b4",
    )
    ax.plot(
        summary["global_step"],
        summary["mean_final_front_to_back"] * 1000.0,
        linewidth=2.0,
        label="front_to_back",
        color="#ff7f0e",
    )
    ax.legend(frameon=False)
    _style_axis(ax, "Final Depth Metrics", ylabel="mm")
    _save(fig, out_dir / "final_depth_metrics.png")


def _plot_failure_breakdown(episodes: pd.DataFrame, out_dir: Path, window: int = 1000) -> None:
    episodes = episodes.copy()
    episodes["bin"] = episodes["episode_id"] // window
    grouped = episodes.groupby(["bin", "failure_reason"]).size().unstack(fill_value=0)
    rates = grouped.div(grouped.sum(axis=1), axis=0)
    x_steps = episodes.groupby("bin")["global_step"].max().reindex(rates.index)

    preferred = ["success", "not_push", "depth", "lateral", "z", "yaw", "upright", "unstable", "drop", "timeout"]
    cols = [col for col in preferred if col in rates.columns] + [col for col in rates.columns if col not in preferred]
    rates = rates[cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(x_steps, *(rates[col] * 100.0 for col in rates.columns), labels=rates.columns)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _style_axis(ax, f"Failure Mode Breakdown ({window}-episode bins)", ylabel="Episode share (%)")
    _save(fig, out_dir / "failure_breakdown.png")


def _plot_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()

    axes[0].plot(summary["global_step"], summary["success_rate"] * 100.0, linewidth=2.0)
    axes[0].set_ylim(0, 100)
    _style_axis(axes[0], "Success Rate", ylabel="%")

    axes[1].plot(summary["global_step"], summary["mean_episode_return"], linewidth=2.0, color="#2ca02c")
    _style_axis(axes[1], "Episode Return", ylabel="mean")

    axes[2].plot(summary["global_step"], summary["mean_final_lat_err"] * 1000.0, label="lateral", linewidth=2.0)
    axes[2].plot(summary["global_step"], summary["mean_final_z_err"] * 1000.0, label="z", linewidth=2.0)
    axes[2].legend(frameon=False)
    _style_axis(axes[2], "Final Position Errors", ylabel="mm")

    axes[3].plot(summary["global_step"], summary["mean_final_yaw_err_deg"], linewidth=2.0, color="#17becf")
    _style_axis(axes[3], "Final Yaw Error", ylabel="degrees")

    _save(fig, out_dir / "overview.png")


def _write_summary(summary: pd.DataFrame, episodes: pd.DataFrame, run_dir: Path, out_dir: Path) -> None:
    last = summary.iloc[-1]
    payload = {
        "run_dir": str(run_dir),
        "num_episode_rows": int(len(episodes)),
        "num_summary_rows": int(len(summary)),
        "final_global_step": int(last["global_step"]),
        "final_success_rate": float(last["success_rate"]),
        "final_mean_episode_return": float(last["mean_episode_return"]),
        "final_mean_episode_length": float(last["mean_episode_length"]),
        "final_mean_lat_err_m": float(last["mean_final_lat_err"]),
        "final_mean_z_err_m": float(last["mean_final_z_err"]),
        "final_mean_yaw_err_deg": float(last["mean_final_yaw_err_deg"]),
        "final_mean_rear_to_mouth_m": float(last["mean_final_rear_to_mouth"]),
        "final_mean_front_to_back_m": float(last["mean_final_front_to_back"]),
        "final_failure_counts": episodes["failure_reason"].value_counts().to_dict(),
    }
    path = out_dir / "plot_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[plot] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Bookshelf-Direct-v4 training metrics.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Run directory. Defaults to latest v4 CSV run.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory. Defaults to <run_dir>/plots.")
    parser.add_argument("--failure_window", type=int, default=1000, help="Episode bin size for failure breakdown.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser() if args.run_dir is not None else _latest_run(DEFAULT_ROOT)
    summary_path = run_dir / "training_summary.csv"
    episode_path = run_dir / "episode_metrics.csv"
    if not summary_path.exists() or not episode_path.exists():
        raise FileNotFoundError(f"Expected CSV metrics in {run_dir}")

    out_dir = args.out_dir.expanduser() if args.out_dir is not None else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    episodes = pd.read_csv(episode_path)
    print(f"[plot] run: {run_dir}")
    print(f"[plot] summary rows: {len(summary)}  episode rows: {len(episodes)}")

    _plot_overview(summary, out_dir)
    _plot_success(summary, out_dir)
    _plot_return_length(summary, out_dir)
    _plot_pose_errors(summary, out_dir)
    _plot_depth(summary, out_dir)
    _plot_failure_breakdown(episodes, out_dir, window=int(args.failure_window))
    _write_summary(summary, episodes, run_dir, out_dir)


if __name__ == "__main__":
    main()
