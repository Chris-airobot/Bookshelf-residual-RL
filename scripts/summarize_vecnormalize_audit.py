#!/usr/bin/env python3
"""Print a concise observation, VecNormalize, and actor audit summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _nonzero(values, threshold=0.0):
    return {
        name: float(value)
        for name, value in values.items()
        if float(value) > threshold
    }


def _print_fraction_group(title, values):
    selected = _nonzero(values)
    print(f"{title}: {selected if selected else 'none'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    document = json.loads(args.summary.expanduser().read_text(encoding="utf-8"))
    report = document.get("policy_stream", document)
    print(f"Report: {args.summary.expanduser().resolve()}")
    print(
        "Complete stream: "
        f"{report.get('complete_samples', 0)}/{report.get('samples', 0)} "
        f"({100.0 * float(report.get('complete_fraction', 0.0)):.1f}%)"
    )
    print(f"Invalid reasons: {report.get('invalid_reasons', {}) or 'none'}")
    print(
        "Overall fractions: "
        f"raw_clip={float(report.get('observation_clip_fraction', 0.0)):.4f}, "
        f"|normalized|>3={float(report.get('normalized_abs_gt_3_fraction', 0.0)):.4f}, "
        f"|normalized|>5={float(report.get('normalized_abs_gt_5_fraction', 0.0)):.4f}, "
        f"action_saturation={float(report.get('policy_action_saturation_fraction', 0.0)):.4f}"
    )
    _print_fraction_group(
        "Raw observation clipped by channel",
        report.get("observation_clip_fraction_by_label", {}),
    )
    _print_fraction_group(
        "Normalized magnitude > 3 by channel",
        report.get("normalized_abs_gt_3_fraction_by_label", {}),
    )
    _print_fraction_group(
        "Normalized magnitude > 5 by channel",
        report.get("normalized_abs_gt_5_fraction_by_label", {}),
    )
    _print_fraction_group(
        "Bounded action saturated by channel",
        report.get("policy_action_saturation_fraction_by_label", {}),
    )

    normalized = report.get("normalized_observation", {})
    actor = report.get("actor_mean", {})
    if normalized:
        print("Normalized means:")
        for name, stats in normalized.items():
            print(
                f"  {name:20s} mean={float(stats['mean']):+9.3f} "
                f"min={float(stats['min']):+9.3f} max={float(stats['max']):+9.3f}"
            )
    if actor:
        print("Actor means before action clipping:")
        for name, stats in actor.items():
            print(
                f"  {name:20s} mean={float(stats['mean']):+9.3f} "
                f"min={float(stats['min']):+9.3f} max={float(stats['max']):+9.3f}"
            )

    activation = document.get("policy_activation", {})
    if activation:
        print("Policy activation handoff:")
        print(
            "  ready="
            f"{activation.get('ready_samples', 0)}/"
            f"{activation.get('samples', 0)}, "
            "instantaneous_ready="
            f"{activation.get('instantaneous_ready_samples', 0)}, "
            "maximum_streak="
            f"{activation.get('maximum_consecutive_ready_samples', 0)}"
        )
        print(f"  reasons={activation.get('reason_counts', {}) or 'none'}")


if __name__ == "__main__":
    main()
