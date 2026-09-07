#!/usr/bin/env python3
"""Evaluate recorded real release observations through SB3 checkpoint actors."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from export_vecnormalize_stats import _StatsUnpickler


def vecnormalize_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.name.replace("model", "model_vecnormalize").replace(".zip", ".pkl"))


def normalization(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    path = vecnormalize_path(checkpoint)
    try:
        with path.open("rb") as stream:
            vec = _StatsUnpickler(stream).load()
        return (
            np.asarray(vec.obs_rms.mean, dtype=np.float64),
            np.asarray(vec.obs_rms.var, dtype=np.float64),
            float(getattr(vec, "epsilon", 1.0e-8)),
            float(vec.clip_obs),
        )
    except Exception:
        stats_path = path.with_name(f"{path.stem}_stats.npz")
        stats = np.load(stats_path)
        return (
            np.asarray(stats["obs_mean"], dtype=np.float64),
            np.asarray(stats["obs_var"], dtype=np.float64),
            1.0e-8,
            10.0,
        )


def actor_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    with zipfile.ZipFile(checkpoint) as archive:
        return torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)


def actor_mean(state: dict[str, torch.Tensor], normalized: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(normalized, dtype=torch.float32)
    value = torch.relu(torch.nn.functional.linear(
        value, state["mlp_extractor.policy_net.0.weight"], state["mlp_extractor.policy_net.0.bias"]
    ))
    value = torch.relu(torch.nn.functional.linear(
        value, state["mlp_extractor.policy_net.2.weight"], state["mlp_extractor.policy_net.2.bias"]
    ))
    value = torch.nn.functional.linear(value, state["action_net.weight"], state["action_net.bias"])
    return value.detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    threshold = float(inputs["release_threshold"])
    x_scale = float(inputs["residual_x_scale_m"])
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, checkpoint_text = line.split("\t", 1)
        checkpoint = Path(checkpoint_text)
        mean, var, epsilon, clip_obs = normalization(checkpoint)
        state = actor_state(checkpoint)
        for case, observation_values in inputs["cases"].items():
            observation = np.asarray(observation_values, dtype=np.float64)
            normalized = np.clip((observation - mean) / np.sqrt(var + epsilon), -clip_obs, clip_obs)
            action_mean = actor_mean(state, normalized.astype(np.float32))
            clipped = np.clip(action_mean, -1.0, 1.0)
            rows.append({
                "policy": label,
                "case": case,
                "release_actor_output": float(action_mean[5]),
                "release_margin": float(action_mean[5] - threshold),
                "release_threshold_crossed": int(float(clipped[5]) > threshold),
                "insertion_residual_x_action": float(clipped[0]),
                "insertion_residual_x_m": float(clipped[0] * x_scale),
                "checkpoint": str(checkpoint),
                "vecnormalize": str(vecnormalize_path(checkpoint)),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} counterfactual rows to {args.output}")


if __name__ == "__main__":
    main()
