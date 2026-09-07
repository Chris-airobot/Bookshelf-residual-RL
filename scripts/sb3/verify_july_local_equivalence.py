#!/usr/bin/env python3
"""Verify the July checkpoint actor on the recorded real release observation."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import torch


RAW = np.asarray([
    0.0, -0.05896743759512901, 0.09403601288795471, -0.0013351569650694728,
    0.01059524342417717, -0.023005304858088493, -0.03684841841459274,
    -0.0002175553672714159, -0.0005290252156555653, 0.38823533058166504,
    -0.03493500128388405, 0.03920329734683037,
], dtype=np.float64)
SCALES = np.asarray([
    1.0, 0.08, 0.08, 0.05, 0.05, np.deg2rad(30.0),
    0.25, 0.25, 0.25, 1.0, 1.0, 1.0,
], dtype=np.float64)
EXPECTED_RELEASE = 0.5011026263237


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("stats", type=Path)
    args = parser.parse_args()
    stats = np.load(args.stats)
    policy_observation = np.clip(RAW / SCALES, -1.0, 1.0)
    normalized = np.clip(
        (policy_observation - stats["obs_mean"]) / np.sqrt(stats["obs_var"] + 1.0e-8),
        -10.0, 10.0,
    ).astype(np.float32)
    with zipfile.ZipFile(args.checkpoint) as archive:
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    hidden = torch.relu(torch.nn.functional.linear(
        torch.from_numpy(normalized), state["mlp_extractor.policy_net.0.weight"],
        state["mlp_extractor.policy_net.0.bias"],
    ))
    hidden = torch.relu(torch.nn.functional.linear(
        hidden, state["mlp_extractor.policy_net.2.weight"],
        state["mlp_extractor.policy_net.2.bias"],
    ))
    action = torch.nn.functional.linear(
        hidden, state["action_net.weight"], state["action_net.bias"]
    ).detach().numpy()
    print(f"observation_shape={policy_observation.shape} action_shape={action.shape}")
    print("normalized=" + np.array2string(normalized, precision=9))
    print("actor_mean=" + np.array2string(action, precision=9))
    print(f"release_delta={float(action[5] - EXPECTED_RELEASE):+.12g}")
    if policy_observation.shape != (12,) or action.shape != (6,):
        raise SystemExit("July observation/action contract mismatch")
    if not np.isclose(action[5], EXPECTED_RELEASE, rtol=0.0, atol=2.0e-6):
        raise SystemExit("July deterministic release output mismatch")


if __name__ == "__main__":
    main()
