"""Load reviewed operator joint-state snapshots without commanding a robot."""

import math
from pathlib import Path

import yaml


JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]


def load_joint_pose(path, expected_names=JOINT_NAMES):
    pose_path = Path(path).expanduser().resolve()
    with pose_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    names = list(document.get("name", []))
    positions = list(document.get("position", []))
    by_name = dict(zip(names, positions))
    missing = [name for name in expected_names if name not in by_name]
    if missing:
        raise ValueError(f"joint pose {pose_path} is missing {missing}")
    result = [float(by_name[name]) for name in expected_names]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"joint pose {pose_path} contains non-finite positions")
    return result
