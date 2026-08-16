#!/usr/bin/env python3
"""Generate a policy-independent frozen bookshelf evaluation scenario bank."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source/bookshelf/bookshelf/tasks/direct/bookshelf/frozen_scenario_bank.py"
SPEC = importlib.util.spec_from_file_location("bookshelf_frozen_scenario_bank", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load frozen scenario bank helpers from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CURRENT_RESET_NOISE = (
    math.radians(1.5),
    0.003,
    0.003,
    0.0015,
    math.radians(3.0),
)
FINAL_TRAINING_RESET_NOISE = (
    math.radians(3.0),
    0.008,
    0.006,
    0.003,
    math.radians(8.0),
)
RESET_NOISE_KEYS = (
    "arm_joint_noise",
    "grasp_x_jitter",
    "grasp_y_jitter",
    "grasp_z_jitter",
    "grasp_yaw_jitter",
)


def scaled_training_reset_noise(scale: float) -> tuple[float, ...]:
    """Scale the final training randomization while preserving its proportions."""

    value = float(scale)
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("reset-noise scale must be finite and non-negative")
    return tuple(value * maximum for maximum in FINAL_TRAINING_RESET_NOISE)


def resolve_reset_noise(
    *, old_reset_noise: bool, reset_noise_scale: float | None
) -> tuple[float, ...]:
    if old_reset_noise and reset_noise_scale is not None:
        raise ValueError("--old-reset-noise and --reset-noise-scale are mutually exclusive")
    if reset_noise_scale is not None:
        return scaled_training_reset_noise(reset_noise_scale)
    return FINAL_TRAINING_RESET_NOISE if old_reset_noise else CURRENT_RESET_NOISE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--slot-clearance", type=float, default=0.003)
    parser.add_argument("--old-reset-noise", action="store_true")
    parser.add_argument(
        "--reset-noise-scale",
        type=float,
        default=None,
        help=(
            "Scale all final-training reset perturbation maxima together; "
            "1.0 is the training boundary and values above 1.0 are OOD."
        ),
    )
    args = parser.parse_args()

    try:
        noise = resolve_reset_noise(
            old_reset_noise=args.old_reset_noise,
            reset_noise_scale=args.reset_noise_scale,
        )
    except ValueError as error:
        parser.error(str(error))
    output = MODULE.write_generated_frozen_scenario_bank(
        args.output,
        scenario_count=args.scenarios,
        seed=args.seed,
        slot_clearance_min=args.slot_clearance,
        slot_clearance_max=args.slot_clearance,
        slot_pitch=0.034,
        row_book_count=10,
        side_book_merge_probability=0.35,
        arm_joint_noise=noise[0],
        grasp_x_jitter=noise[1],
        grasp_y_jitter=noise[2],
        grasp_z_jitter=noise[3],
        grasp_yaw_jitter=noise[4],
    )
    bank = MODULE.load_frozen_scenario_bank(output)
    print(f"Frozen bank: {output}")
    print(f"Scenarios: {bank['scenario_count']}")
    print(f"SHA256: {bank['scenario_sha256']}")
    print(f"Generation: {bank['source']['generation']}")


if __name__ == "__main__":
    main()
