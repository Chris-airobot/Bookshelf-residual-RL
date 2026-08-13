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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--slot-clearance", type=float, default=0.003)
    parser.add_argument("--old-reset-noise", action="store_true")
    args = parser.parse_args()

    if args.old_reset_noise:
        noise = (math.radians(3.0), 0.008, 0.006, 0.003, math.radians(8.0))
    else:
        noise = (math.radians(1.5), 0.003, 0.003, 0.0015, math.radians(3.0))
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
