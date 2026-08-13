#!/usr/bin/env python3
"""Export a complete evaluation trace as a frozen bookshelf scenario bank."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/bookshelf/bookshelf/tasks/direct/bookshelf/frozen_scenario_bank.py"
)
SPEC = importlib.util.spec_from_file_location("bookshelf_frozen_scenario_bank", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load frozen scenario bank helpers from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    output = MODULE.write_frozen_scenario_bank(args.trace_summary, args.output)
    bank = MODULE.load_frozen_scenario_bank(output)
    print(f"Frozen bank: {output}")
    print(f"Scenarios: {bank['scenario_count']}")
    print(f"SHA256: {bank['scenario_sha256']}")
    print(f"Source trace: {bank['source']['trace_summary']}")


if __name__ == "__main__":
    main()
