#!/usr/bin/env python3
"""Check reproducibility and separation of bookshelf evaluation scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SB3_DIR = Path(__file__).resolve().parent / "sb3"
if str(_SB3_DIR) not in sys.path:
    sys.path.insert(0, str(_SB3_DIR))


def _load(path: str) -> dict:
    summary_path = Path(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("scenario_trace_complete", False):
        raise ValueError(f"Incomplete scenario trace: {summary_path}")
    if not summary.get("scenario_sha256"):
        raise ValueError(f"Missing scenario_sha256: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-same",
        nargs=2,
        action="append",
        default=[],
        metavar=("SUMMARY_A", "SUMMARY_B"),
        help="Require two traces to contain identical scenarios.",
    )
    parser.add_argument(
        "--expect-different",
        nargs=2,
        action="append",
        default=[],
        metavar=("SUMMARY_A", "SUMMARY_B"),
        help="Require two traces to contain different scenarios.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON audit report.")
    args = parser.parse_args()
    if not args.expect_same and not args.expect_different:
        parser.error("provide at least one --expect-same or --expect-different pair")

    checks = []
    for expectation, pairs in (("same", args.expect_same), ("different", args.expect_different)):
        for left_path, right_path in pairs:
            left = _load(left_path)
            right = _load(right_path)
            hashes_match = left["scenario_sha256"] == right["scenario_sha256"]
            passed = hashes_match if expectation == "same" else not hashes_match
            checks.append(
                {
                    "expectation": expectation,
                    "left": str(Path(left_path).resolve()),
                    "right": str(Path(right_path).resolve()),
                    "left_hash": left["scenario_sha256"],
                    "right_hash": right["scenario_sha256"],
                    "left_episodes": left["episode_count"],
                    "right_episodes": right["episode_count"],
                    "passed": passed,
                }
            )

    report = {"passed": all(check["passed"] for check in checks), "checks": checks}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(
            f"{status}: expected {check['expectation']}; "
            f"{check['left_hash'][:12]} vs {check['right_hash'][:12]} "
            f"({check['left_episodes']} and {check['right_episodes']} episodes)"
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
