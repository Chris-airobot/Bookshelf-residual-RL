#!/usr/bin/env python3
"""Verify that evaluation traces replayed one complete frozen bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_replays(bank_path: str | Path, summary_paths: list[str | Path]) -> dict[str, Any]:
    bank_file = Path(bank_path).resolve()
    bank = json.loads(bank_file.read_text(encoding="utf-8"))
    bank_hash = bank.get("scenario_sha256")
    bank_count = int(bank.get("scenario_count", -1))
    if not bank_hash or bank_count <= 0:
        raise ValueError(f"Invalid frozen scenario bank: {bank_file}")

    checks = []
    for summary_path in summary_paths:
        path = Path(summary_path).resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        metadata = summary.get("metadata", {}).get("frozen_scenario_bank") or {}
        coverage = summary.get("frozen_scenario_bank_coverage") or {}
        reasons = []
        if metadata.get("scenario_sha256") != bank_hash:
            reasons.append("bank hash mismatch")
        if int(metadata.get("scenario_count", -1)) != bank_count:
            reasons.append("bank count mismatch")
        if not summary.get("scenario_trace_complete", False):
            reasons.append("scenario trace incomplete")
        if not coverage.get("complete", False):
            reasons.append("bank coverage incomplete")
        if int(summary.get("episode_count", -1)) != bank_count:
            reasons.append("episode count mismatch")
        checks.append(
            {
                "summary": str(path),
                "passed": not reasons,
                "reasons": reasons,
                "outcomes": summary.get("outcomes", {}),
                "coverage": coverage,
            }
        )
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "bank": str(bank_file),
        "scenario_sha256": bank_hash,
        "scenario_count": bank_count,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit_replays(args.bank, args.summaries)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        detail = "complete frozen-bank replay" if check["passed"] else ", ".join(check["reasons"])
        print(f"{status}: {check['summary']} ({detail})")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
