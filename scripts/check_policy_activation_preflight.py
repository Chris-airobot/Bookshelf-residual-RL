#!/usr/bin/env python3
"""Produce a fail-closed offline activation report from a shadow audit."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from bookshelf_shadow_ros.offline_policy_preflight import (
    audit_recorded_activation_csv,
    load_policy_stream_summary,
)
from bookshelf_shadow_ros.policy_activation import load_activation_envelope


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--samples-csv", type=Path)
    parser.add_argument("--activation-envelope", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument(
        "--expect-activation",
        choices=("blocked", "ready", "either"),
        default="blocked",
    )
    parser.add_argument("--stable-samples", type=int, default=10)
    parser.add_argument("--minimum-complete-fraction", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    samples_path = (
        args.samples_csv.expanduser().resolve()
        if args.samples_csv
        else summary_path.with_name("policy_stream_samples.csv")
    )
    envelope_path = args.activation_envelope.expanduser().resolve()
    bundle_path = args.policy_bundle.expanduser().resolve()
    document, stream = load_policy_stream_summary(summary_path)
    envelope = load_activation_envelope(envelope_path)
    activation = audit_recorded_activation_csv(
        samples_path,
        envelope=envelope,
        stable_samples=args.stable_samples,
    )

    hardware_safe = document.get("hardware_commanded") is False
    complete_fraction = float(stream.get("complete_fraction", 0.0))
    complete_enough = complete_fraction >= args.minimum_complete_fraction
    if args.expect_activation == "blocked":
        expected_result = activation["activation_ready_samples"] == 0
    elif args.expect_activation == "ready":
        expected_result = activation["activation_ready_samples"] > 0
    else:
        expected_result = True
    passed = bool(hardware_safe and complete_enough and expected_result)

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "passed": passed,
        "hardware_commanded": False,
        "inputs": {
            "summary": str(summary_path),
            "samples_csv": str(samples_path),
            "activation_envelope": str(envelope_path),
            "policy_bundle": str(bundle_path),
            "activation_envelope_sha256": _sha256(envelope_path),
            "policy_bundle_sha256": _sha256(bundle_path),
        },
        "checks": {
            "recorded_report_hardware_commanded_false": hardware_safe,
            "complete_fraction": complete_fraction,
            "minimum_complete_fraction": args.minimum_complete_fraction,
            "complete_fraction_passed": complete_enough,
            "expected_activation": args.expect_activation,
            "expected_activation_passed": expected_result,
        },
        "activation": activation,
        "limitations": [
            "This is an offline software and distribution check; it does not check IK, collision, reachability, contact, or execution.",
            "The activation envelope is only as representative as its saved simulator samples.",
            "A ready result authorizes local-policy inference only; it does not authorize robot execution.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Report: {output}")
    print(f"Passed: {passed}")
    print(
        "Complete stream: "
        f"{stream.get('complete_samples', 0)}/{stream.get('samples', 0)} "
        f"({100.0 * complete_fraction:.1f}%)"
    )
    print(f"Expected activation: {args.expect_activation}")
    print(
        "Activation ready samples: "
        f"{activation['activation_ready_samples']}/{activation['samples']}"
    )
    print(
        "Maximum stable streak: "
        f"{activation['maximum_consecutive_ready_samples']}/"
        f"{activation['required_stable_samples']}"
    )
    print(f"Reasons: {activation['reason_counts'] or 'none'}")
    print("Hardware commanded: False")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
