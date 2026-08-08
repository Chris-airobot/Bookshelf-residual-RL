#!/usr/bin/env python3
"""Check that the handoff gate accepts saved simulator pre-insertion states."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from bookshelf_shadow_ros.offline_policy_preflight import (
    audit_simulator_activation_samples,
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
    parser.add_argument("samples", type=Path)
    parser.add_argument("--activation-envelope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ready-fraction", type=float, default=0.95)
    parser.add_argument("--stable-samples", type=int, default=10)
    parser.add_argument(
        "--observation-key",
        default="simulator_preinsert_observation",
    )
    parser.add_argument(
        "--normalized-key",
        default="simulator_preinsert_normalized_observation",
    )
    parser.add_argument(
        "--raw-metrics-key",
        default="simulator_preinsert_raw_metrics",
    )
    args = parser.parse_args()

    if not 0.0 <= args.minimum_ready_fraction <= 1.0:
        raise ValueError("minimum-ready-fraction must be within [0, 1].")
    samples_path = args.samples.expanduser().resolve()
    envelope_path = args.activation_envelope.expanduser().resolve()
    envelope = load_activation_envelope(envelope_path)
    keys = (args.observation_key, args.normalized_key, args.raw_metrics_key)
    with np.load(samples_path, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise KeyError(f"Simulator sample archive is missing keys: {missing}")
        activation = audit_simulator_activation_samples(
            archive[args.observation_key],
            archive[args.normalized_key],
            archive[args.raw_metrics_key],
            envelope=envelope,
            stable_samples=args.stable_samples,
        )

    ready_fraction_passed = (
        activation["instantaneous_ready_fraction"] >= args.minimum_ready_fraction
    )
    passed = bool(
        ready_fraction_passed and activation["repeated_stability_passed"]
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "passed": passed,
        "hardware_commanded": False,
        "inputs": {
            "samples": str(samples_path),
            "samples_sha256": _sha256(samples_path),
            "activation_envelope": str(envelope_path),
            "activation_envelope_sha256": _sha256(envelope_path),
            "keys": list(keys),
        },
        "checks": {
            "minimum_ready_fraction": args.minimum_ready_fraction,
            "ready_fraction_passed": ready_fraction_passed,
            "representative_repeated_stability_passed": activation[
                "repeated_stability_passed"
            ],
        },
        "activation": activation,
        "limitations": [
            "Simulator rows are parallel environment samples, not a temporal stream.",
            "The stability tracker is tested by repeating one representative acceptable sample.",
            "This report does not authorize robot planning or execution.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Report: {output}")
    print(f"Passed: {passed}")
    print(
        "Instantaneous simulator readiness: "
        f"{activation['instantaneous_ready_samples']}/{activation['samples']} "
        f"({100.0 * activation['instantaneous_ready_fraction']:.1f}%)"
    )
    print(
        "Representative repeated stability: "
        f"{activation['repeated_stability_passed']} "
        f"({activation['repeated_stability_samples']} cycles)"
    )
    print(f"Reasons: {activation['reason_counts'] or 'none'}")
    print("Hardware commanded: False")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
