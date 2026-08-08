#!/usr/bin/env python3
"""Generate a reviewed local-policy activation envelope from simulator samples."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from bookshelf_shadow_ros.policy_activation import build_activation_envelope
from bookshelf_shadow_ros.policy_observation_math import OBSERVATION_LABELS


DEFAULT_KEYS = (
    "simulator_reset_normalized_observation",
    "simulator_preinsert_normalized_observation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", default=list(DEFAULT_KEYS))
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--margin", type=float, default=0.50)
    parser.add_argument("--maximum-abs-bound", type=float, default=5.0)
    args = parser.parse_args()

    samples_path = args.samples.expanduser().resolve()
    arrays = []
    counts = {}
    with np.load(samples_path, allow_pickle=False) as archive:
        missing = [key for key in args.keys if key not in archive]
        if missing:
            raise KeyError(f"Simulator sample archive is missing keys: {missing}")
        for key in args.keys:
            values = np.asarray(archive[key], dtype=np.float64)
            arrays.append(values)
            counts[key] = int(values.shape[0]) if values.ndim else 0
    combined = np.concatenate(arrays, axis=0)
    lower, upper = build_activation_envelope(
        combined,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        margin=args.margin,
        maximum_abs_bound=args.maximum_abs_bound,
    )

    document = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "labels": list(OBSERVATION_LABELS),
        "lower": lower.astype(float).tolist(),
        "upper": upper.astype(float).tolist(),
        "source": str(samples_path),
        "metadata": {
            "source_sha256": _sha256(samples_path),
            "source_keys": list(args.keys),
            "sample_counts": counts,
            "combined_sample_count": int(combined.shape[0]),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "margin": args.margin,
            "maximum_abs_bound": args.maximum_abs_bound,
            "hardware_commanded": False,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Envelope: {output}")
    print(f"Simulator samples: {combined.shape[0]}")
    print(f"Source SHA256: {document['metadata']['source_sha256']}")
    for label, low, high in zip(OBSERVATION_LABELS, lower, upper):
        print(f"  {label:20s} [{low:+8.3f}, {high:+8.3f}]")
    print("Hardware commanded: False")


if __name__ == "__main__":
    main()
