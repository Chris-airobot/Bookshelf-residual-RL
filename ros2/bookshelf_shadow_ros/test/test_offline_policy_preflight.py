import csv

import numpy as np

from bookshelf_shadow_ros.offline_policy_preflight import (
    audit_recorded_activation_csv,
    audit_simulator_activation_samples,
)
from bookshelf_shadow_ros.policy_activation import ActivationEnvelope
from bookshelf_shadow_ros.policy_observation_math import OBSERVATION_LABELS


def _write_rows(path, raw, observation, normalized, count):
    fieldnames = [
        f"{prefix}_{label}"
        for prefix in ("raw", "obs", "normalized")
        for label in OBSERVATION_LABELS
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(count):
            row = {}
            for prefix, values in (
                ("raw", raw),
                ("obs", observation),
                ("normalized", normalized),
            ):
                row.update(
                    {
                        f"{prefix}_{label}": value
                        for label, value in zip(OBSERVATION_LABELS, values)
                    }
                )
            writer.writerow(row)


def _envelope():
    return ActivationEnvelope(
        lower=np.full(12, -4.0),
        upper=np.full(12, 4.0),
        source="test",
        metadata={},
    )


def test_preflight_reports_stable_local_activation(tmp_path):
    raw = [0.0, -0.186, 0.230, 0.0, 0.006, 0.0, -0.032, 0.0, 0.0, 0.0, 0.0, 0.0]
    observation = [0.0] * 12
    normalized = [0.0] * 12
    path = tmp_path / "samples.csv"
    _write_rows(path, raw, observation, normalized, 4)
    result = audit_recorded_activation_csv(
        path,
        envelope=_envelope(),
        stable_samples=3,
    )
    assert result["activation_ready_samples"] == 2
    assert result["first_ready_sample"] == 2
    assert result["final_activation_ready"]


def test_preflight_keeps_far_pose_blocked(tmp_path):
    raw = [0.0, -0.278, 0.417, 0.043, -0.063, -1.598, -0.124, 0.0, 0.0, 0.39, 0.0, 0.0]
    observation = [0.0] * 12
    normalized = [0.0, 0.0, 0.0, -10.0, -9.9, -10.0, 0.0, 0.0, 0.0, 2.8, 0.0, 0.0]
    path = tmp_path / "samples.csv"
    _write_rows(path, raw, observation, normalized, 5)
    result = audit_recorded_activation_csv(
        path,
        envelope=_envelope(),
        stable_samples=3,
    )
    assert result["activation_ready_samples"] == 0
    assert not result["final_activation_ready"]
    assert result["normalized_outlier_counts"] == {
        "lat_err": 5,
        "yaw_err": 5,
        "z_err": 5,
    }


def test_simulator_preinsert_audit_separates_batch_and_temporal_stability():
    valid_raw = np.array(
        [0.0, -0.186, 0.230, 0.0, 0.006, 0.0, -0.032, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    raw = np.tile(valid_raw, (5, 1))
    raw[-1, 3] = 0.10
    observation = np.zeros((5, 12))
    normalized = np.zeros((5, 12))

    result = audit_simulator_activation_samples(
        observation,
        normalized,
        raw,
        envelope=_envelope(),
        stable_samples=3,
    )
    assert result["instantaneous_ready_samples"] == 4
    assert result["instantaneous_ready_fraction"] == 0.8
    assert result["repeated_stability_passed"]
    assert not result["sample_order_is_temporal"]
    assert result["reason_counts"] == {
        "lateral error exceeds the local-policy limit": 1
    }
