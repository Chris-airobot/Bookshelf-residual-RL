"""Robust statistics and explicit promotion for a static RGB-D slot capture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .marker_book_calibration import (
    average_quaternions_xyzw,
    quaternion_angle_deg,
    quaternion_medoid_xyzw,
)
from .policy_observation_math import make_transform, matrix_to_quaternion_xyzw


APPROVAL_TOKEN = "VISUALLY_APPROVED_STATIC_SLOT"


@dataclass(frozen=True)
class StaticSlotSample:
    stamp_ns: int
    transform_base_slot: np.ndarray
    width_m: float
    confidence: float


def _statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Statistics require one or more finite scalar values.")
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _robust_limit(errors, *, minimum: float) -> float:
    errors = np.asarray(errors, dtype=np.float64)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    robust_sigma = 1.4826 * mad
    return max(float(minimum), median + 4.0 * robust_sigma)


class StaticSlotCaptureAccumulator:
    """Estimate one static slot pose while rejecting inconsistent detections."""

    def __init__(
        self,
        *,
        minimum_samples=60,
        minimum_inlier_fraction=0.80,
        maximum_translation_deviation_m=0.010,
        maximum_rotation_deviation_deg=5.0,
        maximum_width_deviation_m=0.005,
    ):
        self.minimum_samples = int(minimum_samples)
        self.minimum_inlier_fraction = float(minimum_inlier_fraction)
        self.maximum_translation_deviation_m = float(
            maximum_translation_deviation_m
        )
        self.maximum_rotation_deviation_deg = float(
            maximum_rotation_deviation_deg
        )
        self.maximum_width_deviation_m = float(maximum_width_deviation_m)
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be at least one")
        if not 0.0 < self.minimum_inlier_fraction <= 1.0:
            raise ValueError("minimum_inlier_fraction must be in (0, 1]")
        if min(
            self.maximum_translation_deviation_m,
            self.maximum_rotation_deviation_deg,
            self.maximum_width_deviation_m,
        ) <= 0.0:
            raise ValueError("Capture deviation limits must be positive")
        self.samples: list[StaticSlotSample] = []

    def add(self, sample: StaticSlotSample) -> None:
        transform = np.asarray(sample.transform_base_slot, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform_base_slot must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("transform_base_slot has an invalid homogeneous row")
        if not math.isfinite(sample.width_m) or sample.width_m <= 0.0:
            raise ValueError("width_m must be finite and positive")
        if not math.isfinite(sample.confidence) or not 0.0 <= sample.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        self.samples.append(sample)

    def result(self) -> dict:
        if len(self.samples) < self.minimum_samples:
            raise ValueError(
                f"Need at least {self.minimum_samples} samples; "
                f"received {len(self.samples)}."
            )

        transforms = np.asarray(
            [sample.transform_base_slot for sample in self.samples],
            dtype=np.float64,
        )
        translations = transforms[:, :3, 3]
        quaternions = np.asarray(
            [matrix_to_quaternion_xyzw(value[:3, :3]) for value in transforms]
        )
        widths = np.asarray([sample.width_m for sample in self.samples])
        confidences = np.asarray([sample.confidence for sample in self.samples])

        translation_seed = np.median(translations, axis=0)
        quaternion_seed = quaternion_medoid_xyzw(quaternions)
        width_seed = float(np.median(widths))
        translation_error = np.linalg.norm(
            translations - translation_seed[None, :], axis=1
        )
        rotation_error = np.asarray(
            [quaternion_angle_deg(value, quaternion_seed) for value in quaternions]
        )
        width_error = np.abs(widths - width_seed)

        translation_limit = min(
            self.maximum_translation_deviation_m,
            _robust_limit(translation_error, minimum=0.0005),
        )
        rotation_limit = min(
            self.maximum_rotation_deviation_deg,
            _robust_limit(rotation_error, minimum=0.25),
        )
        width_limit = min(
            self.maximum_width_deviation_m,
            _robust_limit(width_error, minimum=0.0005),
        )
        inliers = (
            (translation_error <= translation_limit)
            & (rotation_error <= rotation_limit)
            & (width_error <= width_limit)
        )
        inlier_count = int(np.count_nonzero(inliers))
        inlier_fraction = float(np.mean(inliers))
        if inlier_count < self.minimum_samples:
            raise ValueError(
                "Robust filtering retained fewer samples than minimum_samples: "
                f"{inlier_count}/{self.minimum_samples}."
            )
        if inlier_fraction < self.minimum_inlier_fraction:
            raise ValueError(
                "Static slot inlier fraction is below the required threshold: "
                f"{inlier_fraction:.3f} < {self.minimum_inlier_fraction:.3f}."
            )

        inlier_translations = translations[inliers]
        inlier_quaternions = quaternions[inliers]
        inlier_widths = widths[inliers]
        inlier_confidences = confidences[inliers]
        translation = np.mean(inlier_translations, axis=0)
        quaternion = average_quaternions_xyzw(inlier_quaternions)
        width = float(np.median(inlier_widths))

        translation_residuals = np.linalg.norm(
            inlier_translations - translation[None, :], axis=1
        )
        rotation_residuals = np.asarray(
            [quaternion_angle_deg(value, quaternion) for value in inlier_quaternions]
        )
        width_residuals = np.abs(inlier_widths - width)
        return {
            "transform_base_slot": make_transform(translation, quaternion),
            "translation_xyz": translation,
            "quaternion_xyzw": quaternion,
            "width_m": width,
            "confidence": float(np.median(inlier_confidences)),
            "input_samples": len(self.samples),
            "inlier_samples": inlier_count,
            "inlier_fraction": inlier_fraction,
            "translation_filter_limit_m": float(translation_limit),
            "rotation_filter_limit_deg": float(rotation_limit),
            "width_filter_limit_m": float(width_limit),
            "translation_residual_m": _statistics(translation_residuals),
            "rotation_residual_deg": _statistics(rotation_residuals),
            "width_residual_m": _statistics(width_residuals),
            "confidence_statistics": _statistics(inlier_confidences),
        }


def serializable_capture_result(result: dict) -> dict:
    """Convert an accumulator result to the candidate-report schema."""

    return {
        "translation_xyz": np.asarray(result["translation_xyz"]).tolist(),
        "quaternion_xyzw": np.asarray(result["quaternion_xyzw"]).tolist(),
        "width_m": float(result["width_m"]),
        "confidence": float(result["confidence"]),
        "transform_status": "captured_rgbd_static_unapproved",
    }


def promote_capture_candidate(
    candidate_report_path,
    template_directory,
    output_path,
    *,
    approval_token: str,
) -> dict:
    """Create one reviewed ROS parameter file without changing source configs."""

    if approval_token != APPROVAL_TOKEN:
        raise ValueError(
            f"Promotion requires --approval-token {APPROVAL_TOKEN}"
        )
    candidate_report_path = Path(candidate_report_path).expanduser().resolve()
    template_directory = Path(template_directory).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported static-slot candidate schema")
    if report.get("kind") != "bookshelf_static_slot_capture_candidate":
        raise ValueError("Input is not a static-slot capture candidate")
    if not report.get("valid"):
        raise ValueError(f"Candidate is invalid: {report.get('reason')}")
    if report.get("hardware_commanded") is not False:
        raise ValueError("Candidate does not prove hardware_commanded=false")
    if report.get("active_configuration_modified") is not False:
        raise ValueError("Candidate already claims to have modified configuration")

    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Candidate report has no candidate pose")
    transform = make_transform(
        candidate["translation_xyz"], candidate["quaternion_xyzw"]
    )
    translation = transform[:3, 3].tolist()
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3]).tolist()
    width = float(candidate["width_m"])
    confidence = float(candidate["confidence"])
    if width <= 0.0 or not 0.0 <= confidence <= 1.0:
        raise ValueError("Candidate width or confidence is invalid")

    template_names = {
        "static_slot_environment_check": "static_slot_environment_check.yaml",
        "calibrated_preinsert_target": "calibrated_preinsert_target.yaml",
        "policy_observation_adapter": (
            "policy_observation_adapter_policy_tool_candidate.yaml"
        ),
    }
    combined = {}
    for node_name, filename in template_names.items():
        source = template_directory / filename
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if node_name not in data or "ros__parameters" not in data[node_name]:
            raise ValueError(f"Template {source} has no parameters for {node_name}")
        combined[node_name] = copy.deepcopy(data[node_name])

    source_hash = hashlib.sha256(candidate_report_path.read_bytes()).hexdigest()
    approved_at = datetime.now().astimezone().isoformat()
    status = "captured_rgbd_static_human_approved_" + source_hash[:12]

    check = combined["static_slot_environment_check"]["ros__parameters"]
    check.update(
        {
            "static_slot_translation_xyz": translation,
            "static_slot_quaternion_xyzw": quaternion,
            "static_slot_width_m": width,
            "static_slot_transform_status": status,
        }
    )
    target = combined["calibrated_preinsert_target"]["ros__parameters"]
    target.update(
        {
            "static_slot_translation_xyz": translation,
            "static_slot_quaternion_xyzw": quaternion,
            "static_slot_width_m": width,
            "static_slot_confidence": confidence,
            "static_slot_transform_status": status,
        }
    )
    adapter = combined["policy_observation_adapter"]["ros__parameters"]
    adapter.update(
        {
            "configured_static_slot_translation_xyz": translation,
            "configured_static_slot_quaternion_xyzw": quaternion,
            "configured_static_slot_width_m": width,
            "configured_static_slot_confidence": confidence,
            "static_slot_transform_status": status,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Generated trial configuration. Do not edit slot values independently.\n"
        + yaml.safe_dump(combined, sort_keys=False),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "kind": "bookshelf_static_slot_trial_configuration",
        "generated_at": approved_at,
        "human_approval_recorded": True,
        "approval_token": APPROVAL_TOKEN,
        "hardware_commanded": False,
        "candidate_report": str(candidate_report_path),
        "candidate_report_sha256": source_hash,
        "trial_config": str(output_path),
        "transform_status": status,
        "slot": {
            "translation_xyz": translation,
            "quaternion_xyzw": quaternion,
            "width_m": width,
            "confidence": confidence,
        },
    }
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance
