"""Offline validation helpers for the read-only bookshelf shadow pipeline."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from .policy_observation_math import (
    OBSERVATION_LABELS,
    compute_policy_observation,
    quaternion_xyzw_to_matrix,
)
from .policy_shadow_math import (
    MOTION_LABELS,
    NumpyActorBundle,
    NominalInsertConfig,
    POLICY_ACTION_LABELS,
    ResidualMotionConfig,
    combine_motion_delta,
    compute_insert_nominal_delta,
    scale_residual_action,
)


@dataclass(frozen=True)
class ShadowCaseResult:
    raw_metrics: list[float]
    observation: list[float]
    normalized_observation: list[float]
    actor_mean: list[float]
    policy_action: list[float]
    nominal_delta: list[float]
    residual_delta: list[float]
    final_delta: list[float]
    release_requested: bool


class SlotAuditAccumulator:
    """Collect detector outputs and summarize stability without ROS dependencies."""

    def __init__(self, minimum_confidence=0.60):
        self.minimum_confidence = float(minimum_confidence)
        self.rows = []

    def add(self, confidence, *, width=None, position=None, quaternion_xyzw=None):
        confidence = float(confidence)
        valid = math.isfinite(confidence) and confidence >= self.minimum_confidence
        row = {
            "confidence": confidence,
            "valid": bool(valid),
            "width_m": math.nan,
            "position_x_m": math.nan,
            "position_y_m": math.nan,
            "position_z_m": math.nan,
            "quaternion_x": math.nan,
            "quaternion_y": math.nan,
            "quaternion_z": math.nan,
            "quaternion_w": math.nan,
        }
        if valid:
            try:
                width = float(width)
                position = np.asarray(position, dtype=np.float64)
                quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
                if (
                    not math.isfinite(width)
                    or position.shape != (3,)
                    or quaternion.shape != (4,)
                    or not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(quaternion))
                ):
                    valid = False
                else:
                    quaternion_norm = float(np.linalg.norm(quaternion))
                    if quaternion_norm < 1.0e-12:
                        valid = False
                    else:
                        quaternion /= quaternion_norm
            except (TypeError, ValueError):
                valid = False
        row["valid"] = bool(valid)
        if valid:
            row.update(
                {
                    "width_m": width,
                    "position_x_m": float(position[0]),
                    "position_y_m": float(position[1]),
                    "position_z_m": float(position[2]),
                    "quaternion_x": float(quaternion[0]),
                    "quaternion_y": float(quaternion[1]),
                    "quaternion_z": float(quaternion[2]),
                    "quaternion_w": float(quaternion[3]),
                }
            )
        self.rows.append(row)

    def summary(self) -> dict:
        total = len(self.rows)
        valid_rows = [row for row in self.rows if row["valid"]]
        valid_count = len(valid_rows)
        result = {
            "samples": total,
            "valid_samples": valid_count,
            "valid_fraction": valid_count / total if total else 0.0,
            "minimum_confidence": self.minimum_confidence,
        }
        if not valid_rows:
            return result

        confidence = np.asarray([row["confidence"] for row in valid_rows])
        width = np.asarray([row["width_m"] for row in valid_rows])
        position = np.asarray(
            [
                [row["position_x_m"], row["position_y_m"], row["position_z_m"]]
                for row in valid_rows
            ]
        )
        quaternion = np.asarray(
            [
                [
                    row["quaternion_x"],
                    row["quaternion_y"],
                    row["quaternion_z"],
                    row["quaternion_w"],
                ]
                for row in valid_rows
            ]
        )
        reference = quaternion[0]
        quaternion = np.where(
            (quaternion @ reference)[:, None] < 0.0,
            -quaternion,
            quaternion,
        )
        mean_quaternion = np.mean(quaternion, axis=0)
        mean_quaternion /= max(float(np.linalg.norm(mean_quaternion)), 1.0e-12)
        orientation_error = 2.0 * np.arccos(
            np.clip(np.abs(quaternion @ mean_quaternion), 0.0, 1.0)
        )

        result.update(
            {
                "confidence": _scalar_statistics(confidence),
                "width_m": _scalar_statistics(width),
                "position_mean_m": np.mean(position, axis=0).astype(float).tolist(),
                "position_std_m": np.std(position, axis=0).astype(float).tolist(),
                "position_peak_to_peak_m": np.ptp(position, axis=0).astype(float).tolist(),
                "mean_quaternion_xyzw": mean_quaternion.astype(float).tolist(),
                "orientation_error_deg": _scalar_statistics(
                    np.degrees(orientation_error)
                ),
            }
        )
        if valid_count > 1:
            result["frame_to_frame_width_change_m"] = _scalar_statistics(
                np.abs(np.diff(width))
            )
            result["frame_to_frame_position_change_m"] = _scalar_statistics(
                np.linalg.norm(np.diff(position, axis=0), axis=1)
            )
        return result


class PolicyStreamAuditAccumulator:
    """Summarize the complete read-only observation and policy diagnostic stream."""

    def __init__(self, reference_slot_width_m=0.0):
        self.reference_slot_width_m = float(reference_slot_width_m)
        self.rows = []
        self.invalid_reasons = {}
        self.book_pose_sources = {}
        self.eef_book_transform_statuses = {}
        self.policy_tool_transform_statuses = {}
        self.slot_pose_sources = {}
        self.static_slot_transform_statuses = {}

    def add_invalid(self, reason):
        reason = str(reason or "unspecified")
        self.invalid_reasons[reason] = self.invalid_reasons.get(reason, 0) + 1

    def add(
        self,
        *,
        confidence,
        slot_width,
        slot_position,
        slot_quaternion_xyzw,
        book_position,
        book_quaternion_xyzw,
        raw_metrics,
        observation,
        normalized_observation,
        actor_mean,
        policy_action,
        nominal_delta,
        residual_delta,
        final_delta,
        book_pose_source,
        eef_book_transform_status,
        slot_pose_source="unknown",
        static_slot_transform_status="unknown",
        policy_tool_transform_status="unknown",
    ):
        vectors = {
            "slot_position": (slot_position, 3),
            "slot_quaternion_xyzw": (slot_quaternion_xyzw, 4),
            "book_position": (book_position, 3),
            "book_quaternion_xyzw": (book_quaternion_xyzw, 4),
            "raw_metrics": (raw_metrics, len(OBSERVATION_LABELS)),
            "observation": (observation, len(OBSERVATION_LABELS)),
            "normalized_observation": (
                normalized_observation,
                len(OBSERVATION_LABELS),
            ),
            "actor_mean": (actor_mean, len(POLICY_ACTION_LABELS)),
            "policy_action": (policy_action, len(POLICY_ACTION_LABELS)),
            "nominal_delta": (nominal_delta, len(MOTION_LABELS)),
            "residual_delta": (residual_delta, len(MOTION_LABELS)),
            "final_delta": (final_delta, len(MOTION_LABELS)),
        }
        parsed = {}
        try:
            confidence = float(confidence)
            slot_width = float(slot_width)
            if not math.isfinite(confidence) or not math.isfinite(slot_width):
                raise ValueError("confidence or slot width is non-finite")
            for name, (value, size) in vectors.items():
                array = np.asarray(value, dtype=np.float64).reshape(-1)
                if array.shape != (size,):
                    raise ValueError(
                        f"{name} must have shape ({size},), got {array.shape}"
                    )
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains non-finite values")
                parsed[name] = array
            for name in ("slot_quaternion_xyzw", "book_quaternion_xyzw"):
                norm = float(np.linalg.norm(parsed[name]))
                if norm < 1.0e-12:
                    raise ValueError(f"{name} is zero")
                parsed[name] = parsed[name] / norm
        except (TypeError, ValueError) as error:
            self.add_invalid(f"malformed complete stream: {error}")
            return False

        slot_rotation = quaternion_xyzw_to_matrix(
            parsed["slot_quaternion_xyzw"]
        )
        row = {
            "confidence": confidence,
            "slot_width_m": slot_width,
            "slot_position": parsed["slot_position"],
            "slot_quaternion_xyzw": parsed["slot_quaternion_xyzw"],
            "slot_axis_x_base": slot_rotation[:, 0],
            "slot_axis_y_base": slot_rotation[:, 1],
            "slot_axis_z_base": slot_rotation[:, 2],
            "book_position": parsed["book_position"],
            "book_quaternion_xyzw": parsed["book_quaternion_xyzw"],
            "raw_metrics": parsed["raw_metrics"],
            "observation": parsed["observation"],
            "normalized_observation": parsed["normalized_observation"],
            "actor_mean": parsed["actor_mean"],
            "policy_action": parsed["policy_action"],
            "nominal_delta": parsed["nominal_delta"],
            "residual_delta": parsed["residual_delta"],
            "final_delta": parsed["final_delta"],
            "book_pose_source": str(book_pose_source or "unknown"),
            "eef_book_transform_status": str(
                eef_book_transform_status or "unknown"
            ),
            "policy_tool_transform_status": str(
                policy_tool_transform_status or "unknown"
            ),
            "slot_pose_source": str(slot_pose_source or "unknown"),
            "static_slot_transform_status": str(
                static_slot_transform_status or "unknown"
            ),
        }
        self.rows.append(row)
        source = row["book_pose_source"]
        status = row["eef_book_transform_status"]
        self.book_pose_sources[source] = self.book_pose_sources.get(source, 0) + 1
        self.eef_book_transform_statuses[status] = (
            self.eef_book_transform_statuses.get(status, 0) + 1
        )
        tool_status = row["policy_tool_transform_status"]
        self.policy_tool_transform_statuses[tool_status] = (
            self.policy_tool_transform_statuses.get(tool_status, 0) + 1
        )
        slot_source = row["slot_pose_source"]
        slot_status = row["static_slot_transform_status"]
        self.slot_pose_sources[slot_source] = (
            self.slot_pose_sources.get(slot_source, 0) + 1
        )
        self.static_slot_transform_statuses[slot_status] = (
            self.static_slot_transform_statuses.get(slot_status, 0) + 1
        )
        return True

    def summary(self) -> dict:
        valid_count = len(self.rows)
        invalid_count = sum(self.invalid_reasons.values())
        total = valid_count + invalid_count
        result = {
            "samples": total,
            "complete_samples": valid_count,
            "invalid_samples": invalid_count,
            "complete_fraction": valid_count / total if total else 0.0,
            "invalid_reasons": dict(sorted(self.invalid_reasons.items())),
            "book_pose_sources": dict(sorted(self.book_pose_sources.items())),
            "eef_book_transform_statuses": dict(
                sorted(self.eef_book_transform_statuses.items())
            ),
            "policy_tool_transform_statuses": dict(
                sorted(self.policy_tool_transform_statuses.items())
            ),
            "slot_pose_sources": dict(sorted(self.slot_pose_sources.items())),
            "static_slot_transform_statuses": dict(
                sorted(self.static_slot_transform_statuses.items())
            ),
            "reference_slot_width_m": (
                self.reference_slot_width_m
                if self.reference_slot_width_m > 0.0
                else None
            ),
        }
        if not self.rows:
            return result

        confidence = np.asarray([row["confidence"] for row in self.rows])
        width = np.asarray([row["slot_width_m"] for row in self.rows])
        result.update(
            {
                "confidence": _scalar_statistics(confidence),
                "slot_width_m": _scalar_statistics(width),
                "slot_pose_base": _pose_statistics(
                    [row["slot_position"] for row in self.rows],
                    [row["slot_quaternion_xyzw"] for row in self.rows],
                ),
                "book_pose_base": _pose_statistics(
                    [row["book_position"] for row in self.rows],
                    [row["book_quaternion_xyzw"] for row in self.rows],
                ),
                "slot_axes_in_base": {
                    axis: _axis_statistics(
                        [row[f"slot_axis_{axis}_base"] for row in self.rows]
                    )
                    for axis in ("x", "y", "z")
                },
                "raw_metrics": _labelled_vector_statistics(
                    [row["raw_metrics"] for row in self.rows],
                    OBSERVATION_LABELS,
                ),
                "observation_12d": _labelled_vector_statistics(
                    [row["observation"] for row in self.rows],
                    OBSERVATION_LABELS,
                ),
                "normalized_observation": _labelled_vector_statistics(
                    [row["normalized_observation"] for row in self.rows],
                    OBSERVATION_LABELS,
                ),
                "actor_mean": _labelled_vector_statistics(
                    [row["actor_mean"] for row in self.rows],
                    POLICY_ACTION_LABELS,
                ),
                "policy_action": _labelled_vector_statistics(
                    [row["policy_action"] for row in self.rows],
                    POLICY_ACTION_LABELS,
                ),
                "nominal_delta": _labelled_vector_statistics(
                    [row["nominal_delta"] for row in self.rows],
                    MOTION_LABELS,
                ),
                "residual_delta": _labelled_vector_statistics(
                    [row["residual_delta"] for row in self.rows],
                    MOTION_LABELS,
                ),
                "final_delta": _labelled_vector_statistics(
                    [row["final_delta"] for row in self.rows],
                    MOTION_LABELS,
                ),
            }
        )
        observations = np.asarray([row["observation"] for row in self.rows])
        normalized = np.asarray(
            [row["normalized_observation"] for row in self.rows]
        )
        actions = np.asarray([row["policy_action"] for row in self.rows])
        result["observation_clip_fraction"] = float(
            np.mean(np.abs(observations) >= 1.0 - 1.0e-7)
        )
        result["observation_clip_fraction_by_label"] = _fraction_by_label(
            np.abs(observations) >= 1.0 - 1.0e-7,
            OBSERVATION_LABELS,
        )
        result["normalized_abs_gt_3_fraction"] = float(
            np.mean(np.abs(normalized) > 3.0)
        )
        result["normalized_abs_gt_3_fraction_by_label"] = _fraction_by_label(
            np.abs(normalized) > 3.0,
            OBSERVATION_LABELS,
        )
        result["normalized_abs_gt_5_fraction"] = float(
            np.mean(np.abs(normalized) > 5.0)
        )
        result["normalized_abs_gt_5_fraction_by_label"] = _fraction_by_label(
            np.abs(normalized) > 5.0,
            OBSERVATION_LABELS,
        )
        result["policy_action_saturation_fraction"] = float(
            np.mean(np.abs(actions) >= 1.0 - 1.0e-7)
        )
        result["policy_action_saturation_fraction_by_label"] = _fraction_by_label(
            np.abs(actions) >= 1.0 - 1.0e-7,
            POLICY_ACTION_LABELS,
        )
        if self.reference_slot_width_m > 0.0:
            error = width - self.reference_slot_width_m
            result["slot_width_error_m"] = {
                **_scalar_statistics(error),
                "mean_absolute": float(np.mean(np.abs(error))),
            }
        return result

    def csv_rows(self):
        for index, row in enumerate(self.rows):
            output = {
                "sample": index,
                "confidence": row["confidence"],
                "slot_width_m": row["slot_width_m"],
                "book_pose_source": row["book_pose_source"],
                "slot_pose_source": row["slot_pose_source"],
                "static_slot_transform_status": row[
                    "static_slot_transform_status"
                ],
                "eef_book_transform_status": row[
                    "eef_book_transform_status"
                ],
                "policy_tool_transform_status": row[
                    "policy_tool_transform_status"
                ],
            }
            for prefix in ("slot_position", "book_position"):
                for axis, value in zip("xyz", row[prefix]):
                    output[f"{prefix}_{axis}_m"] = float(value)
            for prefix in ("slot_quaternion_xyzw", "book_quaternion_xyzw"):
                for axis, value in zip("xyzw", row[prefix]):
                    output[f"{prefix}_{axis}"] = float(value)
            for axis_name in ("x", "y", "z"):
                for component, value in zip(
                    "xyz", row[f"slot_axis_{axis_name}_base"]
                ):
                    output[f"slot_axis_{axis_name}_base_{component}"] = float(
                        value
                    )
            for prefix, labels in (
                ("raw", OBSERVATION_LABELS),
                ("obs", OBSERVATION_LABELS),
                ("normalized", OBSERVATION_LABELS),
                ("actor", POLICY_ACTION_LABELS),
                ("action", POLICY_ACTION_LABELS),
                ("nominal", MOTION_LABELS),
                ("residual", MOTION_LABELS),
                ("final", MOTION_LABELS),
            ):
                source_key = {
                    "raw": "raw_metrics",
                    "obs": "observation",
                    "normalized": "normalized_observation",
                    "actor": "actor_mean",
                    "action": "policy_action",
                    "nominal": "nominal_delta",
                    "residual": "residual_delta",
                    "final": "final_delta",
                }[prefix]
                for label, value in zip(labels, row[source_key]):
                    output[f"{prefix}_{label}"] = float(value)
            yield output


class PolicyActivationAuditAccumulator:
    """Summarize the explicit global-planner to local-policy handoff stream."""

    def __init__(self):
        self.samples = 0
        self.ready_samples = 0
        self.instantaneous_ready_samples = 0
        self.maximum_consecutive_ready_samples = 0
        self.reason_counts = {}
        self.normalized_outlier_counts = {}
        self.envelope_outlier_counts = {}
        self.geometry_rows = []
        self.invalid_payloads = 0

    def add(self, payload):
        try:
            ready = bool(payload["ready"])
            instantaneous_ready = bool(payload["instantaneous_ready"])
            consecutive = int(payload["consecutive_ready_samples"])
            required = int(payload["required_stable_samples"])
            reasons = [str(value) for value in payload.get("reasons", [])]
            normalized_outliers = dict(payload.get("normalized_outliers", {}))
            envelope_outliers = dict(payload.get("envelope_outliers", {}))
            geometry = {
                str(name): float(value)
                for name, value in dict(payload.get("geometry", {})).items()
            }
            if consecutive < 0 or required < 1:
                raise ValueError("activation sample counts are invalid")
            if not all(math.isfinite(value) for value in geometry.values()):
                raise ValueError("activation geometry contains non-finite values")
        except (KeyError, TypeError, ValueError):
            self.invalid_payloads += 1
            return False

        self.samples += 1
        self.ready_samples += int(ready)
        self.instantaneous_ready_samples += int(instantaneous_ready)
        self.maximum_consecutive_ready_samples = max(
            self.maximum_consecutive_ready_samples,
            consecutive,
        )
        for reason in reasons:
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        for label in normalized_outliers:
            self.normalized_outlier_counts[label] = (
                self.normalized_outlier_counts.get(label, 0) + 1
            )
        for label in envelope_outliers:
            self.envelope_outlier_counts[label] = (
                self.envelope_outlier_counts.get(label, 0) + 1
            )
        if geometry:
            self.geometry_rows.append(geometry)
        return True

    def summary(self):
        total = self.samples
        geometry = {}
        if self.geometry_rows:
            shared_labels = sorted(
                set.intersection(*(set(row) for row in self.geometry_rows))
            )
            geometry = {
                label: _scalar_statistics(
                    [row[label] for row in self.geometry_rows]
                )
                for label in shared_labels
            }
        return {
            "samples": total,
            "invalid_payloads": self.invalid_payloads,
            "ready_samples": self.ready_samples,
            "ready_fraction": self.ready_samples / total if total else 0.0,
            "instantaneous_ready_samples": self.instantaneous_ready_samples,
            "instantaneous_ready_fraction": (
                self.instantaneous_ready_samples / total if total else 0.0
            ),
            "maximum_consecutive_ready_samples": (
                self.maximum_consecutive_ready_samples
            ),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "normalized_outlier_counts": dict(
                sorted(self.normalized_outlier_counts.items())
            ),
            "envelope_outlier_counts": dict(
                sorted(self.envelope_outlier_counts.items())
            ),
            "geometry": geometry,
        }


def _scalar_statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }

def _labelled_vector_statistics(values, labels) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        label: _scalar_statistics(values[:, index])
        for index, label in enumerate(labels)
    }


def _fraction_by_label(mask, labels) -> dict:
    mask = np.asarray(mask, dtype=bool)
    return {
        label: float(np.mean(mask[:, index]))
        for index, label in enumerate(labels)
    }


def _axis_statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm > 1.0e-12:
        mean /= mean_norm
    angular_error = np.degrees(
        np.arccos(np.clip(values @ mean, -1.0, 1.0))
    )
    return {
        "mean_direction": mean.astype(float).tolist(),
        "angular_error_deg": _scalar_statistics(angular_error),
    }


def _pose_statistics(positions, quaternions) -> dict:
    position = np.asarray(positions, dtype=np.float64)
    quaternion = np.asarray(quaternions, dtype=np.float64)
    reference = quaternion[0]
    quaternion = np.where(
        (quaternion @ reference)[:, None] < 0.0,
        -quaternion,
        quaternion,
    )
    mean_quaternion = np.mean(quaternion, axis=0)
    mean_quaternion /= max(float(np.linalg.norm(mean_quaternion)), 1.0e-12)
    orientation_error = 2.0 * np.arccos(
        np.clip(np.abs(quaternion @ mean_quaternion), 0.0, 1.0)
    )
    return {
        "position_mean_m": np.mean(position, axis=0).astype(float).tolist(),
        "position_std_m": np.std(position, axis=0).astype(float).tolist(),
        "position_peak_to_peak_m": np.ptp(position, axis=0).astype(float).tolist(),
        "mean_quaternion_xyzw": mean_quaternion.astype(float).tolist(),
        "orientation_error_deg": _scalar_statistics(
            np.degrees(orientation_error)
        ),
    }


def rotation_matrix_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def make_pose_transform(
    translation,
    *,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_xyz(roll, pitch, yaw)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def perturb_transform(
    transform,
    *,
    translation_xyz=(0.0, 0.0, 0.0),
    rotation_rpy=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply a slot-frame pose perturbation to a book transform."""

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")
    result = np.array(transform, copy=True)
    result[:3, :3] = (
        rotation_matrix_xyz(
            float(rotation_rpy[0]),
            float(rotation_rpy[1]),
            float(rotation_rpy[2]),
        )
        @ transform[:3, :3]
    )
    result[:3, 3] = transform[:3, 3] + np.asarray(
        translation_xyz,
        dtype=np.float64,
    )
    return result


def evaluate_shadow_case(
    bundle: NumpyActorBundle,
    transform_slot_book,
    transform_slot_tool,
    *,
    book_size=(0.156, 0.034, 0.236),
    slot_depth=0.20,
    motion_config=ResidualMotionConfig(),
    nominal_config=NominalInsertConfig(),
) -> ShadowCaseResult:
    raw, observation = compute_policy_observation(
        transform_slot_book,
        transform_slot_tool,
        book_size=book_size,
        slot_depth=slot_depth,
        mode_observation=0.0,
        gripper_open=0.0,
    )
    normalized, actor_mean, policy_action = bundle.predict(observation)
    nominal_delta = compute_insert_nominal_delta(raw, nominal_config)
    residual_delta = scale_residual_action(policy_action, motion_config)
    final_delta = combine_motion_delta(nominal_delta, residual_delta, motion_config)
    return ShadowCaseResult(
        raw_metrics=raw.astype(float).tolist(),
        observation=observation.astype(float).tolist(),
        normalized_observation=normalized.astype(float).tolist(),
        actor_mean=actor_mean.astype(float).tolist(),
        policy_action=policy_action.astype(float).tolist(),
        nominal_delta=nominal_delta.astype(float).tolist(),
        residual_delta=residual_delta.astype(float).tolist(),
        final_delta=final_delta.astype(float).tolist(),
        release_requested=bool(policy_action[-1] > motion_config.release_threshold),
    )


def case_as_dict(result: ShadowCaseResult) -> dict:
    return asdict(result)


def audit_shadow_source_tree(source_root) -> list[dict]:
    """Return source-level findings that could expose a robot-command path."""

    source_root = Path(source_root).resolve()
    forbidden_import_prefixes = (
        "control_msgs",
        "moveit",
        "moveit_msgs",
        "trajectory_msgs",
        "xarm_msgs",
    )
    forbidden_call_names = {
        "ActionClient",
        "create_client",
        "send_goal",
        "send_goal_async",
        "call_async",
    }
    forbidden_topic_fragments = (
        "/bookshelf_policy/action",
        "/follow_joint_trajectory",
        "/joint_trajectory",
        "/xarm",
    )
    forbidden_process_fragments = (
        "action_executor",
        "cartesian_action_executor",
        "move_group",
        "policy_to_robot",
        "xarm_planner",
    )
    findings = []

    for path in sorted(source_root.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            findings.append({"path": str(path), "line": 0, "reason": f"parse error: {error}"})
            continue

        # The dedicated logger is allowed to name command topics only inside
        # its rosbag recording list. Merely subscribing/recording those exact
        # strings is important evidence and does not expose a command path.
        record_only_topic_nodes = set()
        if path.name == "experiment_logging.launch.py":
            for statement in tree.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "CORE_TOPICS"
                    for target in statement.targets
                ):
                    continue
                if isinstance(statement.value, (ast.List, ast.Tuple)):
                    record_only_topic_nodes.update(
                        id(element)
                        for element in statement.value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    )

        # The physical preflight names prohibited processes only so it can
        # prove they are absent from the live ROS graph. Those literals are
        # audit inputs, not subprocess or launch targets.
        prohibited_process_name_nodes = set()
        if path.name == "physical_experiment_preflight.py":
            for statement in tree.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name)
                    and target.id in {
                        "CRITICAL_NODE_NAMES",
                        "PROHIBITED_EXECUTION_NODES",
                        "REQUIRED_HARDWARE_NODES",
                    }
                    for target in statement.targets
                ):
                    continue
                if isinstance(statement.value, (ast.Set, ast.List, ast.Tuple)):
                    prohibited_process_name_nodes.update(
                        id(element)
                        for element in statement.value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                if name.startswith(forbidden_import_prefixes):
                    findings.append(
                        {
                            "path": str(path),
                            "line": int(getattr(node, "lineno", 0)),
                            "reason": f"forbidden robot-control import: {name}",
                        }
                    )

            if isinstance(node, ast.Call):
                function = node.func
                call_name = None
                if isinstance(function, ast.Name):
                    call_name = function.id
                elif isinstance(function, ast.Attribute):
                    call_name = function.attr
                if call_name in forbidden_call_names:
                    findings.append(
                        {
                            "path": str(path),
                            "line": int(getattr(node, "lineno", 0)),
                            "reason": f"forbidden robot-control call: {call_name}",
                        }
                    )

            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in record_only_topic_nodes:
                    for fragment in forbidden_topic_fragments:
                        if fragment in node.value:
                            findings.append(
                                {
                                    "path": str(path),
                                    "line": int(getattr(node, "lineno", 0)),
                                    "reason": f"forbidden command namespace: {fragment}",
                                }
                            )
                for fragment in forbidden_process_fragments:
                    if (
                        id(node) not in prohibited_process_name_nodes
                        and fragment in node.value
                    ):
                        findings.append(
                            {
                                "path": str(path),
                                "line": int(getattr(node, "lineno", 0)),
                                "reason": f"forbidden robot-control process: {fragment}",
                            }
                        )
    return findings


def extract_class_numeric_assignments(path, class_name="BookshelfEnvCfg") -> dict[str, float]:
    """Read simple numeric config assignments without importing Isaac Lab."""

    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}

    def numeric_value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
            and node.func.attr == "radians"
            and len(node.args) == 1
        ):
            degrees = numeric_value(node.args[0])
            return math.radians(degrees) if degrees is not None else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = numeric_value(node.operand)
            return -value if value is not None else None
        return None

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = numeric_value(statement.value)
            if value is not None:
                values[target.id] = value
    return values


def controller_config_parity(env_cfg_path) -> dict:
    """Compare portable shadow constants with the simulator config source."""

    source = extract_class_numeric_assignments(env_cfg_path)
    motion = ResidualMotionConfig()
    nominal = NominalInsertConfig()
    expected = {
        "dx_action_scale": motion.action_scales[0],
        "dy_action_scale": motion.action_scales[1],
        "dz_action_scale": motion.action_scales[2],
        "dyaw_action_scale": motion.action_scales[3],
        "dpitch_action_scale": motion.action_scales[4],
        "nominal_insert_dx": nominal.insert_dx,
        "nominal_insert_dx_near_mouth": nominal.insert_dx_near_mouth,
        "nominal_lateral_gain": nominal.lateral_gain,
        "nominal_height_gain": nominal.height_gain,
        "nominal_insert_z_offset": nominal.insert_z_offset,
        "nominal_yaw_gain": nominal.yaw_gain,
        "nominal_pitch_gain": nominal.pitch_gain,
        "nominal_align_lat_thresh": nominal.align_lat_thresh,
        "nominal_align_z_thresh": nominal.align_z_thresh,
        "nominal_align_yaw_thresh": nominal.align_yaw_thresh,
        "nominal_align_tilt_x_thresh": nominal.align_tilt_x_thresh,
        "nominal_unaligned_dx_scale": nominal.unaligned_dx_scale,
        "nominal_dy_limit": nominal.dy_limit,
        "nominal_dz_limit": nominal.dz_limit,
        "nominal_dyaw_limit": nominal.dyaw_limit,
        "nominal_dpitch_limit": nominal.dpitch_limit,
        "nominal_slow_rear_to_mouth": nominal.slow_rear_to_mouth,
        "final_dx_limit": motion.final_limits[0],
        "final_dy_limit": motion.final_limits[1],
        "final_dz_limit": motion.final_limits[2],
        "final_dyaw_limit": motion.final_limits[3],
        "final_dpitch_limit": motion.final_limits[4],
    }
    mismatches = []
    for name, portable_value in expected.items():
        simulator_value = source.get(name)
        if simulator_value is None:
            mismatches.append({"name": name, "reason": "missing from simulator config"})
        elif not math.isclose(simulator_value, portable_value, rel_tol=0.0, abs_tol=1.0e-12):
            mismatches.append(
                {
                    "name": name,
                    "simulator": simulator_value,
                    "portable": portable_value,
                }
            )
    return {
        "passed": not mismatches,
        "checked_values": len(expected),
        "mismatches": mismatches,
    }
