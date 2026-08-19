#!/usr/bin/env python3
"""Run a fail-closed stationary A/B/C observation and policy-shadow replay."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml

from .policy_observation_math import OBSERVATION_LABELS
from .stationary_capture_bundle import REQUIRED_CAPTURE_TOPICS, sha256_file
from .stationary_capture_pipeline import (
    _assert_ros_graph_is_safe,
    process_stationary_captures,
)


def _load_json(path) -> dict:
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path) -> dict:
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _adapter_parameters(document: dict) -> dict:
    node = document.get("policy_observation_adapter")
    if not isinstance(node, dict):
        raise ValueError("candidate is missing policy_observation_adapter")
    parameters = node.get("ros__parameters")
    if not isinstance(parameters, dict):
        raise ValueError("candidate adapter is missing ros__parameters")
    return dict(parameters)


def build_shadow_adapter_configuration(bundle: dict, candidate: dict) -> dict:
    """Build an offline-only adapter config from an unapproved A/B/C bundle."""

    if bundle.get("kind") != "bookshelf_stationary_capture_calibration_bundle_candidate":
        raise ValueError("unexpected stationary calibration bundle kind")
    if bundle.get("candidate_valid") is not True:
        raise ValueError("stationary calibration candidate is not valid")
    if bundle.get("candidate_selected") is not False:
        raise ValueError("stationary calibration candidate must remain unselected")
    safety = bundle.get("safety", {})
    if safety.get("hardware_commanded") is not False:
        raise ValueError("candidate does not prove hardware_commanded=false")
    if safety.get("execution_authorized") is not False:
        raise ValueError("candidate unexpectedly authorizes execution")

    slot = bundle.get("slot")
    if not isinstance(slot, dict):
        raise ValueError("stationary calibration bundle is missing the frozen slot")
    adapter = _adapter_parameters(candidate)
    candidate_book_source = adapter.get("book_pose_source")
    if candidate_book_source == "eef_fixed":
        book_calibration = bundle.get("book_calibration")
        if not isinstance(book_calibration, dict):
            raise ValueError("bundle is missing the measured book calibration")
        measured = book_calibration.get("transform_eef_book")
        if not isinstance(measured, dict):
            raise ValueError("bundle is missing transform_eef_book")
        candidate_translation = np.asarray(
            adapter.get("eef_book_translation_xyz"), dtype=np.float64
        )
        candidate_quaternion = np.asarray(
            adapter.get("eef_book_quaternion_xyzw"), dtype=np.float64
        )
        measured_translation = np.asarray(
            measured.get("translation_xyz_m", measured.get("translation_xyz")),
            dtype=np.float64,
        )
        measured_quaternion = np.asarray(
            measured.get("quaternion_xyzw"), dtype=np.float64
        )
        if (
            candidate_translation.shape != (3,)
            or candidate_quaternion.shape != (4,)
            or measured_translation.shape != (3,)
            or measured_quaternion.shape != (4,)
            or not np.allclose(
                candidate_translation, measured_translation, atol=1.0e-9
            )
            or not (
                np.allclose(
                    candidate_quaternion, measured_quaternion, atol=1.0e-9
                )
                or np.allclose(
                    candidate_quaternion, -measured_quaternion, atol=1.0e-9
                )
            )
        ):
            raise ValueError(
                "legacy fixed-book candidate does not match the validated "
                "bundle book transform"
            )
    elif candidate_book_source != "marker":
        raise ValueError(
            "candidate book source must be marker or a matching legacy eef_fixed"
        )
    if bool(adapter.get("latch_eef_book_from_marker", False)):
        raise ValueError("candidate must not latch marker data into a fixed book pose")

    adapter.update(
        {
            "target_book_frame": "target_book_center",
            "book_pose_source": "marker",
            "latch_eef_book_from_marker": False,
            "slot_pose_source": "configured_static",
            "allow_configured_static_slot": True,
            "configured_static_slot_translation_xyz": list(
                slot["translation_xyz"]
            ),
            "configured_static_slot_quaternion_xyzw": list(
                slot["quaternion_xyzw"]
            ),
            "configured_static_slot_width_m": float(slot["width_m"]),
            "configured_static_slot_confidence": float(slot["confidence"]),
            "static_slot_transform_status": str(slot["transform_status"]),
            # This bypass is confined to a launch with no command interfaces.
            # It lets us inspect the derived candidate before promotion.
            "require_verified_policy_tool_transform": False,
        }
    )
    return {
        "policy_observation_adapter": {
            "ros__parameters": adapter,
        },
        "stationary_shadow_replay_provenance": {
            "ros__parameters": {
                "source_candidate_book_pose_source": str(
                    candidate_book_source
                ),
                "runtime_book_pose_source": "marker",
                "legacy_candidate_transform_matched_bundle": bool(
                    candidate_book_source == "eef_fixed"
                ),
            }
        },
    }


def load_stationary_calibration(calibration_dir) -> tuple[dict, dict, dict]:
    """Load a completed, unapproved calibration without replaying A/B/C."""

    calibration_dir = Path(calibration_dir).expanduser().resolve()
    bundle_path = calibration_dir / "stationary_calibration_bundle_candidate.json"
    candidate_path = calibration_dir / "stationary_calibration_candidate.yaml"
    if not bundle_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError(
            "calibration directory must contain the stationary bundle JSON "
            "and candidate YAML"
        )
    bundle = _load_json(bundle_path)
    candidate = _load_yaml(candidate_path)
    expected_candidate_hash = (
        bundle.get("output_hashes", {})
        .get("unapproved_parameter_candidate_sha256")
    )
    actual_candidate_hash = sha256_file(candidate_path)
    if (
        expected_candidate_hash
        and actual_candidate_hash != expected_candidate_hash
    ):
        raise ValueError("stationary candidate YAML hash does not match its bundle")
    build_shadow_adapter_configuration(bundle, candidate)
    return bundle, candidate, {
        "mode": "existing_validated_candidate",
        "directory": str(calibration_dir),
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": actual_candidate_hash,
    }


def shadow_bag_play_command(bag_directory, *, rate: float) -> list[str]:
    if float(rate) <= 0.0:
        raise ValueError("bag replay rate must be positive")
    return [
        "ros2",
        "bag",
        "play",
        str(Path(bag_directory)),
        "--clock",
        "30",
        "--disable-keyboard-controls",
        "--rate",
        str(float(rate)),
        "--topics",
        *REQUIRED_CAPTURE_TOPICS,
    ]


def shadow_launch_command(
    *,
    adapter_config,
    mount_yaml,
    output_dir,
    policy_bundle,
    activation_envelope,
    candidate_id,
    minimum_valid_samples: int,
    enable_rviz: bool,
) -> list[str]:
    return [
        "ros2",
        "launch",
        "bookshelf_shadow_ros",
        "stationary_shadow_replay.launch.py",
        f"adapter_config:={Path(adapter_config)}",
        f"mount_yaml:={Path(mount_yaml)}",
        f"output_dir:={Path(output_dir)}",
        f"policy_bundle:={Path(policy_bundle)}",
        f"activation_envelope:={Path(activation_envelope)}",
        f"candidate_id:={candidate_id}",
        f"minimum_valid_samples:={int(minimum_valid_samples)}",
        f"enable_rviz:={str(bool(enable_rviz)).lower()}",
        "use_sim_time:=true",
    ]


def _rotation_error_deg(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _pose_stability(positions, quaternions) -> dict:
    if not positions:
        return {
            "samples": 0,
            "translation_spread_m": None,
            "rotation_spread_deg": None,
            "maximum_consecutive_translation_jump_m": None,
            "maximum_consecutive_rotation_jump_deg": None,
        }
    positions = np.asarray(positions, dtype=np.float64)
    centre = np.median(positions, axis=0)
    translation_spread = np.linalg.norm(positions - centre, axis=1)
    rotation_spread = [
        _rotation_error_deg(quaternions[0], value) for value in quaternions
    ]
    consecutive_translation = [
        float(np.linalg.norm(positions[index] - positions[index - 1]))
        for index in range(1, len(positions))
    ]
    consecutive_rotation = [
        _rotation_error_deg(quaternions[index - 1], quaternions[index])
        for index in range(1, len(quaternions))
    ]
    return {
        "samples": int(len(positions)),
        "median_translation_xyz_m": centre.tolist(),
        "translation_spread_m": float(np.max(translation_spread)),
        "rotation_spread_deg": float(max(rotation_spread, default=0.0)),
        "maximum_consecutive_translation_jump_m": float(
            max(consecutive_translation, default=0.0)
        ),
        "maximum_consecutive_rotation_jump_deg": float(
            max(consecutive_rotation, default=0.0)
        ),
    }


def _vector_statistics(values, labels) -> dict:
    if not values:
        return {}
    matrix = np.asarray(values, dtype=np.float64)
    return {
        label: {
            "min": float(np.min(matrix[:, index])),
            "max": float(np.max(matrix[:, index])),
            "mean": float(np.mean(matrix[:, index])),
            "std": float(np.std(matrix[:, index])),
        }
        for index, label in enumerate(labels)
    }


class StationaryShadowReplayAccumulator:
    """Accumulate adapter-first replay evidence without requiring activation."""

    def __init__(
        self,
        *,
        minimum_valid_samples=30,
        maximum_book_translation_jump_m=0.010,
        maximum_book_rotation_jump_deg=5.0,
    ):
        self.minimum_valid_samples = max(int(minimum_valid_samples), 1)
        self.maximum_book_translation_jump_m = float(
            maximum_book_translation_jump_m
        )
        self.maximum_book_rotation_jump_deg = float(
            maximum_book_rotation_jump_deg
        )
        self.adapter_messages = 0
        self.valid_samples = 0
        self.invalid_reasons = Counter()
        self.book_sources = Counter()
        self.slot_sources = Counter()
        self.observations = []
        self.raw_metrics = []
        self.book_positions = []
        self.book_quaternions = []
        self.slot_positions = []
        self.slot_quaternions = []
        self.rows = []
        self.policy_messages = 0
        self.policy_valid_messages = 0
        self.policy_blocked_reasons = Counter()
        self.normalized_observations = []
        self.activation_ready_messages = 0
        self.recent_adapter_observations = []
        self.policy_observation_mismatches = 0

    def add_invalid(self, reason) -> None:
        self.adapter_messages += 1
        self.invalid_reasons[str(reason or "unspecified")] += 1

    def add_adapter_sample(
        self,
        payload,
        *,
        book_position,
        book_quaternion,
        slot_position,
        slot_quaternion,
    ) -> bool:
        self.adapter_messages += 1
        if not bool(payload.get("valid", False)):
            self.invalid_reasons[str(payload.get("reason", "unspecified"))] += 1
            return False
        try:
            observation = np.asarray(payload["observation_12d"], dtype=np.float64)
            raw_mapping = payload["raw_metrics"]
            raw = np.asarray(
                [raw_mapping[label] for label in OBSERVATION_LABELS],
                dtype=np.float64,
            )
            if observation.shape != (len(OBSERVATION_LABELS),):
                raise ValueError("observation_12d has the wrong shape")
            if raw.shape != (len(OBSERVATION_LABELS),):
                raise ValueError("raw_metrics has the wrong shape")
            if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(raw)):
                raise ValueError("observation contains non-finite values")
            book_position = np.asarray(book_position, dtype=np.float64)
            book_quaternion = np.asarray(book_quaternion, dtype=np.float64)
            slot_position = np.asarray(slot_position, dtype=np.float64)
            slot_quaternion = np.asarray(slot_quaternion, dtype=np.float64)
            if book_position.shape != (3,) or slot_position.shape != (3,):
                raise ValueError("pose translation has the wrong shape")
            if book_quaternion.shape != (4,) or slot_quaternion.shape != (4,):
                raise ValueError("pose quaternion has the wrong shape")
            if not all(
                np.all(np.isfinite(value))
                for value in (
                    book_position,
                    book_quaternion,
                    slot_position,
                    slot_quaternion,
                )
            ):
                raise ValueError("pose contains non-finite values")
            if min(
                np.linalg.norm(book_quaternion),
                np.linalg.norm(slot_quaternion),
            ) <= 1.0e-9:
                raise ValueError("pose quaternion has zero norm")
        except (KeyError, TypeError, ValueError) as error:
            self.invalid_reasons[f"malformed valid sample: {error}"] += 1
            return False

        book_source = str(payload.get("book_pose_source", "unknown"))
        slot_source = str(payload.get("slot_pose_source", "unknown"))
        self.book_sources[book_source] += 1
        self.slot_sources[slot_source] += 1
        self.observations.append(observation)
        self.recent_adapter_observations.append(observation.copy())
        self.recent_adapter_observations = self.recent_adapter_observations[-10:]
        self.raw_metrics.append(raw)
        self.book_positions.append(book_position)
        self.book_quaternions.append(book_quaternion)
        self.slot_positions.append(slot_position)
        self.slot_quaternions.append(slot_quaternion)
        self.valid_samples += 1
        row = {
            "sample": self.valid_samples,
            "book_pose_source": book_source,
            "slot_pose_source": slot_source,
        }
        row.update(
            {
                f"observation_{label}": float(value)
                for label, value in zip(OBSERVATION_LABELS, observation)
            }
        )
        row.update(
            {
                f"raw_{label}": float(value)
                for label, value in zip(OBSERVATION_LABELS, raw)
            }
        )
        self.rows.append(row)
        return True

    def add_policy_debug(self, payload) -> None:
        self.policy_messages += 1
        if bool(payload.get("valid", False)):
            self.policy_valid_messages += 1
        else:
            self.policy_blocked_reasons[
                str(payload.get("reason", "unspecified"))
            ] += 1
        normalized = payload.get("normalized_observation")
        if normalized is not None:
            vector = np.asarray(normalized, dtype=np.float64)
            if vector.shape == (len(OBSERVATION_LABELS),) and np.all(
                np.isfinite(vector)
            ):
                self.normalized_observations.append(vector)
        policy_observation = payload.get("observation_12d")
        if policy_observation is not None and self.recent_adapter_observations:
            vector = np.asarray(policy_observation, dtype=np.float64)
            matched = vector.shape == (len(OBSERVATION_LABELS),) and any(
                np.allclose(vector, candidate, rtol=0.0, atol=1.0e-6)
                for candidate in self.recent_adapter_observations
            )
            if not matched:
                self.policy_observation_mismatches += 1
        activation = payload.get("policy_activation")
        if isinstance(activation, dict) and bool(activation.get("ready", False)):
            self.activation_ready_messages += 1

    def summary(self) -> dict:
        book_stability = _pose_stability(
            self.book_positions, self.book_quaternions
        )
        slot_stability = _pose_stability(
            self.slot_positions, self.slot_quaternions
        )
        clipped_counts = Counter()
        samples_with_clips = 0
        for observation in self.observations:
            clipped = [
                label
                for label, value in zip(OBSERVATION_LABELS, observation)
                if abs(float(value)) >= 1.0 - 1.0e-6
            ]
            if clipped:
                samples_with_clips += 1
                clipped_counts.update(clipped)

        failures = []
        if self.valid_samples < self.minimum_valid_samples:
            failures.append(
                "insufficient valid marker observations: "
                f"{self.valid_samples}/{self.minimum_valid_samples}"
            )
        if set(self.book_sources) - {"marker"}:
            failures.append("a non-marker book pose source was used")
        if self.valid_samples and self.book_sources.get("marker", 0) != self.valid_samples:
            failures.append("marker book source did not cover every valid sample")
        if set(self.slot_sources) - {"configured_static"}:
            failures.append("slot pose was not frozen for every valid sample")
        if self.valid_samples and self.slot_sources.get(
            "configured_static", 0
        ) != self.valid_samples:
            failures.append("configured static slot did not cover every valid sample")
        if (
            book_stability["maximum_consecutive_translation_jump_m"] is not None
            and book_stability["maximum_consecutive_translation_jump_m"]
            > self.maximum_book_translation_jump_m
        ):
            failures.append("live book translation jumped beyond its limit")
        if (
            book_stability["maximum_consecutive_rotation_jump_deg"] is not None
            and book_stability["maximum_consecutive_rotation_jump_deg"]
            > self.maximum_book_rotation_jump_deg
        ):
            failures.append("live book rotation jumped beyond its limit")
        if (
            slot_stability["translation_spread_m"] is not None
            and slot_stability["translation_spread_m"] > 1.0e-9
        ):
            failures.append("frozen slot translation changed during replay")
        if (
            slot_stability["rotation_spread_deg"] is not None
            and slot_stability["rotation_spread_deg"] > 1.0e-6
        ):
            failures.append("frozen slot rotation changed during replay")
        if self.policy_messages == 0:
            failures.append("no policy-shadow diagnostics were received")
        if self.valid_samples and not self.normalized_observations:
            failures.append("policy shadow never normalized a valid observation")
        if self.policy_observation_mismatches:
            failures.append("adapter and policy observations did not match")

        return {
            "passed": not failures,
            "failure_reasons": failures,
            "adapter_messages": self.adapter_messages,
            "valid_samples": self.valid_samples,
            "minimum_valid_samples": self.minimum_valid_samples,
            "invalid_samples": self.adapter_messages - self.valid_samples,
            "invalid_reasons": dict(self.invalid_reasons.most_common()),
            "book_pose_sources": dict(self.book_sources),
            "slot_pose_sources": dict(self.slot_sources),
            "book_pose_stability": book_stability,
            "frozen_slot_stability": slot_stability,
            "observation_12d": _vector_statistics(
                self.observations, OBSERVATION_LABELS
            ),
            "raw_metrics": _vector_statistics(self.raw_metrics, OBSERVATION_LABELS),
            "samples_with_clipped_observations": samples_with_clips,
            "observation_clip_fraction": (
                float(samples_with_clips / self.valid_samples)
                if self.valid_samples
                else 0.0
            ),
            "clipped_observation_counts": dict(clipped_counts),
            "policy_diagnostics": {
                "messages": self.policy_messages,
                "valid_inference_messages": self.policy_valid_messages,
                "blocked_messages": self.policy_messages
                - self.policy_valid_messages,
                "blocked_reasons": dict(
                    self.policy_blocked_reasons.most_common()
                ),
                "activation_ready_messages": self.activation_ready_messages,
                "adapter_policy_observation_mismatches": (
                    self.policy_observation_mismatches
                ),
                "normalized_observation": _vector_statistics(
                    self.normalized_observations, OBSERVATION_LABELS
                ),
                "activation_is_required_for_replay_pass": False,
            },
        }


def _stop_process_group(process) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2.0)


def _run_shadow_stage(
    *,
    launch_command,
    bag_directory,
    bag_duration_s,
    replay_rate,
    output_dir,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(output_dir / "ros_logs")
    Path(environment["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    launch_log_path = output_dir / "launch.log"
    playback_log_path = output_dir / "bag_play.log"
    launch_process = None
    playback_process = None
    try:
        with launch_log_path.open("w", encoding="utf-8") as launch_log:
            launch_process = subprocess.Popen(
                launch_command,
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
            )
        time.sleep(2.0)
        if launch_process.poll() is not None:
            raise RuntimeError(
                f"shadow launch exited before replay; inspect {launch_log_path}"
            )
        play_command = shadow_bag_play_command(
            bag_directory, rate=replay_rate
        )
        with playback_log_path.open("w", encoding="utf-8") as playback_log:
            playback_process = subprocess.Popen(
                play_command,
                stdout=playback_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
            )
            timeout_s = max(
                float(bag_duration_s) / float(replay_rate) + 45.0,
                60.0,
            )
            try:
                returncode = playback_process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"book bag replay exceeded {timeout_s:.1f}s"
                ) from error
        if returncode != 0:
            raise RuntimeError(
                f"book bag replay failed; inspect {playback_log_path}"
            )
        time.sleep(3.0)
    finally:
        _stop_process_group(playback_process)
        _stop_process_group(launch_process)
    return {
        "launch_command": list(launch_command),
        "bag_play_command": shadow_bag_play_command(
            bag_directory, rate=replay_rate
        ),
        "launch_log": str(launch_log_path),
        "bag_play_log": str(playback_log_path),
    }


def run_stationary_shadow_replay(
    *,
    view_a_run,
    view_b_run,
    book_run,
    output_dir,
    repository_path,
    policy_bundle,
    activation_envelope,
    mount_yaml,
    calibration_dir=None,
    slot_target_samples=90,
    book_target_samples=60,
    replay_rate=1.0,
    minimum_valid_samples=30,
    view_a_roi_x_min=0.12,
    view_a_roi_x_max=0.88,
    view_b_roi_x_min=0.23,
    view_b_roi_x_max=0.48,
    view_a_minimum_slot_width_m=0.032,
    view_a_maximum_slot_width_m=0.045,
    view_b_minimum_slot_width_m=0.032,
    view_b_maximum_slot_width_m=0.045,
    maximum_rotation_disagreement_deg=5.0,
    maximum_rotation_sanity_disagreement_deg=15.0,
    enable_rviz=False,
    hash_bag_files=True,
) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    policy_bundle = Path(policy_bundle).expanduser().resolve()
    activation_envelope = Path(activation_envelope).expanduser().resolve()
    mount_yaml = Path(mount_yaml).expanduser().resolve()
    for path, label in (
        (policy_bundle, "policy bundle"),
        (activation_envelope, "activation envelope"),
        (mount_yaml, "marker mount YAML"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    _assert_ros_graph_is_safe()
    if calibration_dir is None:
        if not view_a_run or not view_b_run or not book_run:
            raise ValueError(
                "view A, view B, and book runs are required when no "
                "calibration directory is supplied"
            )
        generated_calibration_dir = output_dir / "calibration"
        bundle = process_stationary_captures(
            view_a_run=view_a_run,
            view_b_run=view_b_run,
            book_run=book_run,
            output_dir=generated_calibration_dir,
            repository_path=repository_path,
            mount_yaml=mount_yaml,
            slot_target_samples=slot_target_samples,
            book_target_samples=book_target_samples,
            replay_rate=replay_rate,
            view_a_roi_x_min=view_a_roi_x_min,
            view_a_roi_x_max=view_a_roi_x_max,
            view_b_roi_x_min=view_b_roi_x_min,
            view_b_roi_x_max=view_b_roi_x_max,
            view_a_minimum_slot_width_m=view_a_minimum_slot_width_m,
            view_a_maximum_slot_width_m=view_a_maximum_slot_width_m,
            view_b_minimum_slot_width_m=view_b_minimum_slot_width_m,
            view_b_maximum_slot_width_m=view_b_maximum_slot_width_m,
            maximum_rotation_disagreement_deg=(
                maximum_rotation_disagreement_deg
            ),
            maximum_rotation_sanity_disagreement_deg=(
                maximum_rotation_sanity_disagreement_deg
            ),
            hash_bag_files=hash_bag_files,
        )
        candidate_path = (
            generated_calibration_dir / "stationary_calibration_candidate.yaml"
        )
        candidate = _load_yaml(candidate_path)
        calibration_source = {
            "mode": "regenerated_from_bags",
            "directory": str(generated_calibration_dir),
            "bundle_path": str(
                generated_calibration_dir
                / "stationary_calibration_bundle_candidate.json"
            ),
            "candidate_path": str(candidate_path),
        }
    else:
        bundle, candidate, calibration_source = load_stationary_calibration(
            calibration_dir
        )
    adapter_configuration = build_shadow_adapter_configuration(bundle, candidate)
    shadow_dir = output_dir / "shadow_replay"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = shadow_dir / "stationary_shadow_adapter.yaml"
    adapter_path.write_text(
        "# OFFLINE SHADOW ONLY. This file cannot authorize execution.\n"
        + yaml.safe_dump(adapter_configuration, sort_keys=False),
        encoding="utf-8",
    )

    book_capture = bundle["source_captures"]["book_attached"]
    launch = shadow_launch_command(
        adapter_config=adapter_path,
        mount_yaml=mount_yaml,
        output_dir=shadow_dir,
        policy_bundle=policy_bundle,
        activation_envelope=activation_envelope,
        candidate_id=bundle["candidate_id"],
        minimum_valid_samples=minimum_valid_samples,
        enable_rviz=enable_rviz,
    )
    _assert_ros_graph_is_safe()
    stage = _run_shadow_stage(
        launch_command=launch,
        bag_directory=book_capture["bag_directory"],
        bag_duration_s=book_capture["duration_s"],
        replay_rate=replay_rate,
        output_dir=shadow_dir,
    )
    audit_path = shadow_dir / "stationary_shadow_replay_report.json"
    marker_report_path = shadow_dir / "marker" / "marker_book_calibration_summary.json"
    if not audit_path.is_file():
        raise RuntimeError(
            f"shadow audit report was not written; inspect {stage['launch_log']}"
        )
    if not marker_report_path.is_file():
        raise RuntimeError(
            f"marker report was not written; inspect {stage['launch_log']}"
        )
    audit = _load_json(audit_path)
    marker_report = _load_json(marker_report_path)
    result = {
        "schema_version": 1,
        "kind": "bookshelf_stationary_shadow_replay_pipeline",
        "generated_at": datetime.now().astimezone().isoformat(),
        "passed": bool(
            audit.get("passed", False)
            and marker_report.get("calibration_valid", False)
        ),
        "candidate_id": bundle["candidate_id"],
        "calibration_source": calibration_source,
        "calibration_bundle": {
            "path": calibration_source["bundle_path"],
            "sha256": sha256_file(calibration_source["bundle_path"]),
        },
        "runtime_adapter": {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
            "book_pose_source": "marker",
            "slot_pose_source": "configured_static",
            "policy_tool_candidate_used_for_diagnostics_only": True,
        },
        "policy_bundle": {
            "path": str(policy_bundle),
            "sha256": sha256_file(policy_bundle),
        },
        "activation_envelope": {
            "path": str(activation_envelope),
            "sha256": sha256_file(activation_envelope),
        },
        "shadow_stage": stage,
        "observation_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "passed": bool(audit.get("passed", False)),
        },
        "marker_detection": {
            "path": str(marker_report_path),
            "sha256": sha256_file(marker_report_path),
            "calibration_valid": bool(
                marker_report.get("calibration_valid", False)
            ),
        },
        "safety": {
            "shadow_only": True,
            "plan_requested": False,
            "execution_authorized": False,
            "hardware_commanded": False,
            "active_configuration_modified": False,
            "candidate_selected": False,
        },
    }
    report_path = output_dir / "stationary_shadow_replay_pipeline_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise RuntimeError(
            f"offline shadow replay did not pass; inspect {audit_path}"
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Use a completed stationary calibration, or regenerate A/B/C, then "
            "replay Bag C through the frozen-slot, live-marker 12D observation "
            "and policy-shadow path."
        )
    )
    parser.add_argument("--view-a-run")
    parser.add_argument("--view-b-run")
    parser.add_argument("--book-run")
    parser.add_argument(
        "--calibration-dir",
        help=(
            "Completed stationary_capture_pipeline output. When supplied, "
            "A/B calibration is not replayed and Bag C comes from its bundle."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--policy-bundle", required=True)
    parser.add_argument("--activation-envelope", required=True)
    parser.add_argument("--mount-yaml", required=True)
    parser.add_argument("--slot-target-samples", type=int, default=90)
    parser.add_argument("--book-target-samples", type=int, default=60)
    parser.add_argument("--minimum-valid-samples", type=int, default=30)
    parser.add_argument("--replay-rate", type=float, default=1.0)
    parser.add_argument("--view-a-roi-x-min", type=float, default=0.12)
    parser.add_argument("--view-a-roi-x-max", type=float, default=0.88)
    parser.add_argument("--view-b-roi-x-min", type=float, default=0.23)
    parser.add_argument("--view-b-roi-x-max", type=float, default=0.48)
    parser.add_argument(
        "--view-a-minimum-slot-width-m", type=float, default=0.032
    )
    parser.add_argument(
        "--view-a-maximum-slot-width-m", type=float, default=0.045
    )
    parser.add_argument(
        "--view-b-minimum-slot-width-m", type=float, default=0.032
    )
    parser.add_argument(
        "--view-b-maximum-slot-width-m", type=float, default=0.045
    )
    parser.add_argument(
        "--maximum-rotation-disagreement-deg", type=float, default=5.0
    )
    parser.add_argument(
        "--maximum-rotation-sanity-disagreement-deg",
        type=float,
        default=15.0,
    )
    parser.add_argument("--enable-rviz", action="store_true")
    parser.add_argument("--skip-bag-hashes", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        result = run_stationary_shadow_replay(
            view_a_run=args.view_a_run,
            view_b_run=args.view_b_run,
            book_run=args.book_run,
            output_dir=output_dir,
            repository_path=args.repository,
            policy_bundle=args.policy_bundle,
            activation_envelope=args.activation_envelope,
            mount_yaml=args.mount_yaml,
            calibration_dir=args.calibration_dir,
            slot_target_samples=args.slot_target_samples,
            book_target_samples=args.book_target_samples,
            minimum_valid_samples=args.minimum_valid_samples,
            replay_rate=args.replay_rate,
            view_a_roi_x_min=args.view_a_roi_x_min,
            view_a_roi_x_max=args.view_a_roi_x_max,
            view_b_roi_x_min=args.view_b_roi_x_min,
            view_b_roi_x_max=args.view_b_roi_x_max,
            view_a_minimum_slot_width_m=args.view_a_minimum_slot_width_m,
            view_a_maximum_slot_width_m=args.view_a_maximum_slot_width_m,
            view_b_minimum_slot_width_m=args.view_b_minimum_slot_width_m,
            view_b_maximum_slot_width_m=args.view_b_maximum_slot_width_m,
            maximum_rotation_disagreement_deg=(
                args.maximum_rotation_disagreement_deg
            ),
            maximum_rotation_sanity_disagreement_deg=(
                args.maximum_rotation_sanity_disagreement_deg
            ),
            enable_rviz=args.enable_rviz,
            hash_bag_files=not args.skip_bag_hashes,
        )
    except Exception as error:
        print(f"FAIL: {error}")
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "stationary_shadow_replay_failure.json"
        if not failure_path.exists():
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "bookshelf_stationary_shadow_replay_failure",
                        "generated_at": datetime.now().astimezone().isoformat(),
                        "reason": str(error),
                        "execution_authorized": False,
                        "hardware_commanded": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"Failure report: {failure_path}")
        return 1

    print("Offline observation replay: PASS")
    print(f"Candidate: {result['candidate_id']}")
    print(f"Results: {output_dir}")
    print("Execution authorized: False")
    print("Hardware commanded: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
