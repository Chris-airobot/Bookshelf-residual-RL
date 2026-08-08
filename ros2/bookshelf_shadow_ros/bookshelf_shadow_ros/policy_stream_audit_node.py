#!/usr/bin/env python3
"""Audit the complete read-only real-camera-to-policy shadow stream."""

import csv
from datetime import datetime
import json
from pathlib import Path
import time

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from .offline_validation import (
    PolicyActivationAuditAccumulator,
    PolicyStreamAuditAccumulator,
)
from .policy_observation_math import OBSERVATION_LABELS
from .policy_shadow_math import MOTION_LABELS, POLICY_ACTION_LABELS


class PolicyStreamAuditNode(Node):
    """Correlate perception, observation, and policy diagnostics without commands."""

    def __init__(self):
        super().__init__("policy_stream_audit")
        self._declare_parameters()
        self.accumulator = PolicyStreamAuditAccumulator(
            reference_slot_width_m=float(
                self.get_parameter("reference_slot_width_m").value
            )
        )
        self.activation_accumulator = PolicyActivationAuditAccumulator()
        self.latest = {}
        self.completed = False

        self._subscribe(
            Float32,
            "confidence_topic",
            "confidence",
            lambda message: float(message.data),
        )
        self._subscribe(
            Float32,
            "slot_width_topic",
            "slot_width",
            lambda message: float(message.data),
        )
        self._subscribe(
            PoseStamped,
            "slot_pose_base_topic",
            "slot_pose",
            lambda message: message,
        )
        self._subscribe(
            PoseStamped,
            "book_pose_base_topic",
            "book_pose",
            lambda message: message,
        )
        self._subscribe(
            Float32MultiArray,
            "raw_metrics_topic",
            "raw_metrics",
            lambda message: list(message.data),
        )
        self._subscribe(
            Float32MultiArray,
            "observation_topic",
            "observation",
            lambda message: list(message.data),
        )
        self._subscribe(
            Bool,
            "observation_valid_topic",
            "observation_valid",
            lambda message: bool(message.data),
        )
        self._subscribe(
            String,
            "adapter_debug_topic",
            "adapter_debug",
            self._json_payload,
        )
        self._subscribe(
            Bool,
            "inference_valid_topic",
            "inference_valid",
            lambda message: bool(message.data),
        )
        self._subscribe(
            Float32MultiArray,
            "policy_action_topic",
            "policy_action",
            lambda message: list(message.data),
        )
        self._subscribe(
            Float32MultiArray,
            "nominal_delta_topic",
            "nominal_delta",
            lambda message: list(message.data),
        )
        self._subscribe(
            Float32MultiArray,
            "residual_delta_topic",
            "residual_delta",
            lambda message: list(message.data),
        )
        self._subscribe(
            Float32MultiArray,
            "final_delta_topic",
            "final_delta",
            lambda message: list(message.data),
        )
        self.create_subscription(
            String,
            str(self.get_parameter("policy_debug_topic").value),
            self._policy_debug_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("activation_debug_topic").value),
            self._activation_debug_callback,
            10,
        )

        self.get_logger().info(
            "Policy stream audit started. It only subscribes and writes CSV/JSON."
        )
        self.get_logger().info(
            "No action, IK, trajectory, gripper, or robot-command interface is created."
        )

    def _declare_parameters(self):
        self.declare_parameter("confidence_topic", "/slot_detector/confidence")
        self.declare_parameter("slot_width_topic", "/slot_detector/slot_width")
        self.declare_parameter(
            "slot_pose_base_topic", "/bookshelf_policy/slot_pose_base"
        )
        self.declare_parameter(
            "book_pose_base_topic", "/bookshelf_policy/book_pose_base"
        )
        self.declare_parameter("raw_metrics_topic", "/bookshelf_policy/raw_metrics")
        self.declare_parameter(
            "observation_topic", "/bookshelf_policy/observation_12d"
        )
        self.declare_parameter(
            "observation_valid_topic", "/bookshelf_policy/observation_valid"
        )
        self.declare_parameter(
            "adapter_debug_topic", "/bookshelf_policy/adapter_debug"
        )
        self.declare_parameter(
            "inference_valid_topic", "/bookshelf_shadow/inference_valid"
        )
        self.declare_parameter(
            "policy_action_topic", "/bookshelf_shadow/residual_policy_action"
        )
        self.declare_parameter(
            "nominal_delta_topic", "/bookshelf_shadow/nominal_delta"
        )
        self.declare_parameter(
            "residual_delta_topic", "/bookshelf_shadow/residual_delta"
        )
        self.declare_parameter("final_delta_topic", "/bookshelf_shadow/final_delta")
        self.declare_parameter(
            "policy_debug_topic", "/bookshelf_shadow/policy_debug"
        )
        self.declare_parameter(
            "activation_debug_topic", "/bookshelf_shadow/policy_activation_debug"
        )
        self.declare_parameter("expected_base_frame", "link_base")
        self.declare_parameter("pair_max_age_s", 0.20)
        self.declare_parameter("target_samples", 1200)
        self.declare_parameter("reference_slot_width_m", 0.0)
        self.declare_parameter(
            "output_dir", "/tmp/bookshelf_policy_stream_audit"
        )

    def _subscribe(self, message_type, topic_parameter, key, converter):
        topic = str(self.get_parameter(topic_parameter).value)

        def callback(message):
            try:
                value = converter(message)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self.accumulator.add_invalid(f"{key} decode failed: {error}")
                return
            self.latest[key] = (value, time.monotonic())

        self.create_subscription(message_type, topic, callback, 10)

    @staticmethod
    def _json_payload(message):
        return json.loads(message.data)

    @staticmethod
    def _ordered_mapping(payload, key, labels):
        mapping = payload[key]
        return [float(mapping[label]) for label in labels]

    def _recent(self, key, now):
        entry = self.latest.get(key)
        if entry is None:
            raise ValueError(f"waiting for {key}")
        value, timestamp = entry
        maximum_age = float(self.get_parameter("pair_max_age_s").value)
        if maximum_age > 0.0 and now - timestamp > maximum_age:
            raise ValueError(f"{key} is stale")
        return value

    def _policy_debug_callback(self, message):
        try:
            policy_debug = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.accumulator.add_invalid(f"policy_debug decode failed: {error}")
            return

        if not bool(policy_debug.get("valid", False)):
            self.accumulator.add_invalid(
                f"policy invalid: {policy_debug.get('reason', 'unspecified')}"
            )
            self._maybe_report()
            return

        now = time.monotonic()
        try:
            if not self._recent("observation_valid", now):
                raise ValueError("observation_valid is false")
            if not self._recent("inference_valid", now):
                raise ValueError("inference_valid is false")
            adapter_debug = self._recent("adapter_debug", now)
            confidence = adapter_debug.get("slot_confidence")
            slot_width = adapter_debug.get("slot_width_m")
            if confidence is None:
                confidence = self._recent("confidence", now)
            if slot_width is None:
                slot_width = self._recent("slot_width", now)
            slot_pose = self._recent("slot_pose", now)
            book_pose = self._recent("book_pose", now)
            self._recent("raw_metrics", now)
            self._recent("observation", now)
            self._recent("policy_action", now)
            self._recent("nominal_delta", now)
            self._recent("residual_delta", now)
            self._recent("final_delta", now)
            expected_frame = str(self.get_parameter("expected_base_frame").value)
            if slot_pose.header.frame_id != expected_frame:
                raise ValueError(
                    f"slot pose frame is {slot_pose.header.frame_id!r}, "
                    f"expected {expected_frame!r}"
                )
            if book_pose.header.frame_id != expected_frame:
                raise ValueError(
                    f"book pose frame is {book_pose.header.frame_id!r}, "
                    f"expected {expected_frame!r}"
                )
            adapter_observation = np.asarray(
                adapter_debug["observation_12d"], dtype=np.float64
            )
            policy_observation = np.asarray(
                policy_debug["observation_12d"], dtype=np.float64
            )
            if not bool(policy_debug.get("vecnormalize_applied", False)):
                raise ValueError("policy debug did not apply VecNormalize")
            normalized_observation = np.asarray(
                policy_debug["normalized_observation"], dtype=np.float64
            )
            actor_mean = np.asarray(policy_debug["actor_mean"], dtype=np.float64)
            if (
                adapter_observation.shape != (len(OBSERVATION_LABELS),)
                or policy_observation.shape != (len(OBSERVATION_LABELS),)
                or not np.allclose(
                    adapter_observation,
                    policy_observation,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise ValueError("adapter/policy observation pairing mismatch")
            raw_metrics = self._ordered_mapping(
                adapter_debug,
                "raw_metrics",
                OBSERVATION_LABELS,
            )
            policy_action = self._ordered_mapping(
                policy_debug,
                "policy_action",
                POLICY_ACTION_LABELS,
            )
            nominal_delta = self._ordered_mapping(
                policy_debug,
                "nominal_delta",
                MOTION_LABELS,
            )
            residual_delta = self._ordered_mapping(
                policy_debug,
                "residual_delta",
                MOTION_LABELS,
            )
            final_delta = self._ordered_mapping(
                policy_debug,
                "final_delta",
                MOTION_LABELS,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.accumulator.add_invalid(str(error))
            self._maybe_report()
            return

        slot = slot_pose.pose
        book = book_pose.pose
        self.accumulator.add(
            confidence=confidence,
            slot_width=slot_width,
            slot_position=[
                slot.position.x,
                slot.position.y,
                slot.position.z,
            ],
            slot_quaternion_xyzw=[
                slot.orientation.x,
                slot.orientation.y,
                slot.orientation.z,
                slot.orientation.w,
            ],
            book_position=[
                book.position.x,
                book.position.y,
                book.position.z,
            ],
            book_quaternion_xyzw=[
                book.orientation.x,
                book.orientation.y,
                book.orientation.z,
                book.orientation.w,
            ],
            raw_metrics=raw_metrics,
            observation=policy_observation,
            normalized_observation=normalized_observation,
            actor_mean=actor_mean,
            policy_action=policy_action,
            nominal_delta=nominal_delta,
            residual_delta=residual_delta,
            final_delta=final_delta,
            book_pose_source=adapter_debug.get("book_pose_source", "unknown"),
            eef_book_transform_status=adapter_debug.get(
                "eef_book_transform_status", "unknown"
            ),
            policy_tool_transform_status=adapter_debug.get(
                "policy_tool_transform_status", "unknown"
            ),
            slot_pose_source=adapter_debug.get("slot_pose_source", "unknown"),
            static_slot_transform_status=adapter_debug.get(
                "static_slot_transform_status", "unknown"
            ),
        )
        self._maybe_report()

    def _activation_debug_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self.activation_accumulator.invalid_payloads += 1
            return
        self.activation_accumulator.add(payload)

    def _maybe_report(self):
        target = max(int(self.get_parameter("target_samples").value), 1)
        summary = self.accumulator.summary()
        total = int(summary["samples"])
        if total > 0 and total % 100 == 0:
            self.get_logger().info(
                f"samples={total}, complete={summary['complete_samples']}, "
                f"fraction={summary['complete_fraction']:.3f}"
            )
        if total >= target and not self.completed:
            self.completed = True
            self._write_report()

    def _write_report(self):
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_rows = list(self.accumulator.csv_rows())
        if csv_rows:
            rows_path = output_dir / "policy_stream_samples.csv"
            with rows_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)

        summary = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "hardware_commanded": False,
            "ground_truth_available": (
                self.accumulator.reference_slot_width_m > 0.0
            ),
            "pairing_strategy": (
                "adapter and policy debug observations must match; all numeric "
                "topics and base-frame poses must be fresh within pair_max_age_s; "
                "configured slot width/confidence may come directly from adapter_debug"
            ),
            "policy_stream": self.accumulator.summary(),
            "policy_activation": self.activation_accumulator.summary(),
            "limitations": [
                "Policy outputs are diagnostics and were not executed.",
                "Slot pose accuracy is unverified without a physical reference.",
                "Transform status fields distinguish measured calibration from approximate inputs.",
                "Pose topics are paired by arrival-time freshness because they have no shared sequence ID.",
            ],
        }
        summary_path = output_dir / "policy_stream_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.get_logger().info(f"Policy stream audit written to {summary_path}")
        stream = summary["policy_stream"]
        width = stream.get("slot_width_m", {}).get("mean")
        width_text = "n/a" if width is None else f"{width * 1000.0:.3f} mm"
        self.get_logger().info(
            "CALIBRATED SHADOW SUMMARY: "
            f"complete={stream['complete_samples']}/{stream['samples']} "
            f"({100.0 * stream['complete_fraction']:.1f}%), "
            f"slot_width={width_text}, "
            f"observation_clip_fraction="
            f"{stream.get('observation_clip_fraction', float('nan')):.4f}"
        )
        self.get_logger().info(
            "PROVENANCE: "
            f"slot_sources={stream['slot_pose_sources']}, "
            f"slot_statuses={stream['static_slot_transform_statuses']}, "
            f"book_sources={stream['book_pose_sources']}, "
            f"book_statuses={stream['eef_book_transform_statuses']}"
        )
        self.get_logger().info(
            "POLICY TOOL PROVENANCE: "
            f"{stream['policy_tool_transform_statuses']}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyStreamAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.accumulator.summary()["samples"] and not node.completed:
            node._write_report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
