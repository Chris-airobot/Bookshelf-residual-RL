#!/usr/bin/env python3
"""Audit xArm TF candidates for the PPO policy tool without robot commands."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
import tf2_ros
import yaml

from .calibrated_preinsert_target_math import transform_to_dict
from .policy_observation_math import make_transform
from .policy_tool_frame_audit import (
    TrainingToolReference,
    candidate_frame_names,
    evaluate_policy_tool_candidate,
    midpoint_transform,
    summarize_candidates,
    unavailable_candidate,
)


def _stamp_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _message_to_matrix(message) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


class PolicyToolFrameAuditNode(Node):
    """Rank TF and finger-midpoint tool candidates in subscriber-only mode."""

    def __init__(self):
        super().__init__("policy_tool_frame_audit")
        self._declare_parameters()
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.eef_book_status = str(
            self.get_parameter("eef_book_transform_status").value
        ).strip()
        if self.eef_book_status.lower() in ("", "unknown", "unconfigured"):
            raise ValueError("eef_book_transform_status must have explicit provenance.")
        self.transform_eef_book = make_transform(
            self._vector("eef_book_translation_xyz", 3),
            self._vector("eef_book_quaternion_xyzw", 4),
        )
        self.reference = TrainingToolReference(
            hand_to_tool_m=float(self.get_parameter("sim_hand_to_tool_m").value),
            hand_to_book_m=float(self.get_parameter("sim_hand_to_book_m").value),
            minimum_norm_m=float(
                self.get_parameter("training_tool_to_book_norm_min_m").value
            ),
            maximum_norm_m=float(
                self.get_parameter("training_tool_to_book_norm_max_m").value
            ),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.report = None
        self.valid_publisher = self.create_publisher(
            Bool, str(self.get_parameter("valid_topic").value), 10
        )
        self.debug_publisher = self.create_publisher(
            String, str(self.get_parameter("debug_topic").value), 10
        )
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = output_dir / "policy_tool_frame_audit.json"

        frequency = max(float(self.get_parameter("audit_rate_hz").value), 0.2)
        self.timer = self.create_timer(1.0 / frequency, self._timer_callback)
        self.get_logger().info("Policy tool-frame audit started in READ-ONLY mode.")
        self.get_logger().info(
            "No IK, planning, trajectory, controller, gripper, or robot-command "
            "interface is created."
        )
        self.get_logger().info(f"Audit report: {self.report_path}")

    def _declare_parameters(self):
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("eef_book_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("eef_book_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("eef_book_transform_status", "unconfigured")
        self.declare_parameter("sim_hand_to_tool_m", 0.107)
        self.declare_parameter("sim_hand_to_book_m", 0.075)
        self.declare_parameter("training_tool_to_book_norm_min_m", 0.020)
        self.declare_parameter("training_tool_to_book_norm_max_m", 0.050)
        self.declare_parameter(
            "candidate_frames",
            [
                "link_eef",
                "link_tcp",
                "xarm_gripper_base_link",
                "left_finger",
                "right_finger",
                "left_finger_link",
                "right_finger_link",
                "left_inner_finger",
                "right_inner_finger",
                "left_inner_finger_link",
                "right_inner_finger_link",
            ],
        )
        self.declare_parameter(
            "midpoint_pairs",
            [
                "left_finger,right_finger",
                "left_finger_link,right_finger_link",
                "left_inner_finger,right_inner_finger",
                "left_inner_finger_link,right_inner_finger_link",
            ],
        )
        self.declare_parameter("discover_gripper_frames", True)
        self.declare_parameter("tf_lookup_timeout_s", 0.03)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("audit_rate_hz", 1.0)
        self.declare_parameter("output_dir", "/tmp/bookshelf_policy_tool_audit")
        self.declare_parameter(
            "valid_topic", "/bookshelf_shadow/policy_tool_audit_valid"
        )
        self.declare_parameter(
            "debug_topic", "/bookshelf_shadow/policy_tool_audit_debug"
        )

    def _vector(self, name: str, length: int) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Parameter {name} must contain {length} finite values.")
        return value

    def _known_frames(self) -> list[str]:
        try:
            payload = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
            return sorted(str(name) for name in payload)
        except Exception:
            return []

    def _candidate_frame_names(self, known_frames) -> list[str]:
        return candidate_frame_names(
            self.get_parameter("candidate_frames").value,
            known_frames,
            discover=bool(self.get_parameter("discover_gripper_frames").value),
        )

    def _lookup(self, frame: str):
        if frame == self.ee_frame:
            return np.eye(4, dtype=np.float64), None
        timeout = max(float(self.get_parameter("tf_lookup_timeout_s").value), 0.0)
        try:
            message = self.tf_buffer.lookup_transform(
                self.ee_frame,
                frame,
                Time(),
                timeout=Duration(seconds=timeout),
            )
        except Exception as error:
            return None, str(error)
        maximum_age = float(self.get_parameter("tf_max_age_s").value)
        stamp_ns = _stamp_nanoseconds(message.header.stamp)
        if maximum_age > 0.0 and stamp_ns != 0:
            age = abs(int(self.get_clock().now().nanoseconds) - stamp_ns) * 1.0e-9
            if age > maximum_age:
                return None, "transform is stale"
        return _message_to_matrix(message), None

    def _midpoint_pairs(self, known_frames) -> list[tuple[str, str]]:
        pairs = []
        for value in self.get_parameter("midpoint_pairs").value:
            parts = [part.strip() for part in str(value).split(",")]
            if len(parts) == 2 and all(parts):
                pairs.append((parts[0], parts[1]))
        known = set(known_frames)
        for left in known_frames:
            if "left" not in left.lower():
                continue
            index = left.lower().index("left")
            right = left[:index] + "right" + left[index + 4 :]
            if right in known:
                pairs.append((left, right))
        return list(dict.fromkeys(pairs))

    def _timer_callback(self):
        known_frames = self._known_frames()
        transforms = {}
        candidates = []
        for frame in self._candidate_frame_names(known_frames):
            transform, error = self._lookup(frame)
            if error:
                candidates.append(unavailable_candidate(frame, "tf_frame", error))
                continue
            transforms[frame] = transform
            candidates.append(
                evaluate_policy_tool_candidate(
                    frame,
                    transform,
                    self.transform_eef_book,
                    source="tf_frame",
                    reference=self.reference,
                )
            )

        for left, right in self._midpoint_pairs(known_frames):
            left_transform = transforms.get(left)
            right_transform = transforms.get(right)
            if left_transform is None:
                left_transform, left_error = self._lookup(left)
            else:
                left_error = None
            if right_transform is None:
                right_transform, right_error = self._lookup(right)
            else:
                right_error = None
            name = f"midpoint({left},{right})"
            if left_error or right_error:
                candidates.append(
                    unavailable_candidate(
                        name,
                        "derived_midpoint",
                        f"left={left_error}; right={right_error}",
                    )
                )
                continue
            candidates.append(
                evaluate_policy_tool_candidate(
                    name,
                    midpoint_transform(left_transform, right_transform),
                    self.transform_eef_book,
                    source="derived_midpoint_position_only",
                    reference=self.reference,
                )
            )

        summary = summarize_candidates(candidates, self.reference)
        self.report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "safety": {
                "shadow_only": True,
                "hardware_commanded": False,
                "ik_checked": False,
                "execution_authorized": False,
            },
            "eef_frame": self.ee_frame,
            "eef_book_transform_status": self.eef_book_status,
            "transform_eef_book": transform_to_dict(self.transform_eef_book),
            "known_tf_frames": known_frames,
            **summary,
        }
        temporary = self.report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.report_path)
        valid = bool(summary["available_count"] > 0)
        self.valid_publisher.publish(Bool(data=valid))
        self.debug_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "valid": valid,
                        "shadow_only": True,
                        "available_count": summary["available_count"],
                        "plausible_candidate_names": summary[
                            "plausible_candidate_names"
                        ],
                        "selection_required": True,
                        "selection_authorized": False,
                        "report_path": str(self.report_path),
                    },
                    sort_keys=True,
                )
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyToolFrameAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
