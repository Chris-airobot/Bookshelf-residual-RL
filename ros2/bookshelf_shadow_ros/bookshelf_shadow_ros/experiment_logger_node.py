#!/usr/bin/env python3
"""Write compact event and provenance logs without exposing command interfaces."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .experiment_logging import git_snapshot, sha256_file


class ExperimentLoggerNode(Node):
    """Subscriber-only logger paired with the automatic rosbag recorder."""

    def __init__(self):
        super().__init__("bookshelf_experiment_logger")
        self._declare_parameters()
        self.run_dir = Path(str(self.get_parameter("run_dir").value)).expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.started_at = datetime.now().astimezone().isoformat()
        self.event_counts = Counter()
        self.latest_values = {}
        self._write_manifest(completed=False)

        self._subscribe_bool(
            "observation_valid_topic", "observation_valid"
        )
        self._subscribe_bool(
            "activation_ready_topic", "policy_activation_ready"
        )
        self._subscribe_bool("inference_valid_topic", "inference_valid")
        self._subscribe_bool("plan_valid_topic", "plan_valid")
        self._subscribe_string("adapter_debug_topic", "adapter_debug")
        self._subscribe_string("activation_debug_topic", "activation_debug")
        self._subscribe_string("policy_debug_topic", "policy_debug")
        self._subscribe_string("plan_report_topic", "plan_report")

        self.create_timer(5.0, self._write_graph_snapshot)
        self.get_logger().info(f"Automatic experiment log: {self.run_dir}")
        self.get_logger().info(
            "Logger is subscriber-only and has no action, IK, trajectory, "
            "controller, gripper, or robot-command interface."
        )

    def _declare_parameters(self):
        self.declare_parameter("run_dir", "/tmp/bookshelf_experiment")
        self.declare_parameter("trial_name", "unnamed")
        self.declare_parameter("repository_path", "")
        self.declare_parameter("policy_bundle_path", "")
        self.declare_parameter("activation_envelope_path", "")
        self.declare_parameter("camera_recording", True)
        self.declare_parameter(
            "observation_valid_topic", "/bookshelf_policy/observation_valid"
        )
        self.declare_parameter(
            "activation_ready_topic", "/bookshelf_shadow/policy_activation_ready"
        )
        self.declare_parameter(
            "inference_valid_topic", "/bookshelf_shadow/inference_valid"
        )
        self.declare_parameter("plan_valid_topic", "/bookshelf_guarded/plan_valid")
        self.declare_parameter(
            "adapter_debug_topic", "/bookshelf_policy/adapter_debug"
        )
        self.declare_parameter(
            "activation_debug_topic", "/bookshelf_shadow/policy_activation_debug"
        )
        self.declare_parameter(
            "policy_debug_topic", "/bookshelf_shadow/policy_debug"
        )
        self.declare_parameter(
            "plan_report_topic", "/bookshelf_guarded/plan_report"
        )

    def _subscribe_bool(self, parameter, event_name):
        topic = str(self.get_parameter(parameter).value)

        def callback(message):
            value = bool(message.data)
            if self.latest_values.get(event_name) != value:
                self.latest_values[event_name] = value
                self._append_event(event_name, value)

        self.create_subscription(Bool, topic, callback, 10)

    def _subscribe_string(self, parameter, event_name):
        topic = str(self.get_parameter(parameter).value)

        def callback(message):
            try:
                value = json.loads(message.data)
            except json.JSONDecodeError:
                value = {"raw": message.data}
            self.latest_values[event_name] = value
            self._append_event(event_name, value)

        self.create_subscription(String, topic, callback, 10)

    def _append_event(self, event_name, value):
        self.event_counts[event_name] += 1
        entry = {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "ros_time_ns": int(self.get_clock().now().nanoseconds),
            "event": event_name,
            "value": value,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")

    def _write_graph_snapshot(self):
        topics = {
            name: types
            for name, types in sorted(self.get_topic_names_and_types())
        }
        nodes = sorted(
            f"{namespace.rstrip('/')}/{name}".replace("//", "/")
            for name, namespace in self.get_node_names_and_namespaces()
        )
        document = {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "nodes": nodes,
            "topics": topics,
        }
        (self.run_dir / "ros_graph.json").write_text(
            json.dumps(document, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_manifest(self, *, completed):
        repository = str(self.get_parameter("repository_path").value).strip()
        bundle = str(self.get_parameter("policy_bundle_path").value).strip()
        envelope = str(
            self.get_parameter("activation_envelope_path").value
        ).strip()
        manifest = {
            "schema_version": 1,
            "trial_name": str(self.get_parameter("trial_name").value),
            "run_dir": str(self.run_dir.resolve()),
            "started_at": self.started_at,
            "completed_at": (
                datetime.now().astimezone().isoformat() if completed else None
            ),
            "completed_cleanly": bool(completed),
            "hardware_commanded_by_logger": False,
            "camera_recording": bool(
                self.get_parameter("camera_recording").value
            ),
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "ros_distro": os.environ.get("ROS_DISTRO"),
            },
            "repository": git_snapshot(repository) if repository else None,
            "policy_bundle": {
                "path": bundle or None,
                "sha256": sha256_file(bundle),
            },
            "activation_envelope": {
                "path": envelope or None,
                "sha256": sha256_file(envelope),
            },
            "event_counts": dict(sorted(self.event_counts.items())),
            "latest_values": self.latest_values,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def destroy_node(self):
        self._write_graph_snapshot()
        self._write_manifest(completed=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
