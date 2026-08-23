#!/usr/bin/env python3
"""Put the official fake xArm at the reviewed policy start pose."""

from __future__ import annotations

import json
import math

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectoryPoint


class FakePretargetInitializer(Node):
    """Initialize fake hardware, then release simulated policy control."""

    def __init__(self):
        """Create the simulation-only trajectory initializer."""
        super().__init__("fake_pretarget_initializer")
        self.declare_parameter(
            "joint_names", [f"joint{index}" for index in range(1, 8)]
        )
        self.declare_parameter("joint_positions", [0.0] * 7)
        self.declare_parameter(
            "trajectory_action",
            "/xarm7_traj_controller/follow_joint_trajectory",
        )
        self.declare_parameter("move_duration_s", 0.5)
        self.declare_parameter("joint_tolerance_rad", 0.002)
        self.declare_parameter("startup_timeout_s", 20.0)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "control_enable_topic", "/bookshelf_sim/pretarget_ready"
        )
        self.declare_parameter(
            "status_topic", "/bookshelf_sim/pretarget_status"
        )

        self.joint_names = [
            str(value) for value in self.get_parameter("joint_names").value
        ]
        self.joint_positions = [
            float(value)
            for value in self.get_parameter("joint_positions").value
        ]
        if len(self.joint_names) != 7 or len(self.joint_positions) != 7:
            raise ValueError(
                "fake pre-target requires seven joints and positions"
            )
        if not all(math.isfinite(value) for value in self.joint_positions):
            raise ValueError("fake pre-target joint positions must be finite")

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.enable_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("control_enable_topic").value),
            latched,
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), latched
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            10,
        )
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
        )

        self.latest_positions = None
        self.goal_sent = False
        self.goal_finished = False
        self.ready = False
        self.start_ns = self.get_clock().now().nanoseconds
        self.timer = self.create_timer(0.1, self._timer_callback)
        self._publish(False, "waiting for fake xArm controller")

    def _joint_state_callback(self, message: JointState):
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in self.joint_names):
            self.latest_positions = [
                float(by_name[name]) for name in self.joint_names
            ]

    def _timer_callback(self):
        if self.ready:
            return
        elapsed_s = (
            self.get_clock().now().nanoseconds - self.start_ns
        ) * 1.0e-9
        if elapsed_s > float(self.get_parameter("startup_timeout_s").value):
            self._publish(False, "fake pre-target initialization timed out")
            return
        if not self.goal_sent:
            if (
                self.latest_positions is None
                or not self.action_client.server_is_ready()
            ):
                return
            self._send_goal()
            return
        if not self.goal_finished or self.latest_positions is None:
            return

        maximum_error = max(
            abs(actual - expected)
            for actual, expected in zip(
                self.latest_positions, self.joint_positions
            )
        )
        tolerance = float(self.get_parameter("joint_tolerance_rad").value)
        if maximum_error <= tolerance:
            self.ready = True
            self._publish(True, "fake xArm initialized at reviewed pre-target")
            self.get_logger().info(
                "Fake xArm pre-target is ready; simulated policy control "
                "enabled."
            )

    def _send_goal(self):
        duration_s = float(self.get_parameter("move_duration_s").value)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            self._publish(False, "move_duration_s must be finite and positive")
            self.goal_sent = True
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(self.joint_positions)
        duration_ns = int(round(duration_s * 1.0e9))
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]
        self.goal_sent = True
        self._publish(False, "moving fake xArm to reviewed pre-target")
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._publish(False, f"fake pre-target goal failed: {error}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self._publish(False, "fake pre-target goal was rejected")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        try:
            wrapped_result = future.result()
            error_code = int(wrapped_result.result.error_code)
        except Exception as error:
            self._publish(False, f"fake pre-target result failed: {error}")
            return
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self._publish(
                False, f"fake pre-target trajectory error {error_code}"
            )
            return
        self.goal_finished = True
        self._publish(False, "confirming fake pre-target joint state")

    def _publish(self, ready: bool, reason: str):
        self.enable_publisher.publish(Bool(data=bool(ready)))
        report = {
            "ready": bool(ready),
            "reason": str(reason),
            "joint_names": self.joint_names,
            "joint_positions": self.joint_positions,
            "simulation_only": True,
            "hardware_commanded": False,
        }
        self.status_publisher.publish(
            String(data=json.dumps(report, sort_keys=True))
        )


def main(args=None):
    """Run the fake pre-target initializer."""
    rclpy.init(args=args)
    node = FakePretargetInitializer()
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
