#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class ActionExecutorNode(Node):
    """
    Debug executor:
    - subscribes to /bookshelf_policy/action
    - reads current /joint_states
    - sends a small joint trajectory command

    This is NOT the final Cartesian controller.
    It only verifies that the RL-style action pipeline can move the robot.
    """

    def __init__(self):
        super().__init__("action_executor_node")

        # Change this if your controller action name is different.
        self.declare_parameter(
            "action_name",
            "/xarm7_traj_controller/follow_joint_trajectory",
        )
        self.action_name = self.get_parameter("action_name").value

        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self.current_joint_pos = None
        self.busy = False

        # Small joint-space debug scales in radians.
        # These are intentionally tiny.
        self.joint_debug_scale = 0.015

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            self.action_name,
        )

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.action_sub = self.create_subscription(
            Float32MultiArray,
            "/bookshelf_policy/action",
            self.action_callback,
            10,
        )

        self.get_logger().info(f"Action executor started.")
        self.get_logger().info(f"Using trajectory action: {self.action_name}")
        self.get_logger().info("Waiting for /joint_states and /bookshelf_policy/action...")

    def joint_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))

        try:
            self.current_joint_pos = np.array(
                [name_to_pos[name] for name in self.joint_names],
                dtype=np.float64,
            )
        except KeyError:
            # Joint names may not all be available yet.
            return

    def action_callback(self, msg: Float32MultiArray):
        if self.busy:
            return

        if self.current_joint_pos is None:
            self.get_logger().warn("No joint state received yet.")
            return

        action = np.array(msg.data, dtype=np.float64)

        if action.shape[0] != 6:
            self.get_logger().warn(f"Expected 6D action, got {action.shape[0]}D.")
            return

        action = np.clip(action, -1.0, 1.0)

        dx = action[0]
        dy = action[1]
        dz = action[2]
        dyaw = action[3]
        dpitch = action[4]
        release_signal = action[5]

        # Debug-only mapping from RL-style action to tiny joint offsets.
        # This is NOT physically equivalent to dx/dy/dz in task space.
        joint_delta = np.zeros(7, dtype=np.float64)

        joint_delta[0] = dy * self.joint_debug_scale       # joint1: side-ish
        joint_delta[1] = dx * self.joint_debug_scale       # joint2: forward-ish
        joint_delta[3] = dz * self.joint_debug_scale       # joint4: height-ish
        joint_delta[5] = dpitch * self.joint_debug_scale   # joint6: pitch-ish
        joint_delta[6] = dyaw * self.joint_debug_scale     # joint7: yaw-ish

        target = self.current_joint_pos + joint_delta

        self.get_logger().info(
            f"Received action={action.tolist()} | "
            f"joint_delta={joint_delta.tolist()} | "
            f"release={release_signal > 0.5}"
        )

        self.send_joint_goal(target)

    def send_joint_goal(self, target_positions):
        if not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f"Action server not available: {self.action_name}. "
                f"Run: ros2 action list | grep trajectory"
            )
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = target_positions.tolist()
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(sec=1, nanosec=0)

        goal.trajectory.points.append(point)

        self.busy = True
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            self.busy = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Trajectory finished. error_code={result.error_code}")
        self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()