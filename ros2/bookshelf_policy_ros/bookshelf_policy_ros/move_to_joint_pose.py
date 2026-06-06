#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class MoveToJointPose(Node):
    def __init__(self):
        super().__init__("move_to_joint_pose")

        # Change this after checking: ros2 action list | grep trajectory
        self.action_name = "/xarm7_traj_controller/follow_joint_trajectory"

        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self.target_positions = [
            -2.1387282432172583e-05,
            0.45204615237185575,
            -8.855606925539661e-05,
            1.1051906493842676,
            -3.6100247879922165e-05,
            -0.9463594873801702,
            -9.686126306007736e-05,
        ]

        self.client = ActionClient(self, FollowJointTrajectory, self.action_name)

    def send_goal(self):
        self.get_logger().info(f"Waiting for action server: {self.action_name}")
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available. Check controller/action name.")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.target_positions
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(sec=5, nanosec=0)

        goal.trajectory.points.append(point)

        self.get_logger().info("Sending joint trajectory goal...")
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected.")
            rclpy.shutdown()
            return

        self.get_logger().info("Goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Motion finished with error_code: {result.error_code}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveToJointPose()
    node.send_goal()
    rclpy.spin(node)


if __name__ == "__main__":
    main()