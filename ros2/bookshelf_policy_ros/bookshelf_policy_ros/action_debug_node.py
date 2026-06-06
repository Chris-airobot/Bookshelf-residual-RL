#!/usr/bin/env python3

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ActionDebugNode(Node):
    """
    This node does NOT move the robot yet.
    It outputs RL-style 6D actions and decodes them to physical deltas.

    Action format:
        a[0] = dx
        a[1] = dy
        a[2] = dz
        a[3] = dyaw
        a[4] = dpitch
        a[5] = release trigger
    """

    def __init__(self):
        super().__init__("action_debug_node")

        self.pub = self.create_publisher(Float32MultiArray, "/bookshelf_policy/action", 10)

        # Same scales as your Isaac env.
        self.dx_scale = 0.006
        self.dy_scale = 0.002
        self.dz_scale = 0.006
        self.dyaw_scale = math.radians(0.6)
        self.dpitch_scale = math.radians(0.6)
        self.release_threshold = 0.5

        # For first real-robot tests, reduce this to 0.3 or 0.5.
        # For Gazebo-only testing, you can use 1.0.
        self.safety_scale = 1.0

        self.test_actions = [
            ("hold",              [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("forward +x",        [0.5, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("backward -x",       [-0.5, 0.0, 0.0, 0.0, 0.0, -1.0]),
            ("left +y",           [0.0, 0.5, 0.0, 0.0, 0.0, -1.0]),
            ("right -y",          [0.0, -0.5, 0.0, 0.0, 0.0, -1.0]),
            ("up +z",             [0.0, 0.0, 0.5, 0.0, 0.0, -1.0]),
            ("down -z",           [0.0, 0.0, -0.5, 0.0, 0.0, -1.0]),
            ("yaw +",             [0.0, 0.0, 0.0, 0.5, 0.0, -1.0]),
            ("yaw -",             [0.0, 0.0, 0.0, -0.5, 0.0, -1.0]),
            ("pitch +",           [0.0, 0.0, 0.0, 0.0, 0.5, -1.0]),
            ("pitch -",           [0.0, 0.0, 0.0, 0.0, -0.5, -1.0]),
            ("release trigger",   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ]

        self.index = 0
        self.timer = self.create_timer(1.0, self.timer_callback)

    def decode_action(self, action):
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

        dx = float(a[0]) * self.dx_scale * self.safety_scale
        dy = float(a[1]) * self.dy_scale * self.safety_scale
        dz = float(a[2]) * self.dz_scale * self.safety_scale
        dyaw = float(a[3]) * self.dyaw_scale * self.safety_scale
        dpitch = float(a[4]) * self.dpitch_scale * self.safety_scale
        release = float(a[5]) > self.release_threshold

        return dx, dy, dz, dyaw, dpitch, release

    def timer_callback(self):
        name, action = self.test_actions[self.index]
        dx, dy, dz, dyaw, dpitch, release = self.decode_action(action)

        msg = Float32MultiArray()
        msg.data = [float(x) for x in action]
        self.pub.publish(msg)

        self.get_logger().info(
            f"{name:16s} | action={action} | "
            f"dx={dx*1000:.2f} mm, dy={dy*1000:.2f} mm, dz={dz*1000:.2f} mm, "
            f"dyaw={math.degrees(dyaw):.3f} deg, "
            f"dpitch={math.degrees(dpitch):.3f} deg, "
            f"release={release}"
        )

        self.index = (self.index + 1) % len(self.test_actions)


def main(args=None):
    rclpy.init(args=args)
    node = ActionDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()