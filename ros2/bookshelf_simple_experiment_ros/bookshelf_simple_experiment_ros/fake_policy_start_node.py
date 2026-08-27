#!/usr/bin/env python3
"""Put only the official fake xArm/gripper at the reviewed policy start."""

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint


REVIEWED_PREINSERT_JOINT_POSITIONS = [
    1.2342693425054612,
    1.5322427671441177,
    4.904658882462919,
    1.302429752118059,
    3.302595179623167,
    0.6839448116011184,
    4.4791192150828865,
]


class FakePolicyStartNode(Node):
    def __init__(self):
        super().__init__("fake_policy_start")
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/xarm7_traj_controller/follow_joint_trajectory",
        )
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/xarm_gripper_traj_controller/follow_joint_trajectory",
        )
        self.arm_pending = False
        self.gripper_pending = False
        self.arm_complete = False
        self.initialization_complete = False
        self.create_timer(0.5, self._try_initialize)

    @staticmethod
    def _goal(joint_names, positions, duration_s):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        duration_ns = int(float(duration_s) * 1.0e9)
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]
        return goal

    def _try_initialize(self):
        if not self.arm_complete:
            if self.arm_pending or not self.arm_client.server_is_ready():
                return
            self.arm_pending = True
            future = self.arm_client.send_goal_async(self._goal(
                [f"joint{index}" for index in range(1, 8)],
                REVIEWED_PREINSERT_JOINT_POSITIONS,
                2.0,
            ))
            future.add_done_callback(self._arm_goal_response)
            return
        if self.gripper_pending or not self.gripper_client.server_is_ready():
            return
        self.gripper_pending = True
        future = self.gripper_client.send_goal_async(
            self._goal(["drive_joint"], [0.85], 0.5)
        )
        future.add_done_callback(self._gripper_goal_response)

    def _arm_goal_response(self, future):
        self._handle_goal_response(future, "arm", self._arm_result)

    def _gripper_goal_response(self, future):
        self._handle_goal_response(future, "gripper", self._gripper_result)

    def _handle_goal_response(self, future, label, result_callback):
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().warning(f"fake {label} initialization failed: {error}")
            self._reset_pending(label)
            return
        if not handle.accepted:
            self.get_logger().warning(f"fake {label} initialization was rejected")
            self._reset_pending(label)
            return
        handle.get_result_async().add_done_callback(result_callback)

    def _arm_result(self, future):
        if self._result_succeeded(future, "arm"):
            self.arm_complete = True
        self.arm_pending = False

    def _gripper_result(self, future):
        if self._result_succeeded(future, "gripper"):
            self.get_logger().info(
                "fake xArm reached reviewed pre-insertion with fake gripper closed"
            )
            self.initialization_complete = True
            return
        self.gripper_pending = False

    def _result_succeeded(self, future, label):
        try:
            code = int(future.result().result.error_code)
        except Exception as error:
            self.get_logger().warning(f"fake {label} result failed: {error}")
            return False
        if code != 0:
            self.get_logger().warning(f"fake {label} trajectory failed with code {code}")
            return False
        return True

    def _reset_pending(self, label):
        if label == "arm":
            self.arm_pending = False
        else:
            self.gripper_pending = False


def main(args=None):
    rclpy.init(args=args)
    node = FakePolicyStartNode()
    try:
        while rclpy.ok() and not node.initialization_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
