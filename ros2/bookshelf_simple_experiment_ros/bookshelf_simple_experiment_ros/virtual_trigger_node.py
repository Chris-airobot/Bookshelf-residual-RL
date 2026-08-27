#!/usr/bin/env python3
"""Initialize fake xArm away from the task, then request the virtual plan once."""

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


FAR_START_JOINT_POSITIONS = [
    0.4342693425054612,
    1.5322427671441177,
    4.904658882462919,
    1.302429752118059,
    3.302595179623167,
    0.6839448116011184,
    4.4791192150828865,
]


class VirtualTriggerNode(Node):
    def __init__(self):
        super().__init__("virtual_preinsert_trigger")
        self.client = self.create_client(
            Trigger, "/bookshelf_simple/plan_preinsert"
        )
        self.declare_parameter(
            "initial_joint_positions",
            FAR_START_JOINT_POSITIONS,
        )
        self.declare_parameter("initial_move_duration_s", 2.0)
        self.joint_names = [f"joint{index}" for index in range(1, 8)]
        self.initial_positions = [
            float(value)
            for value in self.get_parameter("initial_joint_positions").value
        ]
        if len(self.initial_positions) != len(self.joint_names):
            raise ValueError("virtual initial pose must contain seven joints")
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/xarm7_traj_controller/follow_joint_trajectory",
        )
        self.initialization_submitted = False
        self.initialization_complete = False
        self.pending = False
        self.finished = False
        self.create_timer(1.0, self._try_trigger)

    def _try_trigger(self):
        if self.finished or self.pending:
            return
        if not self.initialization_complete:
            self._try_initialize()
            return
        if not self.client.service_is_ready():
            return
        self.pending = True
        future = self.client.call_async(Trigger.Request())
        future.add_done_callback(self._response)

    def _try_initialize(self):
        if self.initialization_submitted:
            return
        if not self.trajectory_client.server_is_ready():
            return
        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.initial_positions
        duration_ns = int(round(
            max(float(self.get_parameter("initial_move_duration_s").value), 0.1)
            * 1.0e9
        ))
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        trajectory.trajectory.points = [point]
        trajectory.trajectory.header.stamp = self.get_clock().now().to_msg()
        self.initialization_submitted = True
        future = self.trajectory_client.send_goal_async(trajectory)
        future.add_done_callback(self._initial_goal_response)

    def _initial_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.initialization_submitted = False
            self.get_logger().warning(f"virtual initialization failed: {error}")
            return
        if not goal_handle.accepted:
            self.initialization_submitted = False
            self.get_logger().warning("fake xArm rejected its initial-pose trajectory")
            return
        goal_handle.get_result_async().add_done_callback(self._initial_result)

    def _initial_result(self, future):
        try:
            error_code = int(future.result().result.error_code)
        except Exception as error:
            self.initialization_submitted = False
            self.get_logger().warning(f"virtual initialization result failed: {error}")
            return
        if error_code != 0:
            self.initialization_submitted = False
            self.get_logger().warning(
                f"fake xArm initial-pose trajectory failed with code {error_code}"
            )
            return
        self.initialization_complete = True
        self.get_logger().info("fake xArm reached the far virtual starting pose")

    def _response(self, future):
        self.pending = False
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(f"virtual trigger call failed: {error}")
            return
        if response.success:
            self.finished = True
            self.get_logger().info(response.message)
        else:
            self.get_logger().info(f"waiting to trigger: {response.message}")


def main(args=None):
    rclpy.init(args=args)
    node = VirtualTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
