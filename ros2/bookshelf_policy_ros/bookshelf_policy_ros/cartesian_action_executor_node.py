#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK

import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class CartesianActionExecutorNode(Node):
    """
    Executes RL-style 6D action using MoveIt IK + FollowJointTrajectory.

    Action format:
        a[0] = dx
        a[1] = dy
        a[2] = dz
        a[3] = dyaw
        a[4] = dpitch
        a[5] = release trigger

    This node:
        1. Reads current EE pose from TF.
        2. Adds Cartesian delta.
        3. Calls MoveIt /compute_ik.
        4. Sends the IK joint target to trajectory controller.
    """

    def __init__(self):
        super().__init__("cartesian_action_executor_node")

        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("group_name", "xarm7")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("trajectory_action", "/xarm7_traj_controller/follow_joint_trajectory")

        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.group_name = self.get_parameter("group_name").value
        self.ik_service_name = self.get_parameter("ik_service").value
        self.trajectory_action = self.get_parameter("trajectory_action").value

        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # Same Isaac action scales.
        self.dx_scale = 0.006
        self.dy_scale = 0.002
        self.dz_scale = 0.006
        self.dyaw_scale = math.radians(0.6)
        self.dpitch_scale = math.radians(0.6)
        self.release_threshold = 0.5

        # Start conservative. For Gazebo you can set this to 1.0 later.
        self.safety_scale = 0.5

        self.current_joint_state = None
        self.busy = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ik_client = self.create_client(GetPositionIK, self.ik_service_name)
        self.traj_client = ActionClient(self, FollowJointTrajectory, self.trajectory_action)

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

        self.get_logger().info("Cartesian action executor started.")
        self.get_logger().info(f"base_frame: {self.base_frame}")
        self.get_logger().info(f"ee_frame: {self.ee_frame}")
        self.get_logger().info(f"group_name: {self.group_name}")
        self.get_logger().info(f"IK service: {self.ik_service_name}")
        self.get_logger().info(f"Trajectory action: {self.trajectory_action}")

    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg

    def decode_action(self, data):
        action = np.array(data, dtype=np.float64)

        if action.shape[0] != 6:
            raise ValueError(f"Expected 6D action, got {action.shape[0]}D.")

        action = np.clip(action, -1.0, 1.0)

        dx = action[0] * self.dx_scale * self.safety_scale
        dy = action[1] * self.dy_scale * self.safety_scale
        dz = action[2] * self.dz_scale * self.safety_scale
        dyaw = action[3] * self.dyaw_scale * self.safety_scale
        dpitch = action[4] * self.dpitch_scale * self.safety_scale
        release = action[5] > self.release_threshold

        return dx, dy, dz, dyaw, dpitch, release

    def get_current_ee_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().warn(f"Could not get TF {self.base_frame} <- {self.ee_frame}: {e}")
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation
        return pose

    def make_target_pose(self, current_pose, dx, dy, dz, dyaw, dpitch):
        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = self.get_clock().now().to_msg()

        # Deltas are applied in the base frame, matching the Isaac env-style env/world frame.
        target.pose.position.x = current_pose.pose.position.x + float(dx)
        target.pose.position.y = current_pose.pose.position.y + float(dy)
        target.pose.position.z = current_pose.pose.position.z + float(dz)

        q = current_pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        target_q = quaternion_from_euler(
            roll,
            pitch + float(dpitch),
            yaw + float(dyaw),
        )

        target.pose.orientation.x = target_q[0]
        target.pose.orientation.y = target_q[1]
        target.pose.orientation.z = target_q[2]
        target.pose.orientation.w = target_q[3]

        return target

    def action_callback(self, msg: Float32MultiArray):
        if self.busy:
            return

        if self.current_joint_state is None:
            self.get_logger().warn("No /joint_states received yet.")
            return

        try:
            dx, dy, dz, dyaw, dpitch, release = self.decode_action(msg.data)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return

        if release:
            self.get_logger().info("Release trigger received. For now, only logging release=True.")

        current_pose = self.get_current_ee_pose()
        if current_pose is None:
            return

        target_pose = self.make_target_pose(current_pose, dx, dy, dz, dyaw, dpitch)

        self.get_logger().info(
            f"Action decoded: dx={dx*1000:.2f}mm, dy={dy*1000:.2f}mm, dz={dz*1000:.2f}mm, "
            f"dyaw={math.degrees(dyaw):.3f}deg, dpitch={math.degrees(dpitch):.3f}deg, release={release}"
        )

        self.call_ik(target_pose)

    def call_ik(self, target_pose: PoseStamped):
        if not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                f"IK service not available: {self.ik_service_name}. "
                "Make sure MoveIt move_group is running."
            )
            return

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = target_pose
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(0.2 * 1e9)

        # Seed with current joint state.
        req.ik_request.robot_state.joint_state = self.current_joint_state

        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_response_callback)
        self.busy = True

    def ik_response_callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"IK call failed: {e}")
            self.busy = False
            return

        if res.error_code.val != 1:
            self.get_logger().warn(f"IK failed. MoveIt error_code={res.error_code.val}")
            self.busy = False
            return

        name_to_pos = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))

        try:
            target_positions = [name_to_pos[name] for name in self.joint_names]
        except KeyError as e:
            self.get_logger().error(f"IK result missing joint: {e}")
            self.busy = False
            return

        self.send_joint_goal(target_positions)

    def send_joint_goal(self, target_positions):
        if not self.traj_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                f"Trajectory action server not available: {self.trajectory_action}"
            )
            self.busy = False
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in target_positions]
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(sec=1, nanosec=0)

        goal.trajectory.points.append(point)

        send_future = self.traj_client.send_goal_async(goal)
        send_future.add_done_callback(self.goal_response_callback)

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
    node = CartesianActionExecutorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()