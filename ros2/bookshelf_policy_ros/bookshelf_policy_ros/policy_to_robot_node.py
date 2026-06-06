#!/usr/bin/env python3

import math
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK

import tf2_ros
from stable_baselines3 import PPO


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def euler_from_quat_xyzw(q):
    x, y, z, w = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_xyzw_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float64)


class PolicyToRobotNode(Node):
    """
    Combined dummy-observation policy execution node.

    Pipeline:
        dummy 12D obs
        -> PPO model
        -> 6D action [dx, dy, dz, dyaw, dpitch, g]
        -> MoveIt IK
        -> FollowJointTrajectory
        -> xArm moves

    This is for Gazebo testing first.
    """

    def __init__(self):
        super().__init__("policy_to_robot_node")

        self.declare_parameter(
            "checkpoint_path",
            "/home/chris/RL/bookshelf/logs/sb3/Bookshelf-Direct-v5/2026-06-01_04-35-51/model.zip",
        )
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("group_name", "xarm7")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("trajectory_action", "/xarm7_traj_controller/follow_joint_trajectory")

        # For safety, start very small.
        self.declare_parameter("safety_scale", 0.2)

        # Timer period. 1.0 means one policy action per second.
        self.declare_parameter("period_s", 1.0)

        # Set false if you only want to print policy output.
        self.declare_parameter("execute", True)

        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.group_name = self.get_parameter("group_name").value
        self.ik_service_name = self.get_parameter("ik_service").value
        self.trajectory_action = self.get_parameter("trajectory_action").value
        self.safety_scale = float(self.get_parameter("safety_scale").value)
        self.period_s = float(self.get_parameter("period_s").value)
        self.execute = bool(self.get_parameter("execute").value)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.get_logger().info(f"Loading PPO model: {self.checkpoint_path}")
        self.model = PPO.load(self.checkpoint_path, env=None, print_system_info=False)

        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # Same action scales as your current Isaac env.
        self.dx_scale = 0.006
        self.dy_scale = 0.002
        self.dz_scale = 0.006
        self.dyaw_scale = math.radians(0.6)
        self.dpitch_scale = math.radians(0.6)
        self.release_threshold = 0.5

        # Dummy normalized 12D observation.
        # Replace this later with a real observation builder module.
        self.dummy_obs = np.array(
            [
                0.0,   # mode: INSERT
                0.0,   # rear_to_mouth
                0.0,   # front_to_back
                0.0,   # lat_err
                0.0,   # z_err
                0.0,   # yaw_err
                0.0,   # tool_to_book_x
                0.0,   # tool_to_book_y
                0.0,   # tool_to_book_z
                0.0,   # gripper_open01
                0.0,   # tilt_x
                0.0,   # tilt_y
            ],
            dtype=np.float32,
        )

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

        self.timer = self.create_timer(self.period_s, self.timer_callback)

        self.get_logger().info("Policy-to-robot node started.")
        self.get_logger().info(f"execute={self.execute}, safety_scale={self.safety_scale}")
        self.get_logger().info(f"base_frame={self.base_frame}, ee_frame={self.ee_frame}")
        self.get_logger().info(f"group_name={self.group_name}")
        self.get_logger().info(f"trajectory_action={self.trajectory_action}")

    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg

    def get_policy_action(self):
        obs_batch = self.dummy_obs.reshape(1, -1)
        action, _ = self.model.predict(obs_batch, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape[0] != 6:
            raise RuntimeError(f"Expected 6D action, got {action.shape[0]}D: {action}")

        return np.clip(action, -1.0, 1.0)

    def decode_action(self, action):
        dx = float(action[0]) * self.dx_scale * self.safety_scale
        dy = float(action[1]) * self.dy_scale * self.safety_scale
        dz = float(action[2]) * self.dz_scale * self.safety_scale
        dyaw = float(action[3]) * self.dyaw_scale * self.safety_scale
        dpitch = float(action[4]) * self.dpitch_scale * self.safety_scale
        release = float(action[5]) > self.release_threshold
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

        target.pose.position.x = current_pose.pose.position.x + dx
        target.pose.position.y = current_pose.pose.position.y + dy
        target.pose.position.z = current_pose.pose.position.z + dz

        q = current_pose.pose.orientation
        roll, pitch, yaw = euler_from_quat_xyzw([q.x, q.y, q.z, q.w])

        q_target = quat_xyzw_from_euler(
            roll,
            wrap_to_pi(pitch + dpitch),
            wrap_to_pi(yaw + dyaw),
        )

        target.pose.orientation.x = float(q_target[0])
        target.pose.orientation.y = float(q_target[1])
        target.pose.orientation.z = float(q_target[2])
        target.pose.orientation.w = float(q_target[3])

        return target

    def timer_callback(self):
        if self.busy:
            return

        if self.current_joint_state is None:
            self.get_logger().warn("No /joint_states received yet.")
            return

        try:
            action = self.get_policy_action()
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return

        dx, dy, dz, dyaw, dpitch, release = self.decode_action(action)

        self.get_logger().info(
            "dummy_obs="
            + np.array2string(self.dummy_obs, precision=3, suppress_small=True)
            + " -> action="
            + np.array2string(action, precision=3, suppress_small=True)
            + f" | decoded: dx={dx*1000:.2f}mm, dy={dy*1000:.2f}mm, dz={dz*1000:.2f}mm, "
            + f"dyaw={math.degrees(dyaw):.3f}deg, dpitch={math.degrees(dpitch):.3f}deg, "
            + f"release={release}"
        )

        if release:
            self.get_logger().warn("Release trigger is TRUE. For now, this node only logs it.")

        if not self.execute:
            return

        current_pose = self.get_current_ee_pose()
        if current_pose is None:
            return

        target_pose = self.make_target_pose(current_pose, dx, dy, dz, dyaw, dpitch)
        self.call_ik(target_pose)

    def call_ik(self, target_pose):
        if not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"IK service not available: {self.ik_service_name}")
            return

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = target_pose
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(0.2 * 1e9)
        req.ik_request.robot_state.joint_state = self.current_joint_state

        self.busy = True
        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_response_callback)

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
            self.get_logger().error(f"Trajectory action server not available: {self.trajectory_action}")
            self.busy = False
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in target_positions]
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(sec=1, nanosec=0)

        goal.trajectory.points.append(point)

        future = self.traj_client.send_goal_async(goal)
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
    node = PolicyToRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()