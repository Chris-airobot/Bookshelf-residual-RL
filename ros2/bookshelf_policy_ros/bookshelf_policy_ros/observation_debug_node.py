#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration

from std_msgs.msg import Float32MultiArray

import tf2_ros
# from tf_transformations import euler_from_quaternion, quaternion_matrix


def quat_xyzw_to_rot_matrix(q):
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def euler_from_quat_xyzw(q):
    """Quaternion [x, y, z, w] -> roll, pitch, yaw."""
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



def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ObservationDebugNode(Node):
    """
    Builds the 12D observation used by the current Bookshelf-Direct-v5 policy.

    This first version assumes:
    - shelf/slot geometry is known in the robot base frame
    - book pose is estimated from the end-effector pose plus a fixed offset
    - gripper state is manually set by parameter for now
    """

    def __init__(self):
        super().__init__("observation_debug_node")

        # Frames
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("ee_frame", "link_eef")

        # Mode: insert/scripted/push
        self.declare_parameter("mode", "insert")

        # Shelf geometry, matching your Isaac-style default approximately.
        # You MUST calibrate these for the real shelf later.
        self.declare_parameter("slot_x_open", 0.67)
        self.declare_parameter("slot_x_back", 0.80)
        self.declare_parameter("slot_center_y", 0.0)
        self.declare_parameter("shelf_top_z", 0.05)
        self.declare_parameter("shelf_thickness", 0.02)

        # Book size: Isaac _BOOK_LWH = (length_x, height_y, thickness_z)
        self.declare_parameter("book_length", 0.152)
        self.declare_parameter("book_height", 0.229)
        self.declare_parameter("book_thickness", 0.020)

        # Estimated book pose relative to EE frame.
        # Start with this, then calibrate.
        self.declare_parameter("book_offset_x", 0.0)
        self.declare_parameter("book_offset_y", 0.0)
        self.declare_parameter("book_offset_z", 0.075)

        # Manual gripper state for now.
        self.declare_parameter("gripper_open01", 0.0)

        # Observation scales copied from env config.
        self.rear_to_mouth_obs_scale = 0.08
        self.front_to_back_obs_scale = 0.08
        self.lat_err_obs_scale = 0.05
        self.z_err_obs_scale = 0.05
        self.yaw_err_obs_scale = math.radians(30.0)
        self.tool_to_book_pos_obs_scale = 0.25

        self.pub = self.create_publisher(
            Float32MultiArray,
            "/bookshelf_policy/observation",
            10,
        )

        self.raw_pub = self.create_publisher(
            Float32MultiArray,
            "/bookshelf_policy/raw_observation",
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info("Observation debug node started.")
        self.get_logger().info("Publishing normalized obs to /bookshelf_policy/observation")
        self.get_logger().info("Publishing raw obs to /bookshelf_policy/raw_observation")

    def get_param(self, name):
        return self.get_parameter(name).value

    def get_ee_transform(self):
        base_frame = self.get_param("base_frame")
        ee_frame = self.get_param("ee_frame")

        try:
            tf = self.tf_buffer.lookup_transform(
                base_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.2),
            )
            return tf
        except Exception as e:
            self.get_logger().warn(f"TF unavailable {base_frame} <- {ee_frame}: {e}")
            return None

    def estimate_book_pose(self, tf):
        """
        Estimate book pose from EE pose + fixed offset.
        Returns:
            book_pos: np.array [x,y,z]
            book_quat_xyzw: np.array [x,y,z,w]
            tool_pos: np.array [x,y,z]
        """
        t = tf.transform.translation
        q = tf.transform.rotation

        tool_pos = np.array([t.x, t.y, t.z], dtype=np.float64)
        ee_quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        offset_ee = np.array(
            [
                float(self.get_param("book_offset_x")),
                float(self.get_param("book_offset_y")),
                float(self.get_param("book_offset_z")),
            ],
            dtype=np.float64,
        )

        rot = quat_xyzw_to_rot_matrix(ee_quat_xyzw)
        book_pos = tool_pos + rot @ offset_ee

        # First approximation: book orientation follows EE orientation.
        # Later you can replace this with calibrated book pose / vision.
        book_quat_xyzw = ee_quat_xyzw.copy()

        return book_pos, book_quat_xyzw, tool_pos

    def book_corners(self, book_pos, book_quat_xyzw):
        L = float(self.get_param("book_length"))
        H = float(self.get_param("book_height"))
        T = float(self.get_param("book_thickness"))

        hx, hy, hz = 0.5 * L, 0.5 * H, 0.5 * T

        local = np.array(
            [
                [ hx,  hy,  hz],
                [ hx,  hy, -hz],
                [ hx, -hy,  hz],
                [ hx, -hy, -hz],
                [-hx,  hy,  hz],
                [-hx,  hy, -hz],
                [-hx, -hy,  hz],
                [-hx, -hy, -hz],
            ],
            dtype=np.float64,
        )

        rot = quat_xyzw_to_rot_matrix(book_quat_xyzw)
        return (rot @ local.T).T + book_pos.reshape(1, 3)

    def compute_raw_observation(self, book_pos, book_quat_xyzw, tool_pos):
        corners = self.book_corners(book_pos, book_quat_xyzw)

        front_x = np.max(corners[:, 0])
        rear_x = np.min(corners[:, 0])

        slot_x_open = float(self.get_param("slot_x_open"))
        slot_x_back = float(self.get_param("slot_x_back"))
        slot_center_y = float(self.get_param("slot_center_y"))

        # Same idea as Isaac: mouth plane is near the robot-facing face of side books.
        # For upright books, this is approximately row mid-x - half book length.
        book_length = float(self.get_param("book_length"))
        slot_mouth_x = 0.5 * (slot_x_open + slot_x_back) - 0.5 * book_length

        rear_to_mouth = rear_x - slot_mouth_x
        front_to_back = slot_x_back - front_x

        lat_err = slot_center_y - book_pos[1]

        shelf_top_z = float(self.get_param("shelf_top_z"))
        shelf_thickness = float(self.get_param("shelf_thickness"))
        book_height = float(self.get_param("book_height"))
        z_target = shelf_top_z + shelf_thickness + 0.5 * book_height
        z_err = book_pos[2] - z_target

        roll, pitch, yaw = euler_from_quat_xyzw(book_quat_xyzw)
        yaw_err = wrap_to_pi(yaw)

        tool_to_book = tool_pos - book_pos

        gripper_open01 = float(self.get_param("gripper_open01"))

        # Book upright tilt obs from local spine direction [0,1,0] in world/base frame.
        rot = quat_xyzw_to_rot_matrix(book_quat_xyzw)
        spine_w = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tilt_x = clamp(float(spine_w[0]))
        tilt_y = clamp(float(spine_w[1]))

        mode = str(self.get_param("mode")).lower()
        if mode == "insert":
            mode_obs = 0.0
        elif mode == "scripted":
            mode_obs = 0.5
        elif mode == "push":
            mode_obs = 1.0
        else:
            self.get_logger().warn(f"Unknown mode={mode}, using insert.")
            mode_obs = 0.0

        raw = np.array(
            [
                mode_obs,
                rear_to_mouth,
                front_to_back,
                lat_err,
                z_err,
                yaw_err,
                tool_to_book[0],
                tool_to_book[1],
                tool_to_book[2],
                gripper_open01,
                tilt_x,
                tilt_y,
            ],
            dtype=np.float32,
        )

        return raw

    def normalize_observation(self, raw):
        obs = np.zeros(12, dtype=np.float32)

        obs[0] = raw[0]
        obs[1] = clamp(raw[1] / self.rear_to_mouth_obs_scale)
        obs[2] = clamp(raw[2] / self.front_to_back_obs_scale)
        obs[3] = clamp(raw[3] / self.lat_err_obs_scale)
        obs[4] = clamp(raw[4] / self.z_err_obs_scale)
        obs[5] = clamp(raw[5] / self.yaw_err_obs_scale)
        obs[6] = clamp(raw[6] / self.tool_to_book_pos_obs_scale)
        obs[7] = clamp(raw[7] / self.tool_to_book_pos_obs_scale)
        obs[8] = clamp(raw[8] / self.tool_to_book_pos_obs_scale)
        obs[9] = clamp(raw[9])
        obs[10] = clamp(raw[10])
        obs[11] = clamp(raw[11])

        return obs

    def timer_callback(self):
        tf = self.get_ee_transform()
        if tf is None:
            return

        book_pos, book_quat_xyzw, tool_pos = self.estimate_book_pose(tf)
        raw = self.compute_raw_observation(book_pos, book_quat_xyzw, tool_pos)
        obs = self.normalize_observation(raw)

        raw_msg = Float32MultiArray()
        raw_msg.data = [float(x) for x in raw]
        self.raw_pub.publish(raw_msg)

        obs_msg = Float32MultiArray()
        obs_msg.data = [float(x) for x in obs]
        self.pub.publish(obs_msg)

        self.get_logger().info(
            "obs="
            + np.array2string(obs, precision=3, suppress_small=True)
            + " | raw="
            + np.array2string(raw, precision=4, suppress_small=True)
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObservationDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()