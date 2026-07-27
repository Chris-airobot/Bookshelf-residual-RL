#!/usr/bin/env python3
"""Display a live RGB stream from simulation or the real camera."""

import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class SimRgbCameraViewer(Node):
    def __init__(self):
        super().__init__("sim_rgb_camera_viewer")
        self.declare_parameter("image_topic", "/sim_camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/sim_camera/color/camera_info")
        self.declare_parameter("window_name", "Isaac Sim D435 RGB")

        image_topic = str(self.get_parameter("image_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.window_name = str(self.get_parameter("window_name").value)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_frame_time = None
        self.filtered_fps = 0.0
        self.received_camera_info = False

        self.create_subscription(Image, image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)

        self.get_logger().info(f"Waiting for RGB images on {image_topic}")
        self.get_logger().info(f"Waiting for CameraInfo on {camera_info_topic}")
        self.get_logger().info("Press Q or Escape in the image window to exit.")

    def _camera_info_callback(self, message):
        if self.received_camera_info:
            return
        self.received_camera_info = True
        self.get_logger().info(
            "CameraInfo received: "
            f"{message.width}x{message.height}, "
            f"fx={message.k[0]:.3f}, fy={message.k[4]:.3f}, "
            f"cx={message.k[2]:.3f}, cy={message.k[5]:.3f}, "
            f"frame={message.header.frame_id}"
        )

    def _image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert image: {error}")
            return

        now = time.monotonic()
        if self.last_frame_time is not None:
            period = max(now - self.last_frame_time, 1.0e-6)
            instantaneous_fps = 1.0 / period
            self.filtered_fps = (
                instantaneous_fps if self.filtered_fps <= 0.0 else 0.9 * self.filtered_fps + 0.1 * instantaneous_fps
            )
        self.last_frame_time = now

        display = frame.copy()
        cv2.putText(
            display,
            f"{message.width}x{message.height}  {self.filtered_fps:.1f} FPS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
        self.latest_frame = display

    def show_latest_frame(self):
        if self.latest_frame is None:
            return False
        cv2.imshow(self.window_name, self.latest_frame)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)


def main(args=None):
    rclpy.init(args=args)
    node = SimRgbCameraViewer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.show_latest_frame():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
