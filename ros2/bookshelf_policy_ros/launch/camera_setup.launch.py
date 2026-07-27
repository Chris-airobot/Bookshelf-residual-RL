from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    camera_name = LaunchConfiguration("camera_name")
    camera_namespace = LaunchConfiguration("camera_namespace")
    serial_no = LaunchConfiguration("serial_no")

    color_profile = LaunchConfiguration("color_profile")
    depth_profile = LaunchConfiguration("depth_profile")

    align_depth = LaunchConfiguration("align_depth")
    enable_sync = LaunchConfiguration("enable_sync")
    enable_pointcloud = LaunchConfiguration("enable_pointcloud")

    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name=camera_name,
        namespace=camera_namespace,
        output="screen",
        parameters=[
            {
                "serial_no": ParameterValue(serial_no, value_type=str),
                "camera_name": ParameterValue(camera_name, value_type=str),

                "tf_prefix": "",

                "base_frame_id": "link",
                "depth_frame_id": "depth_frame",
                "depth_optical_frame_id": "depth_optical_frame",
                "infra1_frame_id": "infra1_frame",
                "infra1_optical_frame_id": "infra1_optical_frame",
                "infra2_frame_id": "infra2_frame",
                "infra2_optical_frame_id": "infra2_optical_frame",
                "color_frame_id": "color_frame",
                "color_optical_frame_id": "color_optical_frame",

                "publish_tf": True,

                "enable_color": True,
                "enable_depth": True,
                "enable_infra1": False,
                "enable_infra2": False,

                "rgb_camera.color_profile": ParameterValue(
                    color_profile,
                    value_type=str,
                ),
                "depth_module.depth_profile": ParameterValue(
                    depth_profile,
                    value_type=str,
                ),

                "align_depth.enable": ParameterValue(
                    align_depth,
                    value_type=bool,
                ),
                "enable_sync": ParameterValue(
                    enable_sync,
                    value_type=bool,
                ),
                "pointcloud.enable": ParameterValue(
                    enable_pointcloud,
                    value_type=bool,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_name",
                default_value="camera",
            ),
            DeclareLaunchArgument(
                "camera_namespace",
                default_value="",
            ),
            DeclareLaunchArgument(
                "serial_no",
                default_value="",
            ),
            DeclareLaunchArgument(
                "color_profile",
                default_value="640x480x30",
            ),
            DeclareLaunchArgument(
                "depth_profile",
                default_value="640x480x30",
            ),
            DeclareLaunchArgument(
                "align_depth",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "enable_sync",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "enable_pointcloud",
                default_value="true",
            ),
            realsense_node,
        ]
    )
