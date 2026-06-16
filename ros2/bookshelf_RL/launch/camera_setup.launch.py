from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Declare the launch arguments
    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='camera',
        description='Camera unique name'
    )

    camera_namespace_arg = DeclareLaunchArgument(
        'camera_namespace',
        default_value='',
        description='Namespace for camera'
    )

    serial_no_arg = DeclareLaunchArgument(
        'serial_no',
        default_value='',
        description='Choose device by serial number'
    )

    # Start the RealSense node directly, forcing serial_no to be a string
    realsense_launch = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=LaunchConfiguration('camera_name'),
        namespace=LaunchConfiguration('camera_namespace'),
        output='screen',
        parameters=[
            {
                'serial_no': ParameterValue(LaunchConfiguration('serial_no'), value_type=str),
                'tf_prefix': '',
                # Pass suffix-only frame IDs; driver will prefix with camera_name automatically
                'base_frame_id': 'link',
                'depth_frame_id': 'depth_frame',
                'depth_optical_frame_id': 'depth_optical_frame',
                'infra1_frame_id': 'infra1_frame',
                'infra1_optical_frame_id': 'infra1_optical_frame',
                'infra2_frame_id': 'infra2_frame',
                'infra2_optical_frame_id': 'infra2_optical_frame',
                'color_frame_id': 'color_frame',
                'color_optical_frame_id': 'color_optical_frame',
                'publish_tf': True,
                'pointcloud.enable': True,
                'depth_module.depth_profile': '640x480x15',
            }
        ]
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add the launch arguments
    ld.add_action(camera_name_arg)
    ld.add_action(camera_namespace_arg)
    ld.add_action(serial_no_arg)

    # Add the included launch file
    ld.add_action(realsense_launch)

    return ld
