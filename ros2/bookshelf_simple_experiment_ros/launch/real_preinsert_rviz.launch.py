"""Start only RViz with the real xArm7 MoveIt model parameters."""

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file


def _rviz(context):
    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            "xarm7_controllers.yaml",
        ),
        prefix="",
        add_gripper=True,
        add_bio_gripper=False,
        ros_namespace="",
        robot_type="xarm",
    )
    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name="controllers",
        robot_ip=LaunchConfiguration("robot_ip"),
        report_type="normal",
        dof="7",
        robot_type="xarm",
        prefix="",
        hw_ns="xarm",
        limited="false",
        effort_control=False,
        velocity_control=False,
        ros2_control_plugin="uf_robot_hardware/UFRobotSystemHardware",
        ros2_control_params=ros2_control_params,
        add_gripper="true",
    ).to_moveit_configs()
    rviz_config = PathJoinSubstitution([
        FindPackageShare("bookshelf_simple_experiment_ros"),
        "rviz",
        "real_preinsert_workflow.rviz",
    ])
    return [Node(
        package="rviz2",
        executable="rviz2",
        name="bookshelf_simple_real_preinsert_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
        ],
        condition=IfCondition(LaunchConfiguration("show_rviz")),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.209"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        OpaqueFunction(function=_rviz),
    ])
