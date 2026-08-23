#!/usr/bin/env python3
"""Start MoveIt Servo against the xArm stack that is already running."""

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import (
    generate_ros2_control_params_temp_file,
    load_yaml,
)


def _servo_server(context):
    robot_ip = LaunchConfiguration("robot_ip")
    dof = LaunchConfiguration("dof")
    robot_type = LaunchConfiguration("robot_type")
    prefix = LaunchConfiguration("prefix")
    hw_ns = LaunchConfiguration("hw_ns")
    limited = LaunchConfiguration("limited")
    add_gripper = LaunchConfiguration("add_gripper")
    use_fake_hardware = (
        LaunchConfiguration("use_fake_hardware").perform(context).lower() == "true"
    )

    robot_type_value = robot_type.perform(context)
    dof_value = dof.perform(context)
    prefix_value = prefix.perform(context)
    xarm_type = (
        f"{robot_type_value}{dof_value}"
        if robot_type_value in ("xarm", "lite")
        else robot_type_value
    )
    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(
            get_package_share_directory("xarm_controller"),
            "config",
            f"{xarm_type}_controllers.yaml",
        ),
        prefix=prefix_value,
        add_gripper=add_gripper.perform(context).lower() == "true",
        add_bio_gripper=False,
        ros_namespace="",
        robot_type=robot_type_value,
    )
    controllers_name = "fake_controllers" if use_fake_hardware else "controllers"
    ros2_control_plugin = (
        "uf_robot_hardware/UFRobotFakeSystemHardware"
        if use_fake_hardware
        else "uf_robot_hardware/UFRobotSystemHardware"
    )
    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name=controllers_name,
        robot_ip=robot_ip,
        report_type="normal",
        dof=dof,
        robot_type=robot_type,
        prefix=prefix,
        hw_ns=hw_ns,
        limited=limited,
        effort_control=False,
        velocity_control=False,
        ros2_control_plugin=ros2_control_plugin,
        ros2_control_params=ros2_control_params,
        add_gripper=add_gripper,
    ).to_moveit_configs()

    robot_parameters = {}
    robot_parameters.update(moveit_config.robot_description)
    robot_parameters.update(moveit_config.robot_description_semantic)
    robot_parameters.update(moveit_config.robot_description_kinematics)
    robot_parameters.update(moveit_config.joint_limits)
    robot_parameters.update(moveit_config.planning_pipelines)

    servo = load_yaml("xarm_moveit_servo", "config/xarm_moveit_servo_config.yaml")
    servo["move_group_name"] = xarm_type
    servo["planning_frame"] = "link_base"
    servo["ee_frame_name"] = "link_eef"
    servo["robot_link_command_frame"] = "link_base"
    servo["command_out_topic"] = (
        f"/{prefix_value}{xarm_type}_traj_controller/joint_trajectory"
    )
    servo["cartesian_command_in_topic"] = "~/delta_twist_cmds"
    servo["incoming_command_timeout"] = 0.2
    servo["check_collisions"] = True

    return [
        LogInfo(
            msg=(
                "Starting Servo server only. It reuses the existing xArm "
                f"ROS2-control {'fake system' if use_fake_hardware else 'driver'} "
                "and xarm7 trajectory controller."
            )
        ),
        ComposableNodeContainer(
            name="bookshelf_moveit_servo_container",
            namespace="/",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="moveit_servo",
                    plugin="moveit_servo::ServoNode",
                    name="servo_server",
                    parameters=[{"moveit_servo": servo}, robot_parameters],
                )
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    """Declare the reusable Servo-only launch description."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip"),
            DeclareLaunchArgument("dof", default_value="7"),
            DeclareLaunchArgument("robot_type", default_value="xarm"),
            DeclareLaunchArgument("prefix", default_value=""),
            DeclareLaunchArgument("hw_ns", default_value="xarm"),
            DeclareLaunchArgument("limited", default_value="false"),
            DeclareLaunchArgument("add_gripper", default_value="true"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            OpaqueFunction(function=_servo_server),
        ]
    )
