"""Start only MoveIt Servo against an already-running official fake xArm7."""

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import (
    generate_ros2_control_params_temp_file,
    load_yaml,
)


def _servo_server(context):
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
        controllers_name="fake_controllers",
        robot_ip="",
        report_type="normal",
        dof="7",
        robot_type="xarm",
        prefix="",
        hw_ns="xarm",
        limited="false",
        effort_control=False,
        velocity_control=False,
        ros2_control_plugin="uf_robot_hardware/UFRobotFakeSystemHardware",
        ros2_control_params=ros2_control_params,
        add_gripper="true",
    ).to_moveit_configs()
    robot_parameters = {}
    robot_parameters.update(moveit_config.robot_description)
    robot_parameters.update(moveit_config.robot_description_semantic)
    robot_parameters.update(moveit_config.robot_description_kinematics)
    robot_parameters.update(moveit_config.joint_limits)
    robot_parameters.update(moveit_config.planning_pipelines)

    servo = load_yaml("xarm_moveit_servo", "config/xarm_moveit_servo_config.yaml")
    servo["move_group_name"] = "xarm7"
    servo["planning_frame"] = "link_base"
    servo["ee_frame_name"] = "link_eef"
    servo["robot_link_command_frame"] = "link_base"
    servo["command_out_topic"] = "/xarm7_traj_controller/joint_trajectory"
    servo["cartesian_command_in_topic"] = "~/delta_twist_cmds"
    servo["incoming_command_timeout"] = 0.2
    servo["check_collisions"] = True

    return [ComposableNodeContainer(
        name="bookshelf_simple_moveit_servo_container",
        namespace="/",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[ComposableNode(
            package="moveit_servo",
            plugin="moveit_servo::ServoNode",
            name="servo_server",
            parameters=[{"moveit_servo": servo}, robot_parameters],
        )],
        output="screen",
    )]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_servo_server)])
