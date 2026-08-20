#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start one xArm hardware/MoveIt stack plus its Servo component."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the single hardware launch used for planning and servoing."""

    # Declare launch configurations (can be overridden at runtime)
    robot_ip = LaunchConfiguration('robot_ip')
    add_gripper = LaunchConfiguration('add_gripper')
    limited = LaunchConfiguration('limited')
    dof = LaunchConfiguration('dof')
    robot_type = LaunchConfiguration('robot_type')
    prefix = LaunchConfiguration('prefix')
    hw_ns = LaunchConfiguration('hw_ns')
    show_rviz = LaunchConfiguration('show_rviz')

    # Include xarm7 MoveIt “real‐move” launch (from xarm_moveit_config)
    moveit_realmove = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('xarm_planner'),
                'launch',
                'xarm7_planner_realmove.launch.py'
            ])
        ),
        launch_arguments={
            'robot_ip':    robot_ip,
            'add_gripper': add_gripper,
            'limited':     limited,
            'show_rviz':   show_rviz,
        }.items(),
    )

    # Add only the Servo component. The real-move launch above already owns
    # the robot driver, controller manager, MoveIt and planner.
    servo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('bookshelf_policy_ros'),
                'launch',
                'xarm7_moveit_servo_server.launch.py'
            ])
        ),
        launch_arguments={
            'robot_ip':    robot_ip,
            'dof':         dof,
            'robot_type':  robot_type,
            'prefix':      prefix,
            'hw_ns':       hw_ns,
            'add_gripper': add_gripper,
            'limited':     limited,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.1.209'),
        DeclareLaunchArgument('add_gripper', default_value='true'),
        DeclareLaunchArgument('limited', default_value='false'),
        DeclareLaunchArgument('dof', default_value='7'),
        DeclareLaunchArgument('robot_type', default_value='xarm'),
        DeclareLaunchArgument('prefix', default_value=''),
        DeclareLaunchArgument('hw_ns', default_value='xarm'),
        DeclareLaunchArgument(
            'show_rviz',
            default_value='false',
            description='Start MoveIt RViz. Keep false for SSH/headless bringup.',
        ),
        moveit_realmove,
        servo_server,
    ])
