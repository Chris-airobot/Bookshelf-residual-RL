#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch configurations (can be overridden at runtime)
    robot_ip    = LaunchConfiguration('robot_ip', default='192.168.1.209')
    add_gripper = LaunchConfiguration('add_gripper', default='true')
    limited     = LaunchConfiguration('limited', default='false')
    dof         = LaunchConfiguration('dof', default='7')
    robot_type  = LaunchConfiguration('robot_type', default='xarm')
    prefix      = LaunchConfiguration('prefix', default='')
    hw_ns       = LaunchConfiguration('hw_ns', default='xarm')
    show_rviz   = LaunchConfiguration('show_rviz')

    # ---- Add xarm_api driver launch ----
    xarm_api_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('xarm_api'),
                'launch',
                'xarm7_driver.launch.py'
            ])
        ),
        launch_arguments={
            'robot_ip': robot_ip,
            'report_type': 'normal',  # default, or set as needed
        }.items(),
    )

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

    # Include the planner‐wrapper (_robot_planner.launch.py)
    robot_planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('xarm_planner'),
                'launch',
                '_robot_planner.launch.py'
            ])
        ),
        launch_arguments={
            'dof':         dof,
            'robot_type':  robot_type,
            'prefix':      prefix,
            'hw_ns':       hw_ns,
            'add_gripper': add_gripper,
            'velocity_control': 'true',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'show_rviz',
            default_value='false',
            description='Start MoveIt RViz. Keep false for SSH/headless bringup.',
        ),
        xarm_api_driver,
        moveit_realmove,
        robot_planner,
    ])
