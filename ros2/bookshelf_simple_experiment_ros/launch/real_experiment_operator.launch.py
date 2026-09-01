"""One-command physical bookshelf bringup with reviewed operator controls."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    physical = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("bookshelf_policy_ros"),
            "launch",
            "physical_hardware_bringup.launch.py",
        ])),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            # The preinsert workflow below owns the sole RViz instance.
            "show_rviz": "false",
        }.items(),
    )
    preinsert = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("bookshelf_simple_experiment_ros"),
            "launch",
            "real_preinsert_workflow.launch.py",
        ])),
        launch_arguments={
            "show_rviz": LaunchConfiguration("show_rviz"),
            "frozen_slot_output": LaunchConfiguration("frozen_slot_output"),
            "allow_execution": LaunchConfiguration("allow_execution"),
            "shadow_full_sequence": LaunchConfiguration("shadow_full_sequence"),
            "scan_joint_state_path": LaunchConfiguration("scan_joint_state_path"),
            "loading_joint_state_path": LaunchConfiguration("loading_joint_state_path"),
        }.items(),
    )
    policy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("bookshelf_simple_experiment_ros"),
            "launch",
            "simple_policy_one_step.launch.py",
        ])),
        launch_arguments={
            "approved_config": LaunchConfiguration("approved_config"),
            "actor_path": LaunchConfiguration("actor_path"),
            "execute": LaunchConfiguration("allow_execution"),
            "shadow_full_sequence": LaunchConfiguration("shadow_full_sequence"),
            "wait_for_start": "true",
            "rollout": "true",
            "max_steps": LaunchConfiguration("max_steps"),
            "record_bag": "false",
            "visualization_hold_s": "0.0",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.209"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        DeclareLaunchArgument("allow_execution", default_value="false"),
        DeclareLaunchArgument("shadow_full_sequence", default_value="false"),
        DeclareLaunchArgument("max_steps", default_value="150"),
        DeclareLaunchArgument(
            "scan_joint_state_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"), "BookshelfFiles", "experiment_configs",
                "operator_joint_poses", "scan_joint_state.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "loading_joint_state_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"), "BookshelfFiles", "experiment_configs",
                "operator_joint_poses", "loading_joint_state.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "approved_config",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"), "BookshelfFiles", "experiment_configs",
                "stationary_approved_53e7fe80d56d_20260819_142355",
                "trial_static_slot.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "actor_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"), "BookshelfFiles", "trained_models",
                "bookshelf_residual_2026-07-08_shadow_actor.npz",
            ]),
        ),
        DeclareLaunchArgument(
            "frozen_slot_output",
            default_value="/tmp/bookshelf_simple_frozen_slot.yaml",
        ),
        LogInfo(msg=(
            "BOOKSHELF REAL EXPERIMENT: one physical hardware/MoveIt/Servo stack, "
            "one preinsert workflow, one RViz, and the reviewed operator console. "
            "No motion, gripper, policy, or execution goal is sent automatically."
        )),
        physical,
        preinsert,
        policy,
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="operator_actions",
            name="bookshelf_operator_actions",
            output="screen",
            parameters=[{
                "allow_execution": ParameterValue(
                    LaunchConfiguration("allow_execution"), value_type=bool
                ),
                "shadow_full_sequence": ParameterValue(
                    LaunchConfiguration("shadow_full_sequence"), value_type=bool
                ),
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="real_experiment_operator",
            name="bookshelf_real_experiment_operator",
            output="screen",
            emulate_tty=True,
        ),
    ])
