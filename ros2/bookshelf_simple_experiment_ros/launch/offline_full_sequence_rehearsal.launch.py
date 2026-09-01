"""Full rosbag + fake-hardware rehearsal of the operator-controlled episode."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("bookshelf_simple_experiment_ros")
    preview = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            share, "launch", "offline_rosbag_preinsert_visualization.launch.py"
        ])),
        launch_arguments={
            "bag_path": LaunchConfiguration("bag_path"),
            "preview_rviz": LaunchConfiguration("show_rviz"),
            "allow_execution": "true",
            "require_slot_acceptance": "true",
            "auto_plan": "false",
            "scan_joint_state_path": LaunchConfiguration("scan_joint_state_path"),
            "loading_joint_state_path": LaunchConfiguration("loading_joint_state_path"),
        }.items(),
    )
    servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            share, "launch", "simple_xarm7_servo_server.launch.py"
        ]))
    )
    policy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            share, "launch", "simple_policy_one_step.launch.py"
        ])),
        launch_arguments={
            "approved_config": LaunchConfiguration("approved_config"),
            "actor_path": LaunchConfiguration("actor_path"),
            "execute": "true",
            "shadow_full_sequence": "false",
            "wait_for_start": "true",
            "rollout": "true",
            "max_steps": LaunchConfiguration("max_steps"),
            "command_scale": "1.0",
            "record_bag": "false",
            "visualization_hold_s": "0.0",
            "translation_tolerance_m": "0.00005",
            "rotation_tolerance_rad": "0.0001",
            "gripper_action": (
                "/xarm_gripper_traj_controller/follow_joint_trajectory"
            ),
            "gripper_action_type": "follow_joint_trajectory",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"), "BookshelfFiles", "real_rgbd",
                "slot_view_01_complete", "slot_view_01",
            ]),
        ),
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
        DeclareLaunchArgument("max_steps", default_value="150"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        LogInfo(msg=(
            "ALIENWARE FULL REHEARSAL: recorded RGB-D, real detector, official "
            "fake xArm7/MoveIt/gripper, reviewed preinsert and PPO sequence. "
            "No physical hardware is started."
        )),
        preview,
        servo,
        policy,
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="operator_actions",
            name="bookshelf_operator_actions",
            output="screen",
            parameters=[{
                "allow_execution": True,
                "shadow_full_sequence": False,
                "joint_move_duration_s": 2.0,
                "gripper_action": (
                    "/xarm_gripper_traj_controller/follow_joint_trajectory"
                ),
                "gripper_action_type": "follow_joint_trajectory",
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="real_experiment_operator",
            name="bookshelf_full_rehearsal_operator",
            output="screen",
            emulate_tty=True,
        ),
    ])
