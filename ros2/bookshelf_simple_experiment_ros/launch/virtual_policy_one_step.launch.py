"""Software-only xArm7 + MoveIt + one lightweight policy step."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_simple_experiment_ros")
    execute = LaunchConfiguration("execute")
    show_rviz = LaunchConfiguration("show_rviz")
    rviz_config = PathJoinSubstitution([
        package_share, "rviz", "simple_policy_one_step.rviz"
    ])
    fake_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("xarm_moveit_config"),
            "launch",
            "_robot_moveit_fake.launch.py",
        ])),
        launch_arguments={
            "dof": "7",
            "robot_type": "xarm",
            "limited": "false",
            "add_gripper": "true",
            "no_gui_ctrl": "false",
            "show_rviz": show_rviz,
            "rviz_config": rviz_config,
        }.items(),
    )
    servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            package_share, "launch", "simple_xarm7_servo_server.launch.py"
        ])),
        condition=IfCondition(execute),
    )
    initializer = Node(
        package="bookshelf_simple_experiment_ros",
        executable="fake_policy_start",
        name="fake_policy_start",
        output="screen",
    )
    policy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            package_share, "launch", "simple_policy_one_step.launch.py"
        ])),
        launch_arguments={
            "approved_config": LaunchConfiguration("approved_config"),
            "actor_path": LaunchConfiguration("actor_path"),
            "execute": execute,
            "rollout": LaunchConfiguration("rollout"),
            "max_steps": LaunchConfiguration("max_steps"),
            "command_scale": LaunchConfiguration("command_scale"),
            "record_bag": LaunchConfiguration("record_bag"),
            "visualization_hold_s": LaunchConfiguration("visualization_hold_s"),
            "translation_tolerance_m": LaunchConfiguration("translation_tolerance_m"),
            "rotation_tolerance_rad": LaunchConfiguration("rotation_tolerance_rad"),
            "run_dir": LaunchConfiguration("run_dir"),
        }.items(),
    )
    start_policy_after_initialization = RegisterEventHandler(OnProcessExit(
        target_action=initializer,
        on_exit=[policy],
    ))

    return LaunchDescription([
        DeclareLaunchArgument(
            "approved_config",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "BookshelfFiles",
                "experiment_configs",
                "stationary_approved_53e7fe80d56d_20260819_142355",
                "trial_static_slot.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "actor_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "BookshelfFiles",
                "trained_models",
                "bookshelf_residual_2026-07-08_shadow_actor.npz",
            ]),
        ),
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("rollout", default_value="false"),
        DeclareLaunchArgument("max_steps", default_value="150"),
        DeclareLaunchArgument("command_scale", default_value="0.10"),
        DeclareLaunchArgument("record_bag", default_value="false"),
        DeclareLaunchArgument("visualization_hold_s", default_value="60.0"),
        DeclareLaunchArgument("translation_tolerance_m", default_value="0.0005"),
        DeclareLaunchArgument(
            "rotation_tolerance_rad", default_value="0.004363323129985824"
        ),
        DeclareLaunchArgument("run_dir", default_value=""),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        LogInfo(msg=[
            "SOFTWARE ONLY: official xArm7 fake hardware; policy controller; execute=",
            execute,
            "; rollout=",
            LaunchConfiguration("rollout"),
        ]),
        fake_moveit,
        servo,
        initializer,
        start_policy_after_initialization,
    ])
