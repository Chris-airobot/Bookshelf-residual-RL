"""One lightweight INSERT policy step; assumes xArm and Servo are already up."""

from datetime import datetime
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


BAG_TOPICS = [
    "/joint_states",
    "/tf",
    "/tf_static",
    "/camera/color/image_raw",
    "/camera/color/camera_info",
    "/bookshelf_simple/policy/raw_observation",
    "/bookshelf_simple/policy/policy_observation",
    "/bookshelf_simple/policy/residual_action",
    "/bookshelf_simple/policy/nominal_delta",
    "/bookshelf_simple/policy/scaled_residual_delta",
    "/bookshelf_simple/policy/final_delta",
    "/bookshelf_simple/policy/scaled_command",
    "/bookshelf_simple/policy/target_tcp",
    "/bookshelf_simple/policy/markers",
    "/bookshelf_simple/policy/current_book_pose",
    "/bookshelf_simple/policy/current_tcp_pose",
    "/bookshelf_simple/policy/current_policy_tool_pose",
    "/bookshelf_simple/policy/target_policy_tool_pose",
    "/bookshelf_simple/policy/release_requested",
    "/bookshelf_simple/policy/status",
    "/servo_server/delta_twist_cmds",
    "/servo_server/status",
]


def _launch_setup(context):
    requested = LaunchConfiguration("run_dir").perform(context).strip()
    if requested:
        run_dir = Path(os.path.expanduser(requested)).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            Path.home()
            / "BookshelfFiles"
            / "experiment_logs"
            / f"simple_policy_{timestamp}"
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    config = PathJoinSubstitution([
        FindPackageShare("bookshelf_simple_experiment_ros"),
        "config",
        "simple_policy_control.yaml",
    ])
    controller = Node(
        package="bookshelf_simple_experiment_ros",
        executable="simple_policy_control",
        name="simple_policy_control",
        output="screen",
        parameters=[config, {
            "approved_config": LaunchConfiguration("approved_config"),
            "actor_path": LaunchConfiguration("actor_path"),
            "run_dir": str(run_dir),
            "execute": ParameterValue(LaunchConfiguration("execute"), value_type=bool),
            "rollout": ParameterValue(LaunchConfiguration("rollout"), value_type=bool),
            "max_steps": ParameterValue(LaunchConfiguration("max_steps"), value_type=int),
            "command_scale": ParameterValue(
                LaunchConfiguration("command_scale"), value_type=float
            ),
            "visualization_hold_s": ParameterValue(
                LaunchConfiguration("visualization_hold_s"), value_type=float
            ),
            "translation_tolerance_m": ParameterValue(
                LaunchConfiguration("translation_tolerance_m"), value_type=float
            ),
            "rotation_tolerance_rad": ParameterValue(
                LaunchConfiguration("rotation_tolerance_rad"), value_type=float
            ),
        }],
    )
    bag = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", str(run_dir / "rosbag"), *BAG_TOPICS],
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_bag")),
    )
    shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=controller,
            on_exit=[Shutdown(reason="policy controller finished")],
        )
    )
    return [bag, controller, shutdown]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "approved_config",
            default_value=(
                "/home/riot/BookshelfFiles/experiment_configs/"
                "stationary_approved_53e7fe80d56d_20260819_142355/"
                "trial_static_slot.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "actor_path",
            default_value=(
                "/home/riot/BookshelfFiles/trained_models/"
                "bookshelf_residual_2026-07-08_shadow_actor.npz"
            ),
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
        OpaqueFunction(function=_launch_setup),
    ])
