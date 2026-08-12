"""Automatically record one bookshelf experiment without commanding hardware."""

from datetime import datetime
from pathlib import Path
import shutil

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


CORE_TOPICS = [
    "/joint_states",
    "/tf",
    "/tf_static",
    "/camera/color/camera_info",
    "/camera/aligned_depth_to_color/camera_info",
    "/camera/extrinsics/depth_to_color",
    "/slot_detector/confidence",
    "/slot_detector/slot_width",
    "/slot_detector/slot_pose",
    "/slot_detector/left_boundary",
    "/slot_detector/right_boundary",
    "/bookshelf_environment/slot_markers",
    "/bookshelf_environment/static_slot_pose",
    "/bookshelf_environment/live_slot_pose_base",
    "/bookshelf_environment/static_slot_check_passed",
    "/bookshelf_environment/static_slot_check_status",
    "/display_planned_path",
    "/monitored_planning_scene",
    "/bookshelf_scene/status",
    "/bookshelf_scene/ready",
    "/bookshelf_scene/mode",
    "/bookshelf_policy/observation_valid",
    "/bookshelf_policy/observation_12d",
    "/bookshelf_policy/raw_metrics",
    "/bookshelf_policy/adapter_debug",
    "/bookshelf_policy/slot_pose_base",
    "/bookshelf_policy/book_pose_base",
    "/bookshelf_shadow/policy_activation_ready",
    "/bookshelf_shadow/policy_activation_debug",
    "/bookshelf_shadow/inference_valid",
    "/bookshelf_shadow/policy_debug",
    "/bookshelf_shadow/residual_policy_action",
    "/bookshelf_shadow/nominal_delta",
    "/bookshelf_shadow/residual_delta",
    "/bookshelf_shadow/final_delta",
    "/bookshelf_guarded/plan_valid",
    "/bookshelf_guarded/plan_report",
    "/bookshelf_guarded/target_policy_tool",
    "/bookshelf_guarded/target_tcp",
    "/bookshelf_guarded/planned_trajectory",
    "/execute_trajectory/_action/goal",
    "/execute_trajectory/_action/result",
    "/execute_trajectory/_action/feedback",
    "/execute_trajectory/_action/status",
    "/xarm7_traj_controller/follow_joint_trajectory/_action/goal",
    "/xarm7_traj_controller/follow_joint_trajectory/_action/result",
    "/xarm7_traj_controller/follow_joint_trajectory/_action/feedback",
    "/xarm7_traj_controller/follow_joint_trajectory/_action/status",
]

CAMERA_TOPICS = [
    "/camera/color/image_raw/compressed",
    "/camera/aligned_depth_to_color/image_raw/compressedDepth",
]


def _as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context):
    root = Path(LaunchConfiguration("output_root").perform(context)).expanduser()
    trial_name = LaunchConfiguration("trial_name").perform(context).strip()
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in trial_name
    ).strip("_") or "unnamed"
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / f"{stamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    minimum_free_gb = float(
        LaunchConfiguration("minimum_free_space_gb").perform(context)
    )
    free_gb = shutil.disk_usage(run_dir).free / (1024.0**3)
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB free at {root}; "
            f"{minimum_free_gb:.2f} GiB is required."
        )

    record_camera = _as_bool(
        LaunchConfiguration("record_camera").perform(context)
    )
    topics = CORE_TOPICS + (CAMERA_TOPICS if record_camera else [])
    bag_dir = run_dir / "rosbag"
    recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--include-hidden-topics",
            "--compression-mode",
            "file",
            "--compression-format",
            "zstd",
            "-o",
            str(bag_dir),
            *topics,
        ],
        output="screen",
        sigterm_timeout="60",
        sigkill_timeout="120",
    )
    logger = Node(
        package="bookshelf_shadow_ros",
        executable="experiment_logger",
        name="bookshelf_experiment_logger",
        output="screen",
        parameters=[
            {
                "run_dir": str(run_dir),
                "trial_name": trial_name,
                "repository_path": LaunchConfiguration("repository_path"),
                "policy_bundle_path": LaunchConfiguration("policy_bundle"),
                "activation_envelope_path": LaunchConfiguration(
                    "activation_envelope"
                ),
                "camera_recording": ParameterValue(
                    LaunchConfiguration("record_camera"), value_type=bool
                ),
            }
        ],
    )
    return [
        LogInfo(msg=f"AUTOMATIC EXPERIMENT LOG DIRECTORY: {run_dir}"),
        LogInfo(
            msg=(
                f"Compressed camera recording: {record_camera}; "
                f"free space before recording: {free_gb:.2f} GiB"
            )
        ),
        logger,
        recorder,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("trial_name", default_value="bookshelf_trial"),
            DeclareLaunchArgument(
                "output_root",
                default_value="/tmp/bookshelf_experiments",
            ),
            DeclareLaunchArgument("repository_path", default_value=""),
            DeclareLaunchArgument("policy_bundle", default_value=""),
            DeclareLaunchArgument("activation_envelope", default_value=""),
            DeclareLaunchArgument("record_camera", default_value="true"),
            DeclareLaunchArgument("minimum_free_space_gb", default_value="5.0"),
            LogInfo(
                msg=(
                    "Starting automatic experiment logging only. This launch "
                    "contains no policy executor or robot-command node."
                )
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
