"""Plan to the unapproved spine-mount candidate without execution."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "launch",
            "bookshelf_scene_manager.launch.py",
        ]
    )
    candidate_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "spine_mount_book_calibration_candidate.yaml",
        ]
    )
    planner_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "config",
            "calibrated_preinsert_plan_only.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_config",
                description="Approved trial_static_slot.yaml parameter file.",
            ),
            DeclareLaunchArgument(
                "candidate_config",
                default_value=candidate_config,
                description="Unapproved spine-mount calibration candidate YAML.",
            ),
            DeclareLaunchArgument(
                "scene_config",
                description=(
                    "Reviewed physical shelf, table, and held-book scene YAML."
                ),
            ),
            DeclareLaunchArgument(
                "planner_config",
                default_value=planner_config,
                description="Fail-closed global pre-insertion planner parameters.",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="/tmp/bookshelf_spine_mount_candidate_plan",
                description="Directory for candidate target and plan-only reports.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("tf_max_age_s", default_value="0.50"),
            DeclareLaunchArgument(
                "maximum_preserved_book_orientation_error_deg",
                default_value="15.0",
            ),
            LogInfo(
                msg=(
                    "Starting UNAPPROVED SPINE-MOUNT CANDIDATE PLAN-ONLY "
                    "bridge. It may request a MoveIt path for inspection, but "
                    "creates no execution, controller, gripper, or robot-command "
                    "client. Execution remains unauthorized."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(scene_launch),
                launch_arguments={
                    "scene_config": LaunchConfiguration("scene_config"),
                }.items(),
            ),
            Node(
                package="bookshelf_shadow_ros",
                executable="calibrated_preinsert_target",
                name="calibrated_preinsert_target",
                output="screen",
                parameters=[
                    LaunchConfiguration("target_config"),
                    LaunchConfiguration("candidate_config"),
                    {
                        "output_dir": LaunchConfiguration("output_dir"),
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                        "tf_max_age_s": ParameterValue(
                            LaunchConfiguration("tf_max_age_s"), value_type=float
                        ),
                        "target_orientation_mode": "preserve_current_tcp",
                        "maximum_preserved_book_orientation_error_deg": ParameterValue(
                            LaunchConfiguration(
                                "maximum_preserved_book_orientation_error_deg"
                            ),
                            value_type=float,
                        ),
                    },
                ],
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="calibrated_preinsert_plan_only",
                name="calibrated_preinsert_plan_only",
                output="screen",
                parameters=[
                    LaunchConfiguration("planner_config"),
                    {"output_dir": LaunchConfiguration("output_dir")},
                ],
            ),
        ]
    )
