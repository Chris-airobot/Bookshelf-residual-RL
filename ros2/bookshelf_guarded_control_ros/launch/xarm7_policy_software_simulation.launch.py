"""Run the bookshelf policy against UFACTORY's official fake xArm7 stack."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import yaml

from bookshelf_guarded_control_ros.policy_tool_control_math import (
    make_transform,
    matrix_to_quaternion_xyzw,
)
from bookshelf_guarded_control_ros.grasp_alignment import (
    derive_simulation_grasp_setback,
)
from bookshelf_guarded_control_ros.rehearsal_configuration import (
    validate_shadow_rehearsal_assets,
)


def _parameters(document: dict, node_name: str) -> dict:
    try:
        value = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"approved configuration is missing {node_name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid parameters for {node_name}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simulation_actions(context):
    approved_config = Path(
        LaunchConfiguration("approved_config").perform(context)
    ).expanduser().resolve()
    policy_bundle = Path(
        LaunchConfiguration("policy_bundle").perform(context)
    ).expanduser().resolve()
    activation_envelope = Path(
        LaunchConfiguration("activation_envelope").perform(context)
    ).expanduser().resolve()
    trial_name = LaunchConfiguration("trial_name").perform(context).strip()
    safe_trial_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in trial_name
    ).strip("_") or "xarm7_policy_software_simulation"
    output_root = Path(
        LaunchConfiguration("monitor_output_root").perform(context)
    ).expanduser().resolve()
    result = validate_shadow_rehearsal_assets(
        approved_config, policy_bundle, activation_envelope
    )
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / f"{stamp}_{safe_trial_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_document = yaml.safe_load(approved_config.read_text(encoding="utf-8"))
    document = source_document
    nominal_adapter = _parameters(source_document, "policy_observation_adapter")
    grasp_setback_m = float(
        LaunchConfiguration("physical_grasp_setback_m").perform(context)
    )
    runtime_config = approved_config
    grasp_report = None
    if grasp_setback_m > 0.0:
        document, grasp_report = derive_simulation_grasp_setback(
            document, grasp_setback_m
        )
        runtime_config = run_dir / "simulation_grasp_config.yaml"
        runtime_config.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        source_provenance = approved_config.with_suffix(".provenance.json")
        provenance = json.loads(source_provenance.read_text(encoding="utf-8"))
        provenance.update(
            {
                "trial_config_sha256": _sha256_file(runtime_config),
                "simulation_only_derived_config": True,
                "source_approved_config": str(approved_config),
                "source_approved_config_sha256": _sha256_file(approved_config),
                "physical_grasp_setback_m": grasp_setback_m,
                "hardware_commanded": False,
                "execution_authorized": False,
            }
        )
        runtime_config.with_suffix(".provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    adapter = _parameters(document, "policy_observation_adapter")
    scene = _parameters(document, "bookshelf_scene_manager")
    static_check = _parameters(document, "static_slot_environment_check")

    marker_mount_path = (
        Path(get_package_share_directory("bookshelf_shadow_ros"))
        / "config"
        / "real_book_aruco0_mount.yaml"
    )
    marker_mount = yaml.safe_load(marker_mount_path.read_text(encoding="utf-8"))
    marker_center = marker_mount["marker_center_in_book_m"]
    marker_quaternion = matrix_to_quaternion_xyzw(
        marker_mount["rotation_book_marker"]
    ).tolist()

    eef_book_translation = adapter["eef_book_translation_xyz"]
    eef_book_quaternion = adapter["eef_book_quaternion_xyzw"]
    transform_base_slot = make_transform(
        adapter["configured_static_slot_translation_xyz"],
        adapter["configured_static_slot_quaternion_xyzw"],
    )
    retreat_direction = (-transform_base_slot[:3, 0]).tolist()
    task_sequence = Node(
        package="bookshelf_guarded_control_ros",
        executable="fake_release_retreat_sequence",
        name="fake_release_retreat_sequence",
        output="screen",
        parameters=[
            {
                "simulation_only": True,
                "base_frame": "link_base",
                "eef_frame": "link_eef",
                "tcp_frame": "link_tcp",
                "book_frame": "target_book_center",
                "book_size_xyz": scene["held_book_size_xyz"],
                "eef_book_translation_xyz": eef_book_translation,
                "eef_book_quaternion_xyzw": eef_book_quaternion,
                "nominal_eef_book_translation_xyz": nominal_adapter[
                    "eef_book_translation_xyz"
                ],
                "nominal_eef_book_quaternion_xyzw": nominal_adapter[
                    "eef_book_quaternion_xyzw"
                ],
                "initial_grasp_alignment_enabled": grasp_setback_m > 0.0,
                "physical_release_guard_enabled": ParameterValue(
                    LaunchConfiguration("physical_release_guard_enabled"),
                    value_type=bool,
                ),
                "slot_translation_base_xyz": adapter[
                    "configured_static_slot_translation_xyz"
                ],
                "slot_quaternion_base_xyzw": adapter[
                    "configured_static_slot_quaternion_xyzw"
                ],
                "physical_release_tcp_x_limit_m": ParameterValue(
                    LaunchConfiguration("physical_release_tcp_x_limit_m"),
                    value_type=float,
                ),
                "minimum_book_leading_penetration_m": ParameterValue(
                    LaunchConfiguration("minimum_book_leading_penetration_m"),
                    value_type=float,
                ),
                "start_servo_service": "/servo_server/start_servo",
                "retreat_direction_base_xyz": retreat_direction,
                "retreat_distance_m": LaunchConfiguration(
                    "scripted_retreat_distance_m"
                ),
                "retreat_speed_m_s": LaunchConfiguration(
                    "scripted_retreat_speed_m_s"
                ),
                "push_book_distance_m": LaunchConfiguration(
                    "policy_push_book_distance_m"
                ),
                "push_to_target_trailing_depth_enabled": True,
                "push_target_trailing_depth_m": ParameterValue(
                    LaunchConfiguration("policy_push_target_trailing_depth_m"),
                    value_type=float,
                ),
                "push_timeout_s": LaunchConfiguration("policy_push_timeout_s"),
                "pretarget_ready_topic": "/bookshelf_sim/pretarget_ready",
                "policy_control_enable_topic": (
                    "/bookshelf_sim/policy_control_enabled"
                ),
                "mode_topic": "/bookshelf_policy/mode",
                "policy_debug_topic": "/bookshelf_shadow/policy_debug",
                "twist_command_topic": "/servo_server/delta_twist_cmds",
                "gripper_action": (
                    "/xarm_gripper_traj_controller/follow_joint_trajectory"
                ),
                # Official xArm fake-gripper convention.
                "gripper_open_position": 0.0,
                "gripper_closed_position": 0.85,
            }
        ],
    )

    pretarget_initializer = Node(
        package="bookshelf_guarded_control_ros",
        executable="fake_pretarget_initializer",
        name="fake_pretarget_initializer",
        output="screen",
        parameters=[
            {
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "joint_positions": [
                    1.2342693425054612,
                    1.5322427671441177,
                    4.904658882462919,
                    1.302429752118059,
                    3.302595179623167,
                    0.6839448116011184,
                    4.4791192150828865,
                ],
                "trajectory_action": (
                    "/xarm7_traj_controller/follow_joint_trajectory"
                ),
                "move_duration_s": 0.5,
                "control_enable_topic": "/bookshelf_sim/pretarget_ready",
            }
        ],
    )

    scene_visualizer = Node(
        package="bookshelf_shadow_ros",
        executable="offline_scene_visualizer",
        name="bookshelf_xarm_fake_scene",
        output="screen",
        parameters=[
            {
                "visualization_only": True,
                "scene_configuration_confirmed": True,
                "publish_joint_states": False,
                "show_coarse_bookshelf": False,
                "base_frame": "link_base",
                "tcp_frame": "target_book_center",
                "target_book_frame": "target_book_center",
                "slot_translation_xyz": adapter[
                    "configured_static_slot_translation_xyz"
                ],
                "slot_quaternion_xyzw": adapter[
                    "configured_static_slot_quaternion_xyzw"
                ],
                "slot_width_m": float(
                    adapter["configured_static_slot_width_m"]
                ),
                "slot_visual_height_m": float(
                    static_check["visual_slot_height_m"]
                ),
                "shelf_size_xyz": scene["shelf_box_size_xyz"],
                "shelf_center_offset_slot_xyz": scene[
                    "shelf_box_center_offset_slot_xyz"
                ],
                "shelf_bottom_height_base_m": float(
                    scene["shelf_bottom_height_base_m"]
                ),
                "table_size_xyz": scene["table_box_size_xyz"],
                "table_center_base_xyz": scene["table_box_center_base_xyz"],
                "table_quaternion_base_xyzw": scene[
                    "table_box_quaternion_base_xyzw"
                ],
                "held_book_size_xyz": scene["held_book_size_xyz"],
                "held_book_center_tcp_xyz": [0.0, 0.0, 0.0],
                "held_book_quaternion_tcp_xyzw": [0.0, 0.0, 0.0, 1.0],
                "marker_enabled": True,
                "marker_center_book_xyz": [
                    float(marker_center["x"]),
                    float(marker_center["y"]),
                    float(marker_center["z"]),
                ],
                "marker_quaternion_book_xyzw": marker_quaternion,
                "marker_size_m": float(marker_mount["marker_black_size_m"]),
                "marker_thickness_m": float(
                    marker_mount["cardboard_thickness_m"]
                ),
                "marker_topic": "/bookshelf_sim/markers",
            }
        ],
    )

    monitor_logger = Node(
        package="bookshelf_shadow_ros",
        executable="experiment_logger",
        name="bookshelf_simulation_monitor",
        output="screen",
        parameters=[
            {
                "run_dir": str(run_dir),
                "trial_name": trial_name,
                "repository_path": LaunchConfiguration("repository_path"),
                "policy_bundle_path": str(policy_bundle),
                "activation_envelope_path": str(activation_envelope),
                "camera_recording": False,
                "raw_replay_inputs_recorded": False,
                "capture_condition": "official_xarm_fake_hardware",
            }
        ],
    )
    release_geometry_capture = Node(
        package="bookshelf_shadow_ros",
        executable="ros_release_geometry",
        name="bookshelf_sim_release_geometry",
        output="screen",
        parameters=[
            {
                "approved_config_path": str(runtime_config),
                "output_path": str(run_dir / "xarm_release_geometry.json"),
                "capture_condition": "task_release",
            }
        ],
    )

    guarded_share = FindPackageShare("bookshelf_guarded_control_ros")
    policy_share = FindPackageShare("bookshelf_policy_ros")
    fake_moveit_launch = PathJoinSubstitution(
        [
            FindPackageShare("xarm_moveit_config"),
            "launch",
            "_robot_moveit_fake.launch.py",
        ]
    )
    servo_launch = PathJoinSubstitution(
        [policy_share, "launch", "xarm7_moveit_servo_server.launch.py"]
    )
    deployment_launch = PathJoinSubstitution(
        [guarded_share, "launch", "physical_policy_deployment.launch.py"]
    )
    rviz_config = PathJoinSubstitution(
        [guarded_share, "rviz", "xarm7_policy_software_simulation.rviz"]
    )
    return [
        LogInfo(
            msg=(
                "Starting OFFICIAL xArm7 fake hardware + MoveIt + Servo + "
                f"bookshelf policy candidate {result['candidate_id']}. The fake "
                "gripper executes the trained release/retreat/push sequence; no robot "
                "IP, xArm API, camera, or physical hardware is used."
            )
        ),
        LogInfo(msg=f"SIMULATION MONITOR DIRECTORY: {run_dir}"),
        LogInfo(
            msg=(
                "RELEASE GEOMETRY OUTPUT: "
                f"{run_dir / 'xarm_release_geometry.json'}"
            )
        ),
        LogInfo(
            msg=(
                "SIMULATION GRASP ALIGNMENT: "
                f"{json.dumps(grasp_report, sort_keys=True)}"
                if grasp_report is not None
                else "SIMULATION GRASP ALIGNMENT: original approved grasp"
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(fake_moveit_launch),
            launch_arguments={
                "dof": "7",
                "robot_type": "xarm",
                "limited": "false",
                "add_gripper": "true",
                "no_gui_ctrl": "false",
                "show_rviz": LaunchConfiguration("show_rviz"),
                "rviz_config": rviz_config,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(servo_launch),
            launch_arguments={
                "robot_ip": "",
                "dof": "7",
                "robot_type": "xarm",
                "limited": "false",
                "add_gripper": "true",
                "use_fake_hardware": "true",
            }.items(),
        ),
        pretarget_initializer,
        task_sequence,
        scene_visualizer,
        monitor_logger,
        release_geometry_capture,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(deployment_launch),
            launch_arguments={
                "trial_name": LaunchConfiguration("trial_name"),
                "approved_config": str(runtime_config),
                "repository_path": LaunchConfiguration("repository_path"),
                "policy_bundle": str(policy_bundle),
                "activation_envelope": str(activation_envelope),
                "enable_logging": "false",
                "enable_policy_audit": "false",
                "operation": LaunchConfiguration("operation"),
                "maximum_total_translation_m": LaunchConfiguration(
                    "maximum_total_translation_m"
                ),
                "base_frame": "link_base",
                "eef_frame": "link_eef",
                "tcp_frame": "link_tcp",
                "target_book_frame": "target_book_center",
                "joint_states_topic": "/joint_states",
                "start_servo_service": "/servo_server/start_servo",
                "servo_already_started": (
                    "true" if grasp_setback_m > 0.0 else "false"
                ),
                "twist_command_topic": "/servo_server/delta_twist_cmds",
                "command_target_is_hardware": "false",
                "enforce_translation_budget": "false",
                "require_control_enable": "true",
                "control_enable_topic": (
                    "/bookshelf_sim/policy_control_enabled"
                ),
                "yield_when_control_disabled": "true",
                "message_max_age_s": "1.0",
                "tf_max_age_s": "1.0",
            }.items(),
        ),
    ]


def generate_launch_description():
    """Build the official fake-xArm policy simulation launch."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("approved_config"),
            DeclareLaunchArgument("policy_bundle"),
            DeclareLaunchArgument("activation_envelope"),
            DeclareLaunchArgument(
                "trial_name", default_value="xarm7_policy_software_simulation"
            ),
            DeclareLaunchArgument(
                "repository_path", default_value="/home/chris/Chris/bookshelf-unified"
            ),
            DeclareLaunchArgument(
                "monitor_output_root",
                default_value=(
                    "/home/chris/BookshelfFiles/evaluation_results/"
                    "xarm7_policy_software_simulation"
                ),
            ),
            DeclareLaunchArgument(
                "operation",
                default_value="control",
                description=(
                    "calculate or control; calculate cannot move fake hardware."
                ),
            ),
            DeclareLaunchArgument(
                "maximum_total_translation_m",
                default_value="0.30",
                description="Required positive bound when operation=control.",
            ),
            DeclareLaunchArgument(
                "scripted_retreat_distance_m",
                default_value="0.09",
                description=(
                    "Scripted retreat distance after the policy requests release."
                ),
            ),
            DeclareLaunchArgument(
                "scripted_retreat_speed_m_s",
                default_value="0.05",
                description="Fake-hardware retreat speed after opening the gripper.",
            ),
            DeclareLaunchArgument(
                "policy_push_book_distance_m",
                default_value="0.03",
                description=(
                    "Fake book travel after the closed gripper recontacts it in PUSH mode."
                ),
            ),
            DeclareLaunchArgument(
                "policy_push_timeout_s",
                default_value="90.0",
                description="Maximum time allowed for the policy-controlled push stage.",
            ),
            DeclareLaunchArgument(
                "policy_push_target_trailing_depth_m",
                default_value="-0.012",
                description=(
                    "Task success target for the book trailing edge relative to "
                    "the shelf mouth."
                ),
            ),
            DeclareLaunchArgument(
                "physical_grasp_setback_m",
                default_value="0.0",
                description=(
                    "Simulation-only book setback along its local insertion axis. "
                    "The book-relative virtual policy tool is preserved exactly."
                ),
            ),
            DeclareLaunchArgument(
                "physical_release_guard_enabled",
                default_value="true",
                description=(
                    "Simulation-only release gate based on the physical xArm TCP "
                    "position at the shelf mouth."
                ),
            ),
            DeclareLaunchArgument(
                "physical_release_tcp_x_limit_m",
                default_value="-0.006",
                description=(
                    "Maximum xArm TCP slot-frame X before release. Positive X is "
                    "into the shelf."
                ),
            ),
            DeclareLaunchArgument(
                "minimum_book_leading_penetration_m",
                default_value="0.08",
                description=(
                    "Required leading-edge penetration before either policy or "
                    "physical-boundary release is accepted."
                ),
            ),
            DeclareLaunchArgument("show_rviz", default_value="true"),
            OpaqueFunction(function=_simulation_actions),
        ]
    )
