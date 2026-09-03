#!/usr/bin/env python3
"""One INSERT policy step or a closed-loop INSERT rollout from live xArm state."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, TwistStamped
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int8, String
from std_srvs.srv import Trigger
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .policy_observation_math import (
    ObservationScales,
    compute_policy_observation,
    invert_transform,
)
from .policy_tool_math import (
    OneShotExecutionGuard,
    bounded_error_twist,
    compute_policy_tool_target,
    eef_target_from_tcp_target,
    make_transform,
    matrix_to_quaternion_xyzw,
    validated_transform,
)
from .post_insert_math import (
    NominalPushConfig,
    compute_push_nominal_delta,
    oriented_box_contact_gap,
    retreat_progress,
    simulated_book_push_distance,
)
from .residual_policy_math import (
    NumpyActorBundle,
    POLICY_ACTION_SIZE,
    POLICY_OBSERVATION_SIZE,
    ResidualMotionConfig,
    combine_motion_delta,
    compute_policy_nominal_delta,
    release_requested_for_mode,
    scale_residual_action,
)
from .execution_gate import hardware_commands_allowed
from .operator_action_node import (
    GRIPPER_COMMAND,
    GRIPPER_TRAJECTORY,
    make_gripper_goal,
)
from .per_grasp_calibration import (
    select_eef_book_transform,
    semantic_held_gripper_observation,
)


@dataclass(frozen=True)
class ReviewedPolicyGeometry:
    transform_base_slot: np.ndarray
    transform_eef_book: np.ndarray
    transform_eef_tcp: np.ndarray
    transform_eef_policy_tool: np.ndarray
    transform_tcp_policy_tool: np.ndarray
    book_size: tuple[float, float, float]
    slot_depth_m: float
    slot_width_m: float
    observation_scales: ObservationScales
    gripper_open_joint_position: float
    gripper_closed_joint_position: float


def _parameters(document, node_name: str) -> dict:
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing {node_name}.ros__parameters") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"{node_name}.ros__parameters must be a mapping")
    return parameters


def load_reviewed_policy_geometry(path) -> ReviewedPolicyGeometry:
    """Load only the reviewed geometry needed by the isolated controller."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"approved saved-slot configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    slot = _parameters(document, "static_slot_environment_check")
    target = _parameters(document, "calibrated_preinsert_target")
    adapter = _parameters(document, "policy_observation_adapter")
    scene = _parameters(document, "bookshelf_scene_manager")

    transform_base_slot = make_transform(
        slot["static_slot_translation_xyz"],
        slot["static_slot_quaternion_xyzw"],
    )
    transform_eef_book = make_transform(
        target["eef_book_translation_xyz"],
        target["eef_book_quaternion_xyzw"],
    )
    transform_tcp_book = make_transform(
        scene["held_book_center_tcp_xyz"],
        scene["held_book_quaternion_tcp_xyzw"],
    )
    transform_eef_policy_tool = make_transform(
        target["eef_policy_tool_translation_xyz"],
        target["eef_policy_tool_quaternion_xyzw"],
    )
    transform_eef_tcp = transform_eef_book @ invert_transform(transform_tcp_book)
    transform_tcp_policy_tool = (
        invert_transform(transform_eef_tcp) @ transform_eef_policy_tool
    )
    for transform in (
        transform_base_slot,
        transform_eef_book,
        transform_eef_tcp,
        transform_eef_policy_tool,
        transform_tcp_policy_tool,
    ):
        validated_transform(transform)

    return ReviewedPolicyGeometry(
        transform_base_slot=transform_base_slot,
        transform_eef_book=transform_eef_book,
        transform_eef_tcp=transform_eef_tcp,
        transform_eef_policy_tool=transform_eef_policy_tool,
        transform_tcp_policy_tool=transform_tcp_policy_tool,
        book_size=tuple(float(value) for value in target["book_size_xyz"]),
        slot_depth_m=float(target["slot_depth_m"]),
        slot_width_m=float(slot["static_slot_width_m"]),
        observation_scales=ObservationScales(
            rear_to_mouth=float(adapter["rear_to_mouth_obs_scale"]),
            front_to_back=float(adapter["front_to_back_obs_scale"]),
            lateral=float(adapter["lat_err_obs_scale"]),
            vertical=float(adapter["z_err_obs_scale"]),
            yaw=math.radians(float(adapter["yaw_err_obs_scale_deg"])),
            tool_to_book=float(adapter["tool_to_book_obs_scale"]),
        ),
        gripper_open_joint_position=float(adapter["gripper_open_joint_position"]),
        gripper_closed_joint_position=float(adapter["gripper_closed_joint_position"]),
    )


def _transform_message_to_matrix(message: TransformStamped) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _transform_record(transform) -> dict:
    transform = validated_transform(transform)
    return {
        "translation_xyz": transform[:3, 3].tolist(),
        "quaternion_xyzw": matrix_to_quaternion_xyzw(transform[:3, :3]).tolist(),
        "matrix": transform.tolist(),
    }


def named_joint_positions(message: JointState, joint_names) -> list[float]:
    """Return finite joint positions in an explicit, stable name order."""

    positions_by_name = dict(zip(message.name, message.position))
    missing = [name for name in joint_names if name not in positions_by_name]
    if missing:
        raise ValueError(f"joint state is missing {missing}")
    positions = [float(positions_by_name[name]) for name in joint_names]
    if not np.all(np.isfinite(positions)):
        raise ValueError("joint positions must be finite")
    return positions


def visualization_hold_deadline_ns(now_ns: int, hold_s: float) -> int | None:
    """Return a hold deadline, or None for an indefinite visualization hold."""

    hold_s = float(hold_s)
    if not math.isfinite(hold_s) or hold_s < 0.0:
        raise ValueError("visualization_hold_s must be finite and non-negative")
    return None if hold_s == 0.0 else int(now_ns) + int(hold_s * 1.0e9)


def _pose_from_transform(transform) -> Pose:
    transform = validated_transform(transform)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(
        float, transform[:3, 3]
    )
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
        float, quaternion
    )
    return pose


def build_policy_visualization_markers(
    base_frame,
    transform_base_slot,
    transform_base_book,
    transform_base_tcp,
    transform_base_policy_tool,
    transform_base_tcp_target,
    transform_base_policy_tool_target,
    *,
    slot_depth_m,
    slot_width_m,
    book_size,
    stamp=None,
) -> MarkerArray:
    """Build compact, color-coded snapshot markers without changing geometry."""

    specifications = [
        (
            "saved_slot",
            Marker.CUBE,
            transform_base_slot @ make_transform([0.5 * slot_depth_m, 0.0, 0.0]),
            [slot_depth_m, slot_width_m, 0.25],
            [0.1, 0.8, 0.9, 0.20],
        ),
        ("current_book", Marker.CUBE, transform_base_book, book_size, [0.15, 0.45, 1.0, 0.75]),
        ("current_tcp", Marker.SPHERE, transform_base_tcp, [0.020] * 3, [0.2, 1.0, 0.2, 1.0]),
        ("current_policy_tool", Marker.SPHERE, transform_base_policy_tool, [0.025] * 3, [1.0, 0.65, 0.0, 1.0]),
        ("target_tcp", Marker.SPHERE, transform_base_tcp_target, [0.028] * 3, [1.0, 0.1, 0.1, 1.0]),
        ("target_policy_tool", Marker.SPHERE, transform_base_policy_tool_target, [0.032] * 3, [1.0, 0.1, 1.0, 1.0]),
    ]
    markers = []
    for marker_id, (name, marker_type, transform, scale, color) in enumerate(specifications):
        marker = Marker()
        marker.header.frame_id = str(base_frame)
        if stamp is not None:
            marker.header.stamp = stamp
        marker.ns = "bookshelf_simple_policy"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose = _pose_from_transform(transform)
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = map(float, color)
        marker.text = name
        markers.append(marker)
    return MarkerArray(markers=markers)


class SimplePolicyControlNode(Node):
    """Calculate fixed targets one at a time, optionally rolling out closed loop."""

    def __init__(self):
        super().__init__("simple_policy_control")
        self._declare_parameters()
        self.shadow_full_sequence = bool(
            self.get_parameter("shadow_full_sequence").value
        )
        self.requested_execution = bool(self.get_parameter("execute").value)
        self.execute = hardware_commands_allowed(
            self.requested_execution, self.shadow_full_sequence
        )
        self.rollout = bool(self.get_parameter("rollout").value)
        self.wait_for_start = bool(self.get_parameter("wait_for_start").value)
        self.started = not self.wait_for_start
        self.max_steps = int(self.get_parameter("max_steps").value)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.gripper_action_type = str(
            self.get_parameter("gripper_action_type").value
        )
        if self.gripper_action_type not in (GRIPPER_COMMAND, GRIPPER_TRAJECTORY):
            raise ValueError(
                f"unsupported gripper_action_type: {self.gripper_action_type!r}"
            )
        push_x_uncertainty_m = float(
            self.get_parameter("push_x_uncertainty_m").value
        )
        if not math.isfinite(push_x_uncertainty_m) or push_x_uncertainty_m < 0.0:
            raise ValueError("push_x_uncertainty_m must be finite and non-negative")
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.geometry = load_reviewed_policy_geometry(
            self.get_parameter("approved_config").value
        )
        self.per_grasp_eef_book = None
        self.per_grasp_diagnostics = None
        self.actor = NumpyActorBundle(self.get_parameter("actor_path").value)
        self.execution_guard = OneShotExecutionGuard()
        self.latest_joint_state = None
        self.latest_joint_state_ns = None
        self.latest_servo_status = None
        self.phase = "waiting_for_live_state" if self.started else "waiting_for_start"
        self.record = None
        self.target = None
        self.target_eef = None
        self.execution_start_ns = None
        self.finish_after_ns = None
        self.visualization_hold_deadline_ns = None
        self.nonzero_command_count = 0
        self.continuous_servo_command_count = 0
        self.first_servo_command_ns = None
        self.last_servo_command_ns = None
        self.first_policy_update_ns = None
        self.last_policy_update_ns = None
        self.rollout_start_ns = None
        self.observed_servo_statuses = set()
        self.maximum_rear_to_mouth_m = None
        self.step_index = 0
        self.total_steps = 0
        self.servo_started = False
        self.rollout_final_logged = False
        self.released_book_transform = None
        self.push_book_transform = None
        self.retreat_direction = -self.geometry.transform_base_slot[:3, 0].copy()
        self.retreat_direction /= np.linalg.norm(self.retreat_direction)
        self.phase_start_ns = None
        self.retreat_start_xyz = None
        self.retreat_distance_m = 0.0
        self.push_start_xyz = None
        self.push_book_origin = None
        self.push_geometric_contact_distance_m = None
        self.push_contact_distance_m = None
        self.book_contact_gap_m = None
        self.push_distance_m = 0.0
        self.book_push_distance_m = 0.0
        self.push_policy_index = 0
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_retry_start_ns = None
        self.gripper_next_attempt_ns = 0
        self.shadow_sequence = []
        self.shadow_next_transition_ns = None

        run_dir_value = str(self.get_parameter("run_dir").value).strip()
        if not run_dir_value:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir_value = f"~/BookshelfFiles/experiment_logs/simple_policy_{timestamp}"
        self.run_dir = Path(run_dir_value).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "policy_step.jsonl"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            20,
        )
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            PoseStamped,
            "/bookshelf_simple/per_grasp_eef_book",
            self._per_grasp_callback,
            latched,
        )
        self.create_subscription(
            String,
            "/bookshelf_simple/per_grasp_status",
            self._per_grasp_status_callback,
            latched,
        )
        self.debug_publishers = {
            "raw_observation": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/raw_observation", 10),
            "policy_observation": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/policy_observation", 10),
            "residual_action": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/residual_action", 10),
            "nominal_delta": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/nominal_delta", 10),
            "scaled_residual_delta": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/scaled_residual_delta", 10),
            "final_delta": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/final_delta", 10),
            "scaled_command": self.create_publisher(Float32MultiArray, "/bookshelf_simple/policy/scaled_command", 10),
        }
        self.target_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_simple/policy/target_tcp", latched
        )
        self.visual_pose_publishers = {
            "slot": self.create_publisher(PoseStamped, "/bookshelf_simple/policy/slot_pose", latched),
            "current_book": self.create_publisher(PoseStamped, "/bookshelf_simple/policy/current_book_pose", latched),
            "current_tcp": self.create_publisher(PoseStamped, "/bookshelf_simple/policy/current_tcp_pose", latched),
            "current_policy_tool": self.create_publisher(PoseStamped, "/bookshelf_simple/policy/current_policy_tool_pose", latched),
            "target_policy_tool": self.create_publisher(PoseStamped, "/bookshelf_simple/policy/target_policy_tool_pose", latched),
        }
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/bookshelf_simple/policy/markers", latched
        )
        self.release_publisher = self.create_publisher(
            Bool, "/bookshelf_simple/policy/release_requested", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/bookshelf_simple/policy/status", latched
        )
        self.create_service(
            Trigger, "/bookshelf_simple/start_policy", self._start_policy_callback
        )

        self.servo_start_client = None
        self.twist_publisher = None
        self.gripper_client = None
        if self.execute:
            self.servo_start_client = self.create_client(
                Trigger, str(self.get_parameter("start_servo_service").value)
            )
            self.twist_publisher = self.create_publisher(
                TwistStamped,
                str(self.get_parameter("twist_command_topic").value),
                10,
            )
            self.create_subscription(
                Int8,
                str(self.get_parameter("servo_status_topic").value),
                self._servo_status_callback,
                10,
            )
            self.gripper_client = ActionClient(
                self,
                GripperCommand
                if self.gripper_action_type == GRIPPER_COMMAND
                else FollowJointTrajectory,
                str(self.get_parameter("gripper_action").value),
            )

        self.create_timer(
            1.0 / float(self.get_parameter("control_rate_hz").value),
            self._timer_callback,
        )
        if self.rollout:
            self.create_timer(
                1.0 / float(self.get_parameter("policy_rate_hz").value),
                self._continuous_policy_tick,
            )
        self._publish_status(self.phase)
        self.get_logger().warning(
            f"INSERT {'ROLLOUT' if self.rollout else 'ONE POLICY STEP'}; "
            f"execute={self.execute}; max_steps={self.max_steps}; "
            "learned release starts the post-INSERT sequence; "
            f"run_dir={self.run_dir}"
        )

    def _declare_parameters(self):
        self.declare_parameter(
            "approved_config",
            "/home/riot/BookshelfFiles/experiment_configs/"
            "stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml",
        )
        self.declare_parameter(
            "actor_path",
            "/home/riot/BookshelfFiles/trained_models/"
            "bookshelf_residual_2026-07-08_shadow_actor.npz",
        )
        self.declare_parameter("run_dir", "")
        self.declare_parameter("execute", False)
        self.declare_parameter("shadow_full_sequence", False)
        self.declare_parameter("wait_for_start", False)
        self.declare_parameter("rollout", False)
        self.declare_parameter("max_steps", 150)
        self.declare_parameter("command_scale", 0.10)
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("gripper_joint_name", "drive_joint")
        self.declare_parameter(
            "gripper_action",
            "/xarm_gripper/gripper_action",
        )
        self.declare_parameter("gripper_action_type", GRIPPER_COMMAND)
        self.declare_parameter("gripper_open_position", 0.0)
        self.declare_parameter("gripper_closed_position", 0.85)
        self.declare_parameter("policy_held_gripper_open", 0.009838026859259968)
        self.declare_parameter("gripper_max_effort", 0.0)
        self.declare_parameter("gripper_move_duration_s", 0.6)
        self.declare_parameter("gripper_goal_retry_timeout_s", 15.0)
        self.declare_parameter("gripper_goal_retry_period_s", 0.25)
        self.declare_parameter(
            "expected_arm_joint_names",
            [f"joint{index}" for index in range(1, 8)],
        )
        self.declare_parameter("live_state_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)
        self.declare_parameter("start_servo_service", "/servo_server/start_servo")
        self.declare_parameter("twist_command_topic", "/servo_server/delta_twist_cmds")
        self.declare_parameter("servo_status_topic", "/servo_server/status")
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("policy_rate_hz", 20.0)
        self.declare_parameter("policy_command_duration_s", 0.20)
        self.declare_parameter("maximum_command_duration_s", 2.0)
        self.declare_parameter("maximum_linear_speed_m_s", 0.025)
        self.declare_parameter("maximum_angular_speed_rad_s", 0.10)
        self.declare_parameter("translation_tolerance_m", 0.0005)
        self.declare_parameter("rotation_tolerance_rad", math.radians(0.25))
        self.declare_parameter("post_command_settle_s", 0.20)
        self.declare_parameter("visualization_hold_s", 60.0)
        self.declare_parameter("retreat_distance_m", 0.09)
        self.declare_parameter("retreat_speed_m_s", 0.05)
        self.declare_parameter("retreat_timeout_s", 15.0)
        self.declare_parameter("push_book_distance_m", 0.03)
        self.declare_parameter("push_x_uncertainty_m", 0.005)
        self.declare_parameter("push_timeout_s", 90.0)
        self.declare_parameter("contact_tolerance_m", 0.001)

    def _now_ns(self):
        return self.get_clock().now().nanoseconds

    def _joint_state_callback(self, message):
        self.latest_joint_state = message
        self.latest_joint_state_ns = self._now_ns()

    def _per_grasp_callback(self, message):
        if not self.per_grasp_diagnostics or self.per_grasp_diagnostics.get(
            "source"
        ) != "per_grasp":
            return
        pose = message.pose
        try:
            self.per_grasp_eef_book = validated_transform(make_transform(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w],
            ))
        except ValueError as error:
            self.get_logger().warning(
                f"Ignoring invalid per-grasp EEF->book transform: {error}"
            )
            return
        self.get_logger().info("Using frozen per-grasp EEF->book transform")

    def _per_grasp_status_callback(self, message):
        try:
            self.per_grasp_diagnostics = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.per_grasp_diagnostics = {"source": "invalid_status"}
        if self.per_grasp_diagnostics.get("source") != "per_grasp":
            self.per_grasp_eef_book = None
            return
        try:
            self.per_grasp_eef_book = validated_transform(
                np.asarray(
                    self.per_grasp_diagnostics["transform_eef_book"],
                    dtype=np.float64,
                )
            )
        except (KeyError, TypeError, ValueError):
            self.per_grasp_eef_book = None
            self.get_logger().warning(
                "Per-grasp status did not contain a valid EEF->book transform"
            )

    def _active_eef_book_transform(self):
        return select_eef_book_transform(
            self.per_grasp_eef_book, self.geometry.transform_eef_book
        )

    def _policy_insert_gripper_open(self, measured_value=None):
        return semantic_held_gripper_observation(
            self._gripper_open() if measured_value is None else measured_value,
            self.get_parameter("policy_held_gripper_open").value,
        )

    def _servo_status_callback(self, message):
        self.latest_servo_status = int(message.data)

    def _lookup(self, source_frame):
        message = self.tf_buffer.lookup_transform(
            self.base_frame,
            source_frame,
            Time(),
            timeout=Duration(
                seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
            ),
        )
        return _transform_message_to_matrix(message)

    def _live_input_error(self):
        if self.latest_joint_state is None or self.latest_joint_state_ns is None:
            return "waiting for joint state"
        age = (self._now_ns() - self.latest_joint_state_ns) * 1.0e-9
        if age > float(self.get_parameter("live_state_max_age_s").value):
            return "joint state is stale"
        required = set(self.get_parameter("expected_arm_joint_names").value)
        required.add(str(self.get_parameter("gripper_joint_name").value))
        missing = sorted(required.difference(self.latest_joint_state.name))
        if missing:
            return f"joint state is missing {missing}"
        return None

    def _gripper_open(self):
        name = str(self.get_parameter("gripper_joint_name").value)
        index = self.latest_joint_state.name.index(name)
        position = float(self.latest_joint_state.position[index])
        opened = self.geometry.gripper_open_joint_position
        closed = self.geometry.gripper_closed_joint_position
        if not math.isfinite(position) or math.isclose(opened, closed):
            raise ValueError("gripper joint state/configuration is invalid")
        return float(np.clip((closed - position) / (closed - opened), 0.0, 1.0))

    def _arm_joint_positions(self):
        names = list(self.get_parameter("expected_arm_joint_names").value)
        return names, named_joint_positions(self.latest_joint_state, names)

    def _timer_callback(self):
        if not getattr(self, "started", True):
            return
        if self.phase == "shadow_sequence":
            self._shadow_sequence_tick()
        elif self.phase == "waiting_for_live_state":
            self._try_calculate()
        elif self.phase == "waiting_for_servo":
            self._try_start_servo()
        elif self.phase == "executing":
            self._servo_tick()
        elif self.phase == "continuous_rollout":
            self._continuous_servo_tick()
        elif self.phase == "opening_gripper":
            self._try_send_gripper_goal("open")
        elif self.phase == "retreat":
            self._retreat_tick()
        elif self.phase == "closing_empty_gripper":
            self._try_send_gripper_goal("close_empty")
        elif self.phase == "push":
            self._push_servo_tick()
        elif self.phase == "settling" and self._now_ns() >= self.finish_after_ns:
            self._complete_execution()
        elif (
            self.phase == "holding_visualization"
            and self.visualization_hold_deadline_ns is not None
            and self._now_ns() >= self.visualization_hold_deadline_ns
        ):
            self.phase = "finished_process"

    def _try_calculate(self):
        error = self._live_input_error()
        if error:
            return
        try:
            transform_base_eef = self._lookup(self.eef_frame)
            transform_base_tcp = self._lookup(self.tcp_frame)
        except Exception:
            # Joint states can arrive before this node's new TF buffer is primed.
            return
        try:
            transform_base_book = transform_base_eef @ self._active_eef_book_transform()
            transform_base_policy_tool = (
                transform_base_eef @ self.geometry.transform_eef_policy_tool
            )
            transform_slot_base = invert_transform(self.geometry.transform_base_slot)
            transform_slot_book = transform_slot_base @ transform_base_book
            measured_gripper_open = self._gripper_open()
            policy_gripper_open = self._policy_insert_gripper_open(
                measured_gripper_open
            )
            raw, observation = compute_policy_observation(
                transform_slot_book,
                transform_slot_base @ transform_base_policy_tool,
                book_size=self.geometry.book_size,
                slot_depth=self.geometry.slot_depth_m,
                mode_observation=0.0,
                gripper_open=policy_gripper_open,
                scales=self.geometry.observation_scales,
            )
            normalized, actor_mean, action = self.actor.predict(observation)
            nominal = compute_policy_nominal_delta(raw)
            residual = scale_residual_action(action)
            final_delta = combine_motion_delta(nominal, residual)
            limits = np.asarray(ResidualMotionConfig().final_limits, dtype=np.float32)
            if final_delta.shape != (5,) or not np.all(np.isfinite(final_delta)):
                raise ValueError("final motion must contain five finite values")
            if np.any(np.abs(final_delta) > limits + 1.0e-9):
                raise ValueError("final motion exceeds the verified clipped limits")
            target = compute_policy_tool_target(
                self.geometry.transform_base_slot,
                transform_base_tcp,
                self.geometry.transform_tcp_policy_tool,
                final_delta,
                command_scale=float(self.get_parameter("command_scale").value),
            )
            release_requested = release_requested_for_mode(
                action[5], raw[0], ResidualMotionConfig().release_threshold
            )
            joint_names, joint_positions = self._arm_joint_positions()
        except Exception as error:
            self._fail(f"policy calculation failed: {error}")
            return

        if raw.shape != (POLICY_OBSERVATION_SIZE,) or not np.all(np.isfinite(raw)):
            self._fail("raw observation is not finite 12-D")
            return
        if observation.shape != (POLICY_OBSERVATION_SIZE,) or not np.all(np.isfinite(observation)):
            self._fail("policy observation is not finite 12-D")
            return
        if action.shape != (POLICY_ACTION_SIZE,) or not np.all(np.isfinite(action)):
            self._fail("policy output is not finite 6-D")
            return

        self.target = target
        self.target_eef = eef_target_from_tcp_target(
            target.transform_base_tcp_target, self.geometry.transform_eef_tcp
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        self.record = {
            "step_index": self.step_index,
            "timestamp": timestamp,
            "T_base_slot": _transform_record(self.geometry.transform_base_slot),
            "T_base_eef": _transform_record(transform_base_eef),
            "T_base_tcp": _transform_record(transform_base_tcp),
            "T_base_book": _transform_record(transform_base_book),
            "T_slot_book": _transform_record(transform_slot_book),
            "eef_book_transform_source": (
                "per_grasp" if self.per_grasp_eef_book is not None else "fixed_fallback"
            ),
            "T_eef_book_used": _transform_record(self._active_eef_book_transform()),
            "per_grasp_diagnostics": self.per_grasp_diagnostics,
            "measured_gripper_open": measured_gripper_open,
            "policy_gripper_open": policy_gripper_open,
            "arm_joint_names": joint_names,
            "arm_joint_positions_rad": joint_positions,
            "raw_observation": raw.tolist(),
            "policy_observation": observation.tolist(),
            "vecnormalize_observation": normalized.tolist(),
            "ppo_actor_mean": actor_mean.tolist(),
            "ppo_residual_action": action.tolist(),
            "nominal_delta": nominal.tolist(),
            "scaled_residual_delta": residual.tolist(),
            "final_delta": final_delta.tolist(),
            "command_scale": float(self.get_parameter("command_scale").value),
            "resulting_scaled_command": target.scaled_delta.tolist(),
            "target_tcp_pose": _transform_record(target.transform_base_tcp_target),
            "release_requested": bool(release_requested),
            "release_executed": False,
            "execute": self.execute,
            "before_pose": _transform_record(transform_base_tcp),
            "after_pose": None,
            "servo_result": "not_requested" if not self.execute else "pending",
            "servo_status": self.latest_servo_status,
            "actor_path": str(self.actor.path),
            "actor_sha256": self.actor.sha256,
            "target_id": target.target_id,
        }
        self._publish_calculation(
            raw,
            observation,
            action,
            nominal,
            residual,
            final_delta,
            release_requested,
            transform_base_book,
            transform_base_tcp,
            transform_base_policy_tool,
        )
        self._write_record("calculated")
        self.total_steps = self.step_index + 1
        policy_update_ns = self._now_ns()
        if self.first_policy_update_ns is None:
            self.first_policy_update_ns = policy_update_ns
        self.last_policy_update_ns = policy_update_ns
        rear_to_mouth = float(raw[1])
        if (
            self.maximum_rear_to_mouth_m is None
            or rear_to_mouth > self.maximum_rear_to_mouth_m
        ):
            self.maximum_rear_to_mouth_m = rear_to_mouth
        if self.shadow_full_sequence:
            self.record["servo_result"] = "shadow_full_sequence_no_motion"
            self.record["shadow_full_sequence"] = True
            self._write_record("complete")
            self._start_shadow_sequence(transform_base_eef, transform_base_tcp)
            return
        if self._stop_rollout_for_release(
            release_requested, transform_base_eef, transform_base_tcp
        ):
            return
        if not self.execute:
            self.record["servo_result"] = "shadow_complete_no_motion"
            self._write_record("complete")
            if self.rollout:
                self._terminate_rollout(
                    "error",
                    transform_base_eef=transform_base_eef,
                    transform_base_tcp=transform_base_tcp,
                    error="rollout requires execute=true",
                )
            else:
                self._publish_status("complete", "execute=false; no motion interface created")
                self._begin_visualization_hold()
            return
        if self.rollout and self.servo_started:
            self.record["servo_result"] = "target_replaced_continuous"
            self.record["servo_status"] = self.latest_servo_status
            self._write_record("complete")
            self.step_index += 1
            self.phase = "continuous_rollout"
        elif self.servo_started:
            self._start_current_target()
        else:
            self.phase = "waiting_for_servo"
            self.servo_wait_start_ns = self._now_ns()
            self._publish_status("waiting_for_servo")

    def _stop_rollout_for_release(
        self, release_requested, transform_base_eef, transform_base_tcp
    ):
        if not self.rollout or not release_requested:
            return False
        if self.twist_publisher is not None:
            self._publish_twist(np.zeros(6))
        self.record["after_pose"] = _transform_record(transform_base_tcp)
        self.record["after_eef_pose"] = _transform_record(transform_base_eef)
        joint_names, joint_positions = self._arm_joint_positions()
        self.record["after_arm_joint_names"] = joint_names
        self.record["after_arm_joint_positions_rad"] = joint_positions
        self.record["after_T_slot_book"] = _transform_record(
            invert_transform(self.geometry.transform_base_slot)
            @ transform_base_eef
            @ self._active_eef_book_transform()
        )
        self.record["servo_result"] = "release_requested_no_motion"
        self.record["servo_nonzero_command_count"] = 0
        self._write_record("complete")
        self.released_book_transform = (
            transform_base_eef @ self._active_eef_book_transform()
        )
        self._log_phase_event(
            "release_requested",
            transform_base_eef,
            transform_base_tcp,
            release_action=float(self.record["ppo_residual_action"][5]),
            policy_step=int(self.record["step_index"]),
        )
        self.get_logger().warning("POLICY RELEASE REQUESTED")
        self._log_phase_event(
            "release_started", transform_base_eef, transform_base_tcp
        )
        self.phase = "opening_gripper"
        self.phase_start_ns = self._now_ns()
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_retry_start_ns = None
        self.gripper_next_attempt_ns = 0
        self._publish_status("release_started", "opening gripper")
        return True

    def _try_send_gripper_goal(self, kind):
        if self.gripper_goal_pending or self._now_ns() < self.gripper_next_attempt_ns:
            return
        if self.gripper_client is None:
            self._halt_and_fail("gripper action client is unavailable")
            return
        if self.gripper_retry_start_ns is None:
            self.gripper_retry_start_ns = self._now_ns()
        elapsed_s = (self._now_ns() - self.gripper_retry_start_ns) * 1.0e-9
        if elapsed_s > float(
            self.get_parameter("gripper_goal_retry_timeout_s").value
        ):
            self._halt_and_fail(f"gripper {kind} action remained unavailable")
            return
        if not self.gripper_client.server_is_ready():
            return
        position_parameter = (
            "gripper_open_position" if kind == "open" else "gripper_closed_position"
        )
        goal = make_gripper_goal(
            self.gripper_action_type,
            self.get_parameter(position_parameter).value,
            self.get_parameter("gripper_max_effort").value,
            self.get_parameter("gripper_move_duration_s").value,
        )
        self.gripper_goal_pending = True
        self.gripper_goal_kind = str(kind)
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response)

    def _gripper_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._halt_and_fail(f"gripper goal failed: {error}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self.gripper_goal_pending = False
            self.gripper_goal_kind = None
            retry_s = float(
                self.get_parameter("gripper_goal_retry_period_s").value
            )
            self.gripper_next_attempt_ns = self._now_ns() + int(
                max(retry_s, 0.02) * 1.0e9
            )
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_goal_result)

    def _gripper_goal_result(self, future):
        try:
            wrapped_result = future.result()
            status = int(wrapped_result.status)
        except Exception as error:
            self._halt_and_fail(f"gripper result failed: {error}")
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._halt_and_fail(f"gripper action failed with status {status}")
            return
        kind = self.gripper_goal_kind
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_retry_start_ns = None
        self.gripper_next_attempt_ns = 0
        try:
            transform_base_eef = self._lookup(self.eef_frame)
            transform_base_tcp = self._lookup(self.tcp_frame)
        except Exception as error:
            self._halt_and_fail(f"post-gripper TF failed: {error}")
            return
        if kind == "open":
            self._log_phase_event(
                "release_complete", transform_base_eef, transform_base_tcp
            )
            self.get_logger().warning("GRIPPER RELEASE COMPLETE")
            self._publish_status(
                "release_complete", "best-effort book scene update requested"
            )
            self.phase = "retreat"
            self.phase_start_ns = self._now_ns()
            self.retreat_start_xyz = transform_base_eef[:3, 3].copy()
            self.retreat_distance_m = 0.0
            self._log_phase_event(
                "retreat_started", transform_base_eef, transform_base_tcp
            )
            self.get_logger().warning("RETREAT STARTED")
            self._publish_status("retreat_started")
            return
        if kind == "close_empty":
            self._log_phase_event(
                "empty_gripper_closed", transform_base_eef, transform_base_tcp
            )
            self.phase = "push"
            self.phase_start_ns = self._now_ns()
            self.push_start_xyz = transform_base_eef[:3, 3].copy()
            self.push_book_origin = self.released_book_transform.copy()
            self.push_book_transform = self.released_book_transform.copy()
            self.push_geometric_contact_distance_m = None
            self.push_contact_distance_m = None
            self.book_contact_gap_m = None
            self.push_distance_m = 0.0
            self.book_push_distance_m = 0.0
            self.target = None
            self.target_eef = None
            self._log_phase_event(
                "push_started", transform_base_eef, transform_base_tcp
            )
            self._publish_status("push_started")

    def _post_servo_status_is_fatal(self):
        if self.latest_servo_status is not None:
            self.observed_servo_statuses.add(int(self.latest_servo_status))
        if self.latest_servo_status in (2, 4, 5):
            self._halt_and_fail(
                f"MoveIt Servo reported fatal status {self.latest_servo_status}"
            )
            return True
        return False

    def _retreat_tick(self):
        if self._post_servo_status_is_fatal():
            return
        try:
            transform_base_eef = self._lookup(self.eef_frame)
            transform_base_tcp = self._lookup(self.tcp_frame)
        except Exception:
            self._publish_twist(np.zeros(6))
            return
        self.retreat_distance_m = max(
            0.0,
            retreat_progress(
                self.retreat_start_xyz,
                transform_base_eef[:3, 3],
                self.retreat_direction,
            ),
        )
        requested = float(self.get_parameter("retreat_distance_m").value)
        self._publish_post_visualization(transform_base_eef, transform_base_tcp)
        if self.retreat_distance_m >= requested:
            self._publish_twist(np.zeros(6))
            self._log_phase_event(
                "retreat_complete",
                transform_base_eef,
                transform_base_tcp,
                retreat_distance_m=self.retreat_distance_m,
                requested_retreat_distance_m=requested,
            )
            self.phase = "closing_empty_gripper"
            self.phase_start_ns = self._now_ns()
            self.gripper_retry_start_ns = None
            self.gripper_next_attempt_ns = 0
            self._publish_status("retreat_complete", "closing gripper empty")
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("retreat_timeout_s").value):
            self._halt_and_fail("scripted retreat timed out")
            return
        speed = min(
            float(self.get_parameter("retreat_speed_m_s").value),
            max(requested - self.retreat_distance_m, 0.0) * 50.0,
        )
        twist = np.zeros(6, dtype=np.float64)
        twist[:3] = self.retreat_direction * speed
        self._publish_twist(twist)

    def _try_calculate_push(self):
        error = self._live_input_error()
        if error or self.push_book_transform is None:
            return
        try:
            transform_base_eef = self._lookup(self.eef_frame)
            transform_base_tcp = self._lookup(self.tcp_frame)
            transform_base_policy_tool = (
                transform_base_eef @ self.geometry.transform_eef_policy_tool
            )
            transform_slot_base = invert_transform(self.geometry.transform_base_slot)
            transform_slot_book = transform_slot_base @ self.push_book_transform
            raw, observation = compute_policy_observation(
                transform_slot_book,
                transform_slot_base @ transform_base_policy_tool,
                book_size=self.geometry.book_size,
                slot_depth=self.geometry.slot_depth_m,
                mode_observation=1.0,
                gripper_open=self._gripper_open(),
                scales=self.geometry.observation_scales,
            )
            normalized, actor_mean, action = self.actor.predict(observation)
            nominal = compute_push_nominal_delta(
                raw, NominalPushConfig(book_size=self.geometry.book_size)
            )
            residual = scale_residual_action(action)
            final_delta = combine_motion_delta(nominal, residual)
            self.target = compute_policy_tool_target(
                self.geometry.transform_base_slot,
                transform_base_tcp,
                self.geometry.transform_tcp_policy_tool,
                final_delta,
                command_scale=float(self.get_parameter("command_scale").value),
            )
            self.target_eef = eef_target_from_tcp_target(
                self.target.transform_base_tcp_target,
                self.geometry.transform_eef_tcp,
            )
        except Exception as error:
            self._halt_and_fail(f"PUSH policy calculation failed: {error}")
            return
        self._append_payload(
            {
                "event": "push_policy_step",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "push_step_index": self.push_policy_index,
                "T_base_eef": _transform_record(transform_base_eef),
                "T_base_tcp": _transform_record(transform_base_tcp),
                "T_base_book": _transform_record(self.push_book_transform),
                "T_base_book_release": _transform_record(
                    self.released_book_transform
                ),
                "T_slot_book": _transform_record(transform_slot_book),
                "raw_observation": raw.tolist(),
                "policy_observation": observation.tolist(),
                "vecnormalize_observation": normalized.tolist(),
                "ppo_actor_mean": actor_mean.tolist(),
                "ppo_residual_action": action.tolist(),
                "nominal_delta": nominal.tolist(),
                "scaled_residual_delta": residual.tolist(),
                "final_delta": final_delta.tolist(),
                "target_tcp_pose": _transform_record(
                    self.target.transform_base_tcp_target
                ),
                "servo_status": self.latest_servo_status,
            }
        )
        self.push_policy_index += 1
        self._publish_calculation(
            raw,
            observation,
            action,
            nominal,
            residual,
            final_delta,
            False,
            self.push_book_transform,
            transform_base_tcp,
            transform_base_policy_tool,
        )

    def _push_servo_tick(self):
        if self._post_servo_status_is_fatal():
            return
        try:
            transform_base_eef = self._lookup(self.eef_frame)
            transform_base_tcp = self._lookup(self.tcp_frame)
        except Exception:
            self._publish_twist(np.zeros(6))
            return
        insertion_direction = -self.retreat_direction
        self.push_distance_m = max(
            0.0,
            retreat_progress(
                self.push_start_xyz,
                transform_base_eef[:3, 3],
                insertion_direction,
            ),
        )
        self.book_contact_gap_m = oriented_box_contact_gap(
            transform_base_tcp[:3, 3],
            self.push_book_origin,
            self.geometry.book_size,
            insertion_direction,
        )
        tolerance = float(self.get_parameter("contact_tolerance_m").value)
        if self.push_contact_distance_m is None and self.book_contact_gap_m <= tolerance:
            self.push_geometric_contact_distance_m = max(
                0.0, self.push_distance_m + self.book_contact_gap_m
            )
            self.push_contact_distance_m = self.push_geometric_contact_distance_m
        requested = float(self.get_parameter("push_book_distance_m").value)
        if self.push_contact_distance_m is None:
            self.book_push_distance_m = 0.0
        else:
            self.book_push_distance_m = simulated_book_push_distance(
                self.push_distance_m,
                self.push_contact_distance_m,
                requested,
            )
        self.push_book_transform = self.push_book_origin.copy()
        self.push_book_transform[:3, 3] += (
            insertion_direction * self.book_push_distance_m
        )
        self._publish_post_visualization(transform_base_eef, transform_base_tcp)
        if self.book_push_distance_m >= requested:
            self._publish_twist(np.zeros(6))
            self._log_phase_event(
                "push_complete",
                transform_base_eef,
                transform_base_tcp,
                push_distance_m=self.push_distance_m,
                push_geometric_contact_distance_m=(
                    self.push_geometric_contact_distance_m
                ),
                push_contact_distance_m=self.push_contact_distance_m,
                push_x_uncertainty_m=float(
                    self.get_parameter("push_x_uncertainty_m").value
                ),
                contact_source="release_geometry_no_contact_sensor",
                book_push_distance_m=self.book_push_distance_m,
                requested_book_push_distance_m=requested,
            )
            self._complete_episode(transform_base_eef, transform_base_tcp)
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("push_timeout_s").value):
            self._halt_and_fail("policy PUSH timed out")
            return
        if self.target_eef is None:
            self._publish_twist(np.zeros(6))
            return
        try:
            twist = bounded_error_twist(
                transform_base_eef,
                self.target_eef,
                duration_s=float(self.get_parameter("policy_command_duration_s").value),
                maximum_linear_speed_m_s=float(self.get_parameter("maximum_linear_speed_m_s").value),
                maximum_angular_speed_rad_s=float(self.get_parameter("maximum_angular_speed_rad_s").value),
                translation_tolerance_m=float(self.get_parameter("translation_tolerance_m").value),
                rotation_tolerance_rad=float(self.get_parameter("rotation_tolerance_rad").value),
            )
        except Exception as error:
            self._halt_and_fail(f"PUSH Servo command calculation failed: {error}")
            return
        self._publish_twist(twist)

    def _publish_post_visualization(self, transform_base_eef, transform_base_tcp):
        if self.push_book_transform is None or self.target is None:
            return
        self._publish_visualization(
            self.push_book_transform,
            transform_base_tcp,
            transform_base_eef @ self.geometry.transform_eef_policy_tool,
        )

    def _log_phase_event(
        self, event, transform_base_eef=None, transform_base_tcp=None, **values
    ):
        payload = {
            "event": str(event),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
        }
        if transform_base_eef is not None:
            payload["T_base_eef"] = _transform_record(transform_base_eef)
        if transform_base_tcp is not None:
            payload["T_base_tcp"] = _transform_record(transform_base_tcp)
        book_transform = (
            self.push_book_transform
            if self.phase in ("push", "episode_complete")
            and self.push_book_transform is not None
            else self.released_book_transform
        )
        if book_transform is not None:
            payload["T_base_book"] = _transform_record(book_transform)
            payload["T_slot_book"] = _transform_record(
                invert_transform(self.geometry.transform_base_slot)
                @ book_transform
            )
        if self.released_book_transform is not None:
            payload["T_base_book_release"] = _transform_record(
                self.released_book_transform
            )
        payload.update(values)
        self._append_payload(payload)

    def _complete_episode(self, transform_base_eef, transform_base_tcp):
        self.phase = "episode_complete"
        self._log_phase_event(
            "episode_complete",
            transform_base_eef,
            transform_base_tcp,
            release_requested=True,
            retreat_distance_m=self.retreat_distance_m,
            book_push_distance_m=self.book_push_distance_m,
            push_distance_m=self.push_distance_m,
            policy_update_count=self.total_steps,
            push_policy_update_count=self.push_policy_index,
            observed_servo_statuses=sorted(self.observed_servo_statuses),
        )
        self._publish_status("episode_complete")
        self._begin_visualization_hold()

    def _try_start_servo(self):
        if (self._now_ns() - self.servo_wait_start_ns) * 1.0e-9 > 5.0:
            self._fail("MoveIt Servo interface unavailable")
            return
        if not self.servo_start_client.service_is_ready():
            return
        if self.twist_publisher.get_subscription_count() < 1:
            return
        if not self.execution_guard.try_consume():
            self._fail("one-shot execution allowance already consumed")
            return
        self.phase = "starting_servo"
        future = self.servo_start_client.call_async(Trigger.Request())
        future.add_done_callback(self._servo_start_response)

    def _servo_start_response(self, future):
        try:
            response = future.result()
        except Exception as error:
            self._fail(f"MoveIt Servo start failed: {error}")
            return
        if response is None or not response.success:
            self._fail(
                "MoveIt Servo start rejected: "
                + ("no response" if response is None else response.message)
            )
            return
        self.servo_started = True
        self._start_current_target()

    def _start_current_target(self):
        self.nonzero_command_count = 0
        self.execution_start_ns = self._now_ns()
        if self.rollout:
            if self.rollout_start_ns is None:
                self.rollout_start_ns = self.execution_start_ns
            self.record["servo_result"] = "target_active_continuous"
            self._write_record("complete")
            self.step_index += 1
            self.phase = "continuous_rollout"
            self._publish_status(
                "continuous_rollout", f"policy_update_index={self.step_index}"
            )
            return
        self.phase = "executing"
        self._publish_status("executing", f"step_index={self.step_index}")

    def _continuous_policy_tick(self):
        if self.phase == "push":
            self._try_calculate_push()
            return
        if self.phase != "continuous_rollout":
            return
        if self._continuous_rollout_timed_out():
            self._publish_twist(np.zeros(6))
            self._terminate_rollout(
                "max_steps",
                error=(
                    "continuous rollout exceeded the max_steps-derived "
                    f"{float(self.max_steps):.1f} s overall timeout"
                ),
            )
            return
        self._try_calculate()

    def _continuous_rollout_timed_out(self):
        if self.rollout_start_ns is None:
            return False
        elapsed_s = (self._now_ns() - self.rollout_start_ns) * 1.0e-9
        return elapsed_s >= float(self.max_steps)

    def _continuous_servo_tick(self):
        if self.latest_servo_status is not None:
            self.observed_servo_statuses.add(int(self.latest_servo_status))
        if self.latest_servo_status in (2, 4, 5):
            self._halt_and_fail(
                f"MoveIt Servo reported fatal status {self.latest_servo_status}"
            )
            return
        if self.target_eef is None:
            self._publish_twist(np.zeros(6))
            return
        try:
            current_eef = self._lookup(self.eef_frame)
            twist = bounded_error_twist(
                current_eef,
                self.target_eef,
                duration_s=float(self.get_parameter("policy_command_duration_s").value),
                maximum_linear_speed_m_s=float(self.get_parameter("maximum_linear_speed_m_s").value),
                maximum_angular_speed_rad_s=float(self.get_parameter("maximum_angular_speed_rad_s").value),
                translation_tolerance_m=float(self.get_parameter("translation_tolerance_m").value),
                rotation_tolerance_rad=float(self.get_parameter("rotation_tolerance_rad").value),
            )
        except Exception as error:
            self._halt_and_fail(f"Servo command calculation failed: {error}")
            return
        self._publish_twist(twist)
        command_ns = self._now_ns()
        if self.first_servo_command_ns is None:
            self.first_servo_command_ns = command_ns
        self.last_servo_command_ns = command_ns
        self.continuous_servo_command_count += 1

    def _servo_tick(self):
        try:
            current_eef = self._lookup(self.eef_frame)
            twist = bounded_error_twist(
                current_eef,
                self.target_eef,
                duration_s=float(self.get_parameter("policy_command_duration_s").value),
                maximum_linear_speed_m_s=float(self.get_parameter("maximum_linear_speed_m_s").value),
                maximum_angular_speed_rad_s=float(self.get_parameter("maximum_angular_speed_rad_s").value),
                translation_tolerance_m=float(self.get_parameter("translation_tolerance_m").value),
                rotation_tolerance_rad=float(self.get_parameter("rotation_tolerance_rad").value),
            )
        except Exception as error:
            self._halt_and_fail(f"Servo command calculation failed: {error}")
            return
        elapsed = (self._now_ns() - self.execution_start_ns) * 1.0e-9
        if not np.any(twist):
            self._publish_twist(np.zeros(6))
            self.record["servo_result"] = "target_reached"
            self._begin_settle()
            return
        if elapsed > float(self.get_parameter("maximum_command_duration_s").value):
            self._publish_twist(np.zeros(6))
            self.record["servo_result"] = "timeout_stopped"
            self._begin_settle()
            return
        self._publish_twist(twist)
        self.nonzero_command_count += 1

    def _publish_twist(self, values):
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = map(float, values[:3])
        message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = map(float, values[3:])
        self.twist_publisher.publish(message)

    def _begin_settle(self):
        self.phase = "settling"
        self.finish_after_ns = self._now_ns() + int(
            float(self.get_parameter("post_command_settle_s").value) * 1.0e9
        )

    def _complete_execution(self):
        try:
            after_eef = self._lookup(self.eef_frame)
            after_tcp = self._lookup(self.tcp_frame)
            self.record["after_pose"] = _transform_record(after_tcp)
            self.record["after_eef_pose"] = _transform_record(after_eef)
            joint_names, joint_positions = self._arm_joint_positions()
            self.record["after_arm_joint_names"] = joint_names
            self.record["after_arm_joint_positions_rad"] = joint_positions
            self.record["after_T_slot_book"] = _transform_record(
                invert_transform(self.geometry.transform_base_slot)
                @ after_eef
                @ self._active_eef_book_transform()
            )
            self._publish_visualization(
                after_eef @ self._active_eef_book_transform(),
                after_tcp,
                after_eef @ self.geometry.transform_eef_policy_tool,
            )
        except Exception as error:
            self.record["after_pose_error"] = str(error)
            if self.rollout:
                self.record["servo_status"] = self.latest_servo_status
                self.record["servo_nonzero_command_count"] = self.nonzero_command_count
                self._write_record("complete")
                self._terminate_rollout("error", error=f"post-step TF failed: {error}")
                return
        self.record["servo_status"] = self.latest_servo_status
        self.record["servo_nonzero_command_count"] = self.nonzero_command_count
        self._write_record("complete")
        if not self.rollout:
            self._publish_status("complete", self.record["servo_result"])
            self._begin_visualization_hold()
            return
        if self.record["servo_result"] != "target_reached":
            self._terminate_rollout(
                "error",
                transform_base_eef=after_eef,
                transform_base_tcp=after_tcp,
                error=f"Servo step ended with {self.record['servo_result']}",
            )
            return
        if self.total_steps >= self.max_steps:
            self._terminate_rollout(
                "max_steps",
                transform_base_eef=after_eef,
                transform_base_tcp=after_tcp,
            )
            return
        self.step_index += 1
        self.record = None
        self.target = None
        self.target_eef = None
        self.phase = "waiting_for_live_state"
        self._publish_status("waiting_for_live_state", f"step_index={self.step_index}")

    def _halt_and_fail(self, reason):
        if self.twist_publisher is not None:
            self._publish_twist(np.zeros(6))
        self._fail(reason)

    def _fail(self, reason):
        if self.record is None:
            self.record = {
                "step_index": self.step_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execute": self.execute,
            }
        self.record["servo_result"] = f"failed: {reason}"
        self.record["servo_status"] = self.latest_servo_status
        self._write_record("failed")
        self._publish_status("failed", reason)
        if self.rollout:
            self._terminate_rollout("error", error=reason)
            self.get_logger().error(reason)
            return
        self.phase = "failed"
        self.get_logger().error(reason)

    def _terminate_rollout(
        self,
        reason,
        *,
        transform_base_eef=None,
        transform_base_tcp=None,
        error=None,
    ):
        if self.rollout_final_logged:
            return
        self.rollout_final_logged = True
        if transform_base_eef is None or transform_base_tcp is None:
            try:
                transform_base_eef = self._lookup(self.eef_frame)
                transform_base_tcp = self._lookup(self.tcp_frame)
            except Exception:
                transform_base_eef = None
                transform_base_tcp = None
        payload = {
            "event": "rollout_complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_steps": self.total_steps,
            "termination_reason": str(reason),
            "release_requested": bool(
                self.record is not None and self.record.get("release_requested", False)
            ),
            "T_base_slot": _transform_record(self.geometry.transform_base_slot),
            "final_pose": (
                _transform_record(transform_base_tcp)
                if transform_base_tcp is not None
                else None
            ),
            "final_eef_pose": (
                _transform_record(transform_base_eef)
                if transform_base_eef is not None
                else None
            ),
            "final_book_pose": (
                _transform_record(self.push_book_transform)
                if self.push_book_transform is not None
                else (
                    _transform_record(self.released_book_transform)
                    if self.released_book_transform is not None
                    else (
                        _transform_record(
                            transform_base_eef @ self._active_eef_book_transform()
                        )
                        if transform_base_eef is not None
                        else None
                    )
                )
            ),
            "release_book_pose": (
                _transform_record(self.released_book_transform)
                if self.released_book_transform is not None
                else None
            ),
            "error": None if error is None else str(error),
            "policy_update_count": self.total_steps,
            "servo_command_count": self.continuous_servo_command_count,
            "observed_servo_statuses": sorted(self.observed_servo_statuses),
            "maximum_rear_to_mouth_m": self.maximum_rear_to_mouth_m,
        }
        if (
            self.first_policy_update_ns is not None
            and self.last_policy_update_ns is not None
            and self.total_steps > 1
        ):
            duration_s = (
                self.last_policy_update_ns - self.first_policy_update_ns
            ) * 1.0e-9
            payload["measured_policy_rate_hz"] = (
                (self.total_steps - 1) / duration_s if duration_s > 0.0 else None
            )
        if (
            self.first_servo_command_ns is not None
            and self.last_servo_command_ns is not None
            and self.continuous_servo_command_count > 1
        ):
            duration_s = (
                self.last_servo_command_ns - self.first_servo_command_ns
            ) * 1.0e-9
            payload["measured_servo_command_rate_hz"] = (
                (self.continuous_servo_command_count - 1) / duration_s
                if duration_s > 0.0
                else None
            )
        if self.latest_joint_state is not None:
            try:
                joint_names, joint_positions = self._arm_joint_positions()
                payload["final_arm_joint_names"] = joint_names
                payload["final_arm_joint_positions_rad"] = joint_positions
            except ValueError:
                pass
        if self.push_book_transform is not None:
            payload["final_book_pose_slot"] = _transform_record(
                invert_transform(self.geometry.transform_base_slot)
                @ self.push_book_transform
            )
        elif self.released_book_transform is not None:
            payload["final_book_pose_slot"] = _transform_record(
                invert_transform(self.geometry.transform_base_slot)
                @ self.released_book_transform
            )
        elif transform_base_eef is not None:
            payload["final_book_pose_slot"] = _transform_record(
                invert_transform(self.geometry.transform_base_slot)
                @ transform_base_eef
                @ self._active_eef_book_transform()
            )
        self._append_payload(payload)
        status_phase = "failed" if reason == "error" else "rollout_complete"
        self._publish_status(status_phase, error if error is not None else reason)
        self._begin_visualization_hold()

    def _publish_calculation(
        self,
        raw,
        observation,
        action,
        nominal,
        residual,
        final_delta,
        release_requested,
        transform_base_book,
        transform_base_tcp,
        transform_base_policy_tool,
    ):
        values = {
            "raw_observation": raw,
            "policy_observation": observation,
            "residual_action": action,
            "nominal_delta": nominal,
            "scaled_residual_delta": residual,
            "final_delta": final_delta,
            "scaled_command": self.target.scaled_delta,
        }
        for name, value in values.items():
            self.debug_publishers[name].publish(
                Float32MultiArray(data=np.asarray(value).tolist())
            )
        self.release_publisher.publish(Bool(data=bool(release_requested)))
        self._publish_visualization(
            transform_base_book,
            transform_base_tcp,
            transform_base_policy_tool,
        )

    def _publish_pose(self, publisher, transform, stamp):
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = stamp
        pose.pose = _pose_from_transform(transform)
        publisher.publish(pose)

    def _publish_visualization(
        self,
        transform_base_book,
        transform_base_tcp,
        transform_base_policy_tool,
    ):
        stamp = self.get_clock().now().to_msg()
        poses = {
            "slot": self.geometry.transform_base_slot,
            "current_book": transform_base_book,
            "current_tcp": transform_base_tcp,
            "current_policy_tool": transform_base_policy_tool,
            "target_policy_tool": self.target.transform_base_policy_tool_target,
        }
        for name, transform in poses.items():
            self._publish_pose(self.visual_pose_publishers[name], transform, stamp)
        self._publish_pose(
            self.target_publisher,
            self.target.transform_base_tcp_target,
            stamp,
        )
        self.marker_publisher.publish(build_policy_visualization_markers(
            self.base_frame,
            self.geometry.transform_base_slot,
            transform_base_book,
            transform_base_tcp,
            transform_base_policy_tool,
            self.target.transform_base_tcp_target,
            self.target.transform_base_policy_tool_target,
            slot_depth_m=self.geometry.slot_depth_m,
            slot_width_m=self.geometry.slot_width_m,
            book_size=self.geometry.book_size,
            stamp=stamp,
        ))

    def _begin_visualization_hold(self):
        hold_s = float(self.get_parameter("visualization_hold_s").value)
        self.visualization_hold_deadline_ns = visualization_hold_deadline_ns(
            self._now_ns(), hold_s
        )
        self.phase = "holding_visualization"
        duration = "indefinitely" if self.visualization_hold_deadline_ns is None else f"{hold_s:.1f} s"
        self.get_logger().info(f"Policy visualization held for {duration}")

    def _write_record(self, event):
        payload = dict(self.record)
        payload["event"] = event
        self._append_payload(payload)

    def _append_payload(self, payload):
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _publish_status(self, phase, reason=None):
        payload = {
            "phase": phase,
            "execute": self.execute,
            "requested_execution": self.requested_execution,
            "shadow_full_sequence": self.shadow_full_sequence,
            "run_dir": str(self.run_dir),
        }
        if reason is not None:
            payload["reason"] = str(reason)
        self.status_publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _start_policy_callback(self, _request, response):
        if self.started:
            if self.phase not in ("holding_visualization", "shadow_complete"):
                response.success = False
                response.message = "policy sequence has already started"
                return response
            self._reset_completed_episode()
        self.started = True
        self.phase = "waiting_for_live_state"
        if self.per_grasp_eef_book is None:
            self.get_logger().warning(
                "PER-GRASP EEF->BOOK unavailable; policy will use the explicit "
                "fixed-transform fallback"
            )
        if self.shadow_full_sequence:
            self.get_logger().warning("SHADOW: would start PPO insertion")
            self._publish_status("shadow_policy_start", "SHADOW: would start PPO insertion")
        else:
            self._publish_status("waiting_for_live_state", "operator authorized PPO insertion")
        response.success = True
        response.message = (
            "SHADOW: PPO control-flow rehearsal started"
            if self.shadow_full_sequence
            else "PPO insertion rollout started"
        )
        return response

    def _reset_completed_episode(self):
        """Re-arm only volatile rollout state after a completed episode."""

        self.execution_guard = OneShotExecutionGuard()
        self.record = None
        self.target = None
        self.target_eef = None
        self.execution_start_ns = None
        self.finish_after_ns = None
        self.visualization_hold_deadline_ns = None
        self.nonzero_command_count = 0
        self.continuous_servo_command_count = 0
        self.first_servo_command_ns = None
        self.last_servo_command_ns = None
        self.first_policy_update_ns = None
        self.last_policy_update_ns = None
        self.rollout_start_ns = None
        self.observed_servo_statuses = set()
        self.maximum_rear_to_mouth_m = None
        self.step_index = 0
        self.total_steps = 0
        self.rollout_final_logged = False
        self.released_book_transform = None
        self.push_book_transform = None
        self.phase_start_ns = None
        self.retreat_start_xyz = None
        self.retreat_distance_m = 0.0
        self.push_start_xyz = None
        self.push_book_origin = None
        self.push_geometric_contact_distance_m = None
        self.push_contact_distance_m = None
        self.book_contact_gap_m = None
        self.push_distance_m = 0.0
        self.book_push_distance_m = 0.0
        self.push_policy_index = 0
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_retry_start_ns = None
        self.gripper_next_attempt_ns = 0
        self.shadow_sequence = []
        self.shadow_next_transition_ns = None

    def _start_shadow_sequence(self, transform_base_eef, transform_base_tcp):
        """Advance control flow without fabricating robot motion or TF changes."""

        transform_base_book = transform_base_eef @ self._active_eef_book_transform()
        retreat_target_xyz = (
            transform_base_eef[:3, 3]
            + self.retreat_direction
            * float(self.get_parameter("retreat_distance_m").value)
        )
        insertion_direction = -self.retreat_direction
        contact_gap = oriented_box_contact_gap(
            transform_base_tcp[:3, 3],
            transform_base_book,
            self.geometry.book_size,
            insertion_direction,
        )
        transform_slot_base = invert_transform(self.geometry.transform_base_slot)
        transform_base_policy_tool = (
            transform_base_eef @ self.geometry.transform_eef_policy_tool
        )
        push_raw, push_observation = compute_policy_observation(
            transform_slot_base @ transform_base_book,
            transform_slot_base @ transform_base_policy_tool,
            book_size=self.geometry.book_size,
            slot_depth=self.geometry.slot_depth_m,
            mode_observation=1.0,
            gripper_open=0.0,
            scales=self.geometry.observation_scales,
        )
        push_normalized, push_actor_mean, push_action = self.actor.predict(
            push_observation
        )
        push_nominal = compute_push_nominal_delta(
            push_raw, NominalPushConfig(book_size=self.geometry.book_size)
        )
        push_residual = scale_residual_action(push_action)
        push_final = combine_motion_delta(push_nominal, push_residual)
        self._log_phase_event(
            "shadow_insert_intent",
            transform_base_eef,
            transform_base_tcp,
            intended_target_tcp=self.record["target_tcp_pose"],
            note="stationary live state; no physical state update fabricated",
        )
        self._append_payload({
            "event": "shadow_post_insert_intent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "physical_state_fabricated": False,
            "release_open_intended": True,
            "retreat_target_xyz": retreat_target_xyz.tolist(),
            "retreat_distance_m": float(
                self.get_parameter("retreat_distance_m").value
            ),
            "empty_gripper_close_intended": True,
            "push_contact_gap_m_from_stationary_state": float(contact_gap),
            "requested_book_push_distance_m": float(
                self.get_parameter("push_book_distance_m").value
            ),
            "push_raw_observation": push_raw.tolist(),
            "push_policy_observation": push_observation.tolist(),
            "push_vecnormalize_observation": push_normalized.tolist(),
            "push_ppo_actor_mean": push_actor_mean.tolist(),
            "push_ppo_residual_action": push_action.tolist(),
            "push_nominal_delta": push_nominal.tolist(),
            "push_scaled_residual_delta": push_residual.tolist(),
            "push_final_delta": push_final.tolist(),
            "note": "all post-INSERT targets use stationary live state",
        })
        self.shadow_sequence = [
            ("release_started", "SHADOW: would release/open gripper"),
            ("retreat_started", "SHADOW: would retreat"),
            ("retreat_complete", "SHADOW: would close empty gripper"),
            ("push_started", "SHADOW: would push"),
            ("episode_complete", "SHADOW: push complete; waiting for H"),
        ]
        self.phase = "shadow_sequence"
        self.shadow_next_transition_ns = self._now_ns() + int(0.25e9)
        self._publish_status(
            "shadow_insert", "SHADOW: intended INSERT target calculated; no motion sent"
        )

    def _shadow_sequence_tick(self):
        if self._now_ns() < self.shadow_next_transition_ns:
            return
        phase, message = self.shadow_sequence.pop(0)
        self.get_logger().warning(message)
        self._append_payload({
            "event": f"shadow_{phase}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "message": message,
            "physical_state_fabricated": False,
        })
        self._publish_status(phase, message)
        if self.shadow_sequence:
            self.phase = "shadow_sequence"
            self.shadow_next_transition_ns = self._now_ns() + int(0.25e9)
        else:
            self.phase = "shadow_complete"


def main(args=None):
    rclpy.init(args=args)
    node = SimplePolicyControlNode()
    try:
        while rclpy.ok() and node.phase not in ("failed", "finished_process"):
            rclpy.spin_once(node, timeout_sec=0.1)
            if (
                node.phase == "holding_visualization"
                and node.visualization_hold_deadline_ns is not None
                and node._now_ns() >= node.visualization_hold_deadline_ns
            ):
                node.phase = "finished_process"
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
