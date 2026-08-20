"""Pure geometry and fail-closed gates for physical policy-tool deployment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import threading

import numpy as np


MOTION_LABELS = ("dx", "dy", "dz", "dyaw", "dpitch")
TRAJECTORY_FINGERPRINT_KIND = "canonical_ros_fields_v1"


def canonical_ros_message_sha256(message) -> str:
    """Hash ROS message fields without depending on CDR padding bytes.

    ``rclpy.serialization.serialize_message`` may leave alignment padding with
    process-local byte values.  Those bytes are not ROS message data, so the
    serialized SHA-256 can change across calls or across a DDS round trip.  A
    trajectory approval needs to bind the actual fields instead.
    """

    canonical = _canonical_ros_value(message)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_ros_value(value):
    if hasattr(value, "get_fields_and_field_types"):
        field_types = value.get_fields_and_field_types()
        return {
            "ros_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "field_types": dict(field_types),
            "fields": {
                name: _canonical_ros_value(getattr(value, name))
                for name in field_types
            },
        }
    if isinstance(value, np.ndarray):
        return [_canonical_ros_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "typecode") and hasattr(value, "tolist")
    ):
        items = value.tolist() if hasattr(value, "tolist") else value
        return [_canonical_ros_value(item) for item in items]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_hex": bytes(value).hex()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("ROS message contains a non-finite floating-point value")
        return {"float_hex": number.hex()}
    if isinstance(value, str) or value is None:
        return value
    raise TypeError(f"unsupported ROS message field type: {type(value)!r}")


class OneShotExecutionGuard:
    """Atomically consume the process's single trajectory submission slot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._consumed = False

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    @property
    def execution_count(self) -> int:
        return int(self.consumed)

    def try_consume(self) -> bool:
        """Return true exactly once for the lifetime of this guard."""

        with self._lock:
            if self._consumed:
                return False
            self._consumed = True
            return True


@dataclass(frozen=True)
class TargetSafetyLimits:
    maximum_delta: tuple[float, float, float, float, float] = (
        0.008,
        0.003,
        0.007,
        math.radians(0.8),
        math.radians(0.6),
    )
    maximum_tcp_translation_step_m: float = 0.010
    maximum_tcp_rotation_step_rad: float = math.radians(1.5)
    workspace_min_xyz: tuple[float, float, float] = (0.20, -0.60, 0.05)
    workspace_max_xyz: tuple[float, float, float] = (1.00, 0.60, 1.00)


@dataclass(frozen=True)
class JointTrajectorySafetyLimits:
    """Limits for a small local MoveIt trajectory before it can be executed."""

    expected_joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    )
    minimum_point_count: int = 2
    require_velocities: bool = True
    maximum_start_error_rad: float = 0.02
    maximum_waypoint_joint_jump_rad: float = 0.05
    maximum_endpoint_joint_delta_rad: float = 0.10
    maximum_joint_path_length_rad: float = 0.30
    minimum_duration_s: float = 0.10
    maximum_duration_s: float = 15.0


@dataclass(frozen=True)
class PolicyToolTarget:
    scaled_delta: np.ndarray
    transform_base_policy_tool_current: np.ndarray
    transform_slot_policy_tool_current: np.ndarray
    transform_slot_policy_tool_target: np.ndarray
    transform_base_policy_tool_target: np.ndarray
    transform_base_tcp_target: np.ndarray
    tcp_translation_step_m: float
    tcp_rotation_step_rad: float
    target_id: str


def make_transform(translation, quaternion_xyzw=None) -> np.ndarray:
    translation = _vector(translation, 3, "translation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    if quaternion_xyzw is not None:
        transform[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return transform


def invert_transform(transform) -> np.ndarray:
    transform = validated_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ transform[:3, 3])
    return result


def quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    x, y, z, w = _normalised_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return _normalised_quaternion(quaternion)


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) Ry(pitch) Rx(roll), matching standard ROS RPY."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(matrix) -> tuple[float, float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    sine_pitch = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    pitch = math.asin(sine_pitch)
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1.0e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    return roll, pitch, yaw


def compute_policy_tool_target(
    transform_base_slot,
    transform_base_tcp,
    transform_tcp_policy_tool,
    motion_delta,
    *,
    command_scale=0.10,
) -> PolicyToolTarget:
    """Apply the trained delta in slot coordinates and solve the TCP target.

    The simulator applies translation directly in its world frame and adds
    pitch/yaw to the policy tool's world-frame RPY. The simulator slot is
    aligned with that world frame. For deployment, the same operation is
    performed in the measured slot frame before converting back to ``link_tcp``.
    """

    transform_base_slot = validated_transform(transform_base_slot)
    transform_base_tcp = validated_transform(transform_base_tcp)
    transform_tcp_policy_tool = validated_transform(transform_tcp_policy_tool)
    delta = _vector(motion_delta, 5, "motion_delta")
    command_scale = float(command_scale)
    if not math.isfinite(command_scale) or not 0.0 < command_scale <= 1.0:
        raise ValueError("command_scale must be finite and in (0, 1].")
    scaled_delta = delta * command_scale

    transform_base_policy_tool_current = transform_base_tcp @ transform_tcp_policy_tool
    transform_slot_policy_tool_current = (
        invert_transform(transform_base_slot) @ transform_base_policy_tool_current
    )
    transform_slot_policy_tool_target = np.array(
        transform_slot_policy_tool_current, copy=True
    )
    transform_slot_policy_tool_target[:3, 3] += scaled_delta[:3]

    roll, pitch, yaw = matrix_to_euler_xyz(
        transform_slot_policy_tool_current[:3, :3]
    )
    transform_slot_policy_tool_target[:3, :3] = euler_xyz_to_matrix(
        roll,
        _wrap_to_pi(pitch + float(scaled_delta[4])),
        _wrap_to_pi(yaw + float(scaled_delta[3])),
    )

    transform_base_policy_tool_target = (
        transform_base_slot @ transform_slot_policy_tool_target
    )
    transform_base_tcp_target = (
        transform_base_policy_tool_target @ invert_transform(transform_tcp_policy_tool)
    )
    tcp_step = invert_transform(transform_base_tcp) @ transform_base_tcp_target
    translation_step = float(np.linalg.norm(tcp_step[:3, 3]))
    rotation_step = rotation_angle_rad(tcp_step[:3, :3])
    target_id = target_digest(
        transform_base_tcp_target,
        scaled_delta,
        transform_base_slot,
    )
    return PolicyToolTarget(
        scaled_delta=scaled_delta,
        transform_base_policy_tool_current=transform_base_policy_tool_current,
        transform_slot_policy_tool_current=transform_slot_policy_tool_current,
        transform_slot_policy_tool_target=transform_slot_policy_tool_target,
        transform_base_policy_tool_target=transform_base_policy_tool_target,
        transform_base_tcp_target=transform_base_tcp_target,
        tcp_translation_step_m=translation_step,
        tcp_rotation_step_rad=rotation_step,
        target_id=target_id,
    )


def target_safety_error(
    target: PolicyToolTarget,
    motion_delta,
    limits: TargetSafetyLimits = TargetSafetyLimits(),
) -> str | None:
    delta = _vector(motion_delta, 5, "motion_delta")
    maximum_delta = _vector(limits.maximum_delta, 5, "maximum_delta")
    if np.any(maximum_delta <= 0.0):
        return "configured motion limits must be positive"
    exceeded = [
        label
        for label, value, maximum in zip(MOTION_LABELS, np.abs(delta), maximum_delta)
        if float(value) > float(maximum) + 1.0e-12
    ]
    if exceeded:
        return f"unscaled policy delta exceeds configured limits: {exceeded}"
    if target.tcp_translation_step_m > limits.maximum_tcp_translation_step_m:
        return (
            f"TCP translation step {target.tcp_translation_step_m:.6f} m exceeds "
            f"{limits.maximum_tcp_translation_step_m:.6f} m"
        )
    if target.tcp_rotation_step_rad > limits.maximum_tcp_rotation_step_rad:
        return (
            f"TCP rotation step {math.degrees(target.tcp_rotation_step_rad):.3f} deg "
            f"exceeds {math.degrees(limits.maximum_tcp_rotation_step_rad):.3f} deg"
        )
    workspace_min = _vector(limits.workspace_min_xyz, 3, "workspace_min_xyz")
    workspace_max = _vector(limits.workspace_max_xyz, 3, "workspace_max_xyz")
    if np.any(workspace_min >= workspace_max):
        return "workspace bounds are invalid"
    position = target.transform_base_tcp_target[:3, 3]
    if np.any(position < workspace_min) or np.any(position > workspace_max):
        return (
            "target TCP position is outside workspace bounds: "
            f"{position.tolist()}"
        )
    return None


def provenance_error(
    adapter_debug,
    policy_debug,
    *,
    expected_policy_tool_status: str,
    expected_slot_status: str,
    expected_book_status: str,
    expected_bundle_sha256: str,
    allow_unverified_policy_tool: bool,
) -> str | None:
    if not isinstance(adapter_debug, dict) or not adapter_debug.get("valid", False):
        return "adapter debug is missing or invalid"
    if not isinstance(policy_debug, dict) or not policy_debug.get("valid", False):
        return "policy debug is missing or invalid"
    actual_tool_status = str(adapter_debug.get("policy_tool_transform_status", ""))
    if actual_tool_status != expected_policy_tool_status:
        return (
            "policy-tool status mismatch: "
            f"expected {expected_policy_tool_status}, got {actual_tool_status}"
        )
    if (
        actual_tool_status.lower().startswith("derived_unverified_")
        and not allow_unverified_policy_tool
    ):
        return f"policy-tool transform is unverified: {actual_tool_status}"
    if str(adapter_debug.get("static_slot_transform_status", "")) != expected_slot_status:
        return "static slot transform status mismatch"
    if str(adapter_debug.get("eef_book_transform_status", "")) != expected_book_status:
        return "EEF-to-book transform status mismatch"
    if str(policy_debug.get("bundle_sha256", "")) != expected_bundle_sha256:
        return "policy bundle SHA-256 mismatch"
    if bool(policy_debug.get("release_requested_diagnostic", False)):
        return "policy requested release; physical release is disabled"
    if bool(policy_debug.get("release_executed", False)):
        return "upstream reported a release execution"
    return None


def execution_authorization_error(
    *,
    dry_run: bool,
    allow_execution: bool,
    planning_scene_complete: bool,
    approval_token: str,
    configured_token: str,
    plan_age_s,
    maximum_plan_age_s: float,
    plan_valid: bool,
    busy: bool,
    execution_consumed: bool,
) -> str | None:
    if dry_run:
        return "dry_run is true"
    if not allow_execution:
        return "allow_execution is false"
    if not planning_scene_complete:
        return "planning_scene_complete is false"
    configured_token = str(configured_token)
    if not configured_token or configured_token in ("CHANGE_ME", "DISABLED"):
        return "approval token is not explicitly configured"
    if str(approval_token) != configured_token:
        return "approval token does not match"
    if busy:
        return "an execution is already active"
    if execution_consumed:
        return "the one-execution-per-process allowance has already been consumed"
    if not plan_valid:
        return "no valid collision-checked plan is available"
    if plan_age_s is None or not math.isfinite(float(plan_age_s)):
        return "plan age is unavailable"
    if float(plan_age_s) > float(maximum_plan_age_s):
        return f"plan is stale ({float(plan_age_s):.3f} s)"
    return None


def maximum_named_joint_difference(
    current_names,
    current_positions,
    planned_names,
    planned_positions,
) -> float:
    current = dict(zip(current_names, current_positions))
    planned = dict(zip(planned_names, planned_positions))
    common = sorted(set(current).intersection(planned))
    if not common:
        raise ValueError("Current and planned joint states have no common names.")
    return max(abs(float(current[name]) - float(planned[name])) for name in common)


def named_joint_target_branch_report(
    current_names,
    current_positions,
    target_names,
    target_positions,
    expected_joint_names,
    maximum_joint_delta_rad,
) -> tuple[dict, str | None]:
    """Check that a named IK result remains near the supplied current state."""

    expected = tuple(str(value) for value in expected_joint_names)
    current_names = tuple(str(value) for value in current_names)
    target_names = tuple(str(value) for value in target_names)
    reasons = []
    report = {
        "passed": False,
        "maximum_allowed_delta_rad": float(maximum_joint_delta_rad),
        "largest_delta_joint": None,
        "maximum_delta_rad": None,
        "per_joint": {},
        "reasons": reasons,
    }
    if not expected or len(set(expected)) != len(expected):
        reasons.append("expected joint names are empty or duplicated")
    if len(current_names) != len(current_positions) or len(set(current_names)) != len(
        current_names
    ):
        reasons.append("current joint names and positions are inconsistent")
    if len(target_names) != len(target_positions) or len(set(target_names)) != len(
        target_names
    ):
        reasons.append("IK joint names and positions are inconsistent")
    if not set(expected).issubset(current_names):
        reasons.append("current state is missing expected arm joints")
    if not set(expected).issubset(target_names):
        reasons.append("IK result is missing expected arm joints")
    maximum = float(maximum_joint_delta_rad)
    if not math.isfinite(maximum) or maximum <= 0.0:
        reasons.append("maximum IK joint delta must be finite and positive")
    if reasons:
        return report, "IK branch check failed: " + "; ".join(reasons)

    current = dict(zip(current_names, current_positions))
    target = dict(zip(target_names, target_positions))
    deltas = {}
    for name in expected:
        start = float(current[name])
        goal = float(target[name])
        if not math.isfinite(start) or not math.isfinite(goal):
            reasons.append(f"joint {name} has a non-finite current or IK position")
            continue
        signed = goal - start
        absolute = abs(signed)
        deltas[name] = absolute
        report["per_joint"][name] = {
            "current_position_rad": start,
            "target_position_rad": goal,
            "signed_delta_rad": signed,
            "absolute_delta_rad": absolute,
            "absolute_delta_deg": math.degrees(absolute),
        }
    if reasons:
        return report, "IK branch check failed: " + "; ".join(reasons)

    largest = max(deltas, key=deltas.get)
    report["largest_delta_joint"] = largest
    report["maximum_delta_rad"] = deltas[largest]
    if deltas[largest] > maximum + 1.0e-12:
        reasons.append(
            f"IK target joint {largest} is too far from current state: "
            f"{deltas[largest]:.6f} > {maximum:.6f} rad"
        )
    report["passed"] = not reasons
    if reasons:
        return report, "IK branch check failed: " + "; ".join(reasons)
    return report, None


def joint_trajectory_sanity(
    joint_names,
    point_positions,
    point_velocities,
    point_times_s,
    start_joint_names,
    start_joint_positions,
    limits: JointTrajectorySafetyLimits = JointTrajectorySafetyLimits(),
) -> tuple[dict, str | None]:
    """Validate one local joint trajectory and return reportable statistics.

    The trajectory may list the expected joints in any order. The start state
    may contain additional joints, but every expected arm joint must exist.
    Empty velocity arrays are accepted only when ``require_velocities`` is
    false; any supplied derivative is always checked for finiteness.
    """

    expected = tuple(str(value) for value in limits.expected_joint_names)
    names = tuple(str(value) for value in joint_names)
    start_names = tuple(str(value) for value in start_joint_names)
    reasons = []
    report = {
        "passed": False,
        "expected_joint_names": list(expected),
        "trajectory_joint_names": list(names),
        "point_count": len(point_positions),
        "require_velocities": bool(limits.require_velocities),
        "maximum_start_error_rad": None,
        "maximum_waypoint_joint_jump_rad": None,
        "maximum_endpoint_joint_delta_rad": None,
        "largest_endpoint_delta_joint": None,
        "per_joint": {},
        "joint_path_length_rad": None,
        "maximum_absolute_velocity_rad_s": None,
        "duration_s": None,
        "reasons": reasons,
    }

    if not expected or len(set(expected)) != len(expected):
        reasons.append("expected arm joint names are empty or duplicated")
    if len(set(names)) != len(names):
        reasons.append("trajectory joint names contain duplicates")
    if len(names) != len(expected) or set(names) != set(expected):
        reasons.append("trajectory does not contain exactly the expected arm joints")
    if len(set(start_names)) != len(start_names):
        reasons.append("trajectory start joint names contain duplicates")
    if len(start_joint_positions) != len(start_names):
        reasons.append("trajectory start names and positions have different lengths")
    if not set(expected).issubset(start_names):
        reasons.append("trajectory start state is missing expected arm joints")

    point_count = len(point_positions)
    if point_count < int(limits.minimum_point_count):
        reasons.append(
            f"trajectory has {point_count} points; "
            f"minimum is {int(limits.minimum_point_count)}"
        )
    if int(limits.minimum_point_count) < 2:
        reasons.append("minimum trajectory point count must be at least two")
    if len(point_velocities) != point_count or len(point_times_s) != point_count:
        reasons.append("trajectory point arrays have inconsistent lengths")

    if reasons:
        return report, "trajectory sanity check failed: " + "; ".join(reasons)

    try:
        positions = np.asarray(point_positions, dtype=np.float64)
        times = np.asarray(point_times_s, dtype=np.float64)
        start_map = {
            name: float(value)
            for name, value in zip(start_names, start_joint_positions)
        }
    except (TypeError, ValueError) as error:
        reasons.append(f"trajectory contains non-numeric values: {error}")
        return report, "trajectory sanity check failed: " + "; ".join(reasons)

    if positions.shape != (point_count, len(names)):
        reasons.append(
            "trajectory positions do not match the point and joint dimensions"
        )
    elif not np.all(np.isfinite(positions)):
        reasons.append("trajectory positions contain non-finite values")
    if times.shape != (point_count,) or not np.all(np.isfinite(times)):
        reasons.append("trajectory timestamps are missing or non-finite")
    elif np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
        reasons.append("trajectory timestamps must be non-negative and strictly increasing")
    if len(start_map) != len(start_names) or not all(
        math.isfinite(start_map[name]) for name in expected
    ):
        reasons.append("trajectory start positions are missing or non-finite")

    velocity_rows = []
    for index, values in enumerate(point_velocities):
        try:
            row = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            reasons.append(f"trajectory velocity point {index} is non-numeric: {error}")
            continue
        if row.size == 0 and not limits.require_velocities:
            continue
        if row.shape != (len(names),):
            reasons.append(
                f"trajectory velocity point {index} does not contain every arm joint"
            )
        elif not np.all(np.isfinite(row)):
            reasons.append(f"trajectory velocity point {index} is non-finite")
        else:
            velocity_rows.append(row)

    numeric_limits = {
        "maximum_start_error_rad": limits.maximum_start_error_rad,
        "maximum_waypoint_joint_jump_rad": (
            limits.maximum_waypoint_joint_jump_rad
        ),
        "maximum_endpoint_joint_delta_rad": (
            limits.maximum_endpoint_joint_delta_rad
        ),
        "maximum_joint_path_length_rad": limits.maximum_joint_path_length_rad,
        "minimum_duration_s": limits.minimum_duration_s,
        "maximum_duration_s": limits.maximum_duration_s,
    }
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in numeric_limits.values()
    ):
        reasons.append("trajectory safety limits must be finite and non-negative")
    if float(limits.minimum_duration_s) > float(limits.maximum_duration_s):
        reasons.append("trajectory duration limits are inverted")

    if reasons:
        return report, "trajectory sanity check failed: " + "; ".join(reasons)

    name_indices = [names.index(name) for name in expected]
    ordered_positions = positions[:, name_indices]
    start = np.asarray([start_map[name] for name in expected], dtype=np.float64)
    start_error = float(np.max(np.abs(ordered_positions[0] - start)))
    endpoint_delta = float(np.max(np.abs(ordered_positions[-1] - start)))
    segments = np.diff(ordered_positions, axis=0)
    signed_endpoint_deltas = ordered_positions[-1] - start
    absolute_endpoint_deltas = np.abs(signed_endpoint_deltas)
    waypoint_jump = float(np.max(np.abs(segments)))
    path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
    duration = float(times[-1])
    maximum_velocity = (
        float(np.max(np.abs(np.asarray(velocity_rows, dtype=np.float64))))
        if velocity_rows
        else None
    )
    per_joint = {}
    for index, name in enumerate(expected):
        joint_segments = segments[:, index]
        joint_velocities = (
            np.asarray(velocity_rows, dtype=np.float64)[:, name_indices[index]]
            if velocity_rows
            else None
        )
        per_joint[name] = {
            "start_position_rad": float(start[index]),
            "first_waypoint_position_rad": float(ordered_positions[0, index]),
            "endpoint_position_rad": float(ordered_positions[-1, index]),
            "signed_endpoint_delta_rad": float(signed_endpoint_deltas[index]),
            "absolute_endpoint_delta_rad": float(absolute_endpoint_deltas[index]),
            "absolute_endpoint_delta_deg": float(
                math.degrees(absolute_endpoint_deltas[index])
            ),
            "path_travel_rad": float(np.sum(np.abs(joint_segments))),
            "maximum_waypoint_jump_rad": float(np.max(np.abs(joint_segments))),
            "maximum_absolute_velocity_rad_s": (
                float(np.max(np.abs(joint_velocities)))
                if joint_velocities is not None
                else None
            ),
        }
    largest_endpoint_index = int(np.argmax(absolute_endpoint_deltas))
    report.update(
        {
            "maximum_start_error_rad": start_error,
            "maximum_waypoint_joint_jump_rad": waypoint_jump,
            "maximum_endpoint_joint_delta_rad": endpoint_delta,
            "largest_endpoint_delta_joint": expected[largest_endpoint_index],
            "per_joint": per_joint,
            "joint_path_length_rad": path_length,
            "maximum_absolute_velocity_rad_s": maximum_velocity,
            "duration_s": duration,
        }
    )

    comparisons = (
        (
            start_error,
            float(limits.maximum_start_error_rad),
            "trajectory first waypoint differs from its start state",
        ),
        (
            waypoint_jump,
            float(limits.maximum_waypoint_joint_jump_rad),
            "trajectory contains an excessive adjacent waypoint joint jump",
        ),
        (
            endpoint_delta,
            float(limits.maximum_endpoint_joint_delta_rad),
            "trajectory endpoint is too far from its start state",
        ),
        (
            path_length,
            float(limits.maximum_joint_path_length_rad),
            "trajectory joint-space path is too long",
        ),
    )
    for value, maximum, label in comparisons:
        if value > maximum + 1.0e-12:
            reasons.append(f"{label}: {value:.6f} > {maximum:.6f} rad")
    if duration < float(limits.minimum_duration_s) - 1.0e-12:
        reasons.append(
            f"trajectory duration is too short: {duration:.6f} < "
            f"{float(limits.minimum_duration_s):.6f} s"
        )
    if duration > float(limits.maximum_duration_s) + 1.0e-12:
        reasons.append(
            f"trajectory duration is too long: {duration:.6f} > "
            f"{float(limits.maximum_duration_s):.6f} s"
        )

    report["passed"] = not reasons
    if reasons:
        return report, "trajectory sanity check failed: " + "; ".join(reasons)
    return report, None


def rotation_angle_rad(rotation) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    if cosine >= 1.0 - 1.0e-12:
        return 0.0
    return math.acos(cosine)


def transform_to_dict(transform) -> dict:
    transform = validated_transform(transform)
    return {
        "translation_xyz_m": [float(value) for value in transform[:3, 3]],
        "quaternion_xyzw": [
            float(value) for value in matrix_to_quaternion_xyzw(transform[:3, :3])
        ],
    }


def target_digest(*arrays) -> str:
    hasher = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value, dtype=np.float64)
        hasher.update(array.shape.__repr__().encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError("Transform has an invalid homogeneous final row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("Transform rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError("Transform rotation determinant is not +1.")
    return transform


def _normalised_quaternion(quaternion) -> np.ndarray:
    quaternion = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize a zero quaternion.")
    return quaternion / norm


def _vector(value, size: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return value


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
