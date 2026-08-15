"""Shared construction of collision-aware MoveIt planning requests."""

import math

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from shape_msgs.msg import SolidPrimitive


def build_pose_motion_plan_request(
    *,
    target_pose: Pose,
    start_joint_state,
    base_frame: str,
    planning_link: str,
    group_name: str,
    workspace_min_xyz,
    workspace_max_xyz,
    planning_pipeline_id: str,
    planner_id: str,
    planning_attempts: int,
    allowed_planning_time_s: float,
    velocity_scaling: float,
    acceleration_scaling: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    constraint_name: str,
):
    """Build a service request only; this function cannot execute a trajectory."""

    request = GetMotionPlan.Request()
    motion_request = request.motion_plan_request
    motion_request.group_name = str(group_name)
    motion_request.pipeline_id = str(planning_pipeline_id)
    motion_request.planner_id = str(planner_id)
    motion_request.num_planning_attempts = int(planning_attempts)
    motion_request.allowed_planning_time = float(allowed_planning_time_s)
    motion_request.max_velocity_scaling_factor = float(velocity_scaling)
    motion_request.max_acceleration_scaling_factor = float(acceleration_scaling)
    motion_request.start_state = RobotState(joint_state=start_joint_state)

    motion_request.workspace_parameters.header.frame_id = str(base_frame)
    motion_request.workspace_parameters.min_corner.x = float(workspace_min_xyz[0])
    motion_request.workspace_parameters.min_corner.y = float(workspace_min_xyz[1])
    motion_request.workspace_parameters.min_corner.z = float(workspace_min_xyz[2])
    motion_request.workspace_parameters.max_corner.x = float(workspace_max_xyz[0])
    motion_request.workspace_parameters.max_corner.y = float(workspace_max_xyz[1])
    motion_request.workspace_parameters.max_corner.z = float(workspace_max_xyz[2])

    primitive = SolidPrimitive(type=SolidPrimitive.BOX)
    primitive.dimensions = [2.0 * float(position_tolerance_m)] * 3
    region = BoundingVolume(
        primitives=[primitive],
        primitive_poses=[target_pose],
    )
    position = PositionConstraint()
    position.header.frame_id = str(base_frame)
    position.link_name = str(planning_link)
    position.constraint_region = region
    position.weight = 1.0

    orientation = OrientationConstraint()
    orientation.header.frame_id = str(base_frame)
    orientation.link_name = str(planning_link)
    orientation.orientation = target_pose.orientation
    orientation.absolute_x_axis_tolerance = float(orientation_tolerance_rad)
    orientation.absolute_y_axis_tolerance = float(orientation_tolerance_rad)
    orientation.absolute_z_axis_tolerance = float(orientation_tolerance_rad)
    orientation.weight = 1.0

    motion_request.goal_constraints = [
        Constraints(
            name=str(constraint_name),
            position_constraints=[position],
            orientation_constraints=[orientation],
        )
    ]
    return request


def build_position_ik_request(
    *,
    target_pose: Pose,
    start_joint_state,
    base_frame: str,
    planning_link: str,
    group_name: str,
    timeout_s: float,
    attempts: int,
    avoid_collisions: bool,
):
    """Build a seeded, collision-aware IK service request without execution."""

    request = GetPositionIK.Request()
    ik_request = request.ik_request
    ik_request.group_name = str(group_name)
    ik_request.robot_state = RobotState(joint_state=start_joint_state)
    ik_request.avoid_collisions = bool(avoid_collisions)
    ik_request.ik_link_name = str(planning_link)
    ik_request.pose_stamped = PoseStamped()
    ik_request.pose_stamped.header.frame_id = str(base_frame)
    ik_request.pose_stamped.pose = target_pose
    timeout_ns = int(round(max(float(timeout_s), 0.0) * 1.0e9))
    ik_request.timeout.sec = timeout_ns // 1_000_000_000
    ik_request.timeout.nanosec = timeout_ns % 1_000_000_000
    ik_request.attempts = max(int(attempts), 1)
    return request


def build_joint_motion_plan_request(
    *,
    target_joint_names,
    target_joint_positions,
    start_joint_state,
    group_name: str,
    planning_pipeline_id: str,
    planner_id: str,
    planning_attempts: int,
    allowed_planning_time_s: float,
    velocity_scaling: float,
    acceleration_scaling: float,
    joint_tolerance_rad: float,
    constraint_name: str,
):
    """Build a joint-goal planning request for a separately validated IK result."""

    names = [str(value) for value in target_joint_names]
    positions = [float(value) for value in target_joint_positions]
    if not names or len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("joint goal names and positions are empty or inconsistent")
    tolerance = float(joint_tolerance_rad)
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("joint goal positions must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("joint goal tolerance must be finite and positive")

    request = GetMotionPlan.Request()
    motion_request = request.motion_plan_request
    motion_request.group_name = str(group_name)
    motion_request.pipeline_id = str(planning_pipeline_id)
    motion_request.planner_id = str(planner_id)
    motion_request.num_planning_attempts = int(planning_attempts)
    motion_request.allowed_planning_time = float(allowed_planning_time_s)
    motion_request.max_velocity_scaling_factor = float(velocity_scaling)
    motion_request.max_acceleration_scaling_factor = float(acceleration_scaling)
    motion_request.start_state = RobotState(joint_state=start_joint_state)
    motion_request.goal_constraints = [
        Constraints(
            name=str(constraint_name),
            joint_constraints=[
                JointConstraint(
                    joint_name=name,
                    position=position,
                    tolerance_above=tolerance,
                    tolerance_below=tolerance,
                    weight=1.0,
                )
                for name, position in zip(names, positions)
            ],
        )
    ]
    return request
