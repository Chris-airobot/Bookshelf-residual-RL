"""Shared construction of collision-aware MoveIt pose-plan requests."""

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan
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
