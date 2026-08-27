"""Small, side-effect-free MoveIt request constructors."""

import math

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetMotionPlan, GetPositionIK


def build_position_ik_request(*, target_pose, start_joint_state, base_frame,
                              planning_link, group_name, timeout_s=1.0):
    request = GetPositionIK.Request()
    ik = request.ik_request
    ik.group_name = str(group_name)
    ik.robot_state = RobotState(joint_state=start_joint_state)
    ik.avoid_collisions = True
    ik.ik_link_name = str(planning_link)
    ik.pose_stamped = PoseStamped()
    ik.pose_stamped.header.frame_id = str(base_frame)
    ik.pose_stamped.pose = target_pose
    nanoseconds = int(round(max(float(timeout_s), 0.0) * 1.0e9))
    ik.timeout.sec = nanoseconds // 1_000_000_000
    ik.timeout.nanosec = nanoseconds % 1_000_000_000
    return request


def build_joint_motion_plan_request(*, target_joint_names, target_joint_positions,
                                    start_joint_state, group_name,
                                    planning_pipeline_id="", planner_id="",
                                    planning_attempts=3, allowed_planning_time_s=5.0,
                                    velocity_scaling=0.05,
                                    acceleration_scaling=0.05,
                                    joint_tolerance_rad=0.001):
    names = [str(value) for value in target_joint_names]
    positions = [float(value) for value in target_joint_positions]
    if not names or len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("joint goal names and positions are inconsistent")
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("joint positions must be finite")
    request = GetMotionPlan.Request()
    motion = request.motion_plan_request
    motion.group_name = str(group_name)
    motion.pipeline_id = str(planning_pipeline_id)
    motion.planner_id = str(planner_id)
    motion.num_planning_attempts = int(planning_attempts)
    motion.allowed_planning_time = float(allowed_planning_time_s)
    motion.max_velocity_scaling_factor = float(velocity_scaling)
    motion.max_acceleration_scaling_factor = float(acceleration_scaling)
    motion.start_state = RobotState(joint_state=start_joint_state)
    tolerance = float(joint_tolerance_rad)
    motion.goal_constraints = [Constraints(
        name="simple_preinsert_ik_goal",
        joint_constraints=[JointConstraint(
            joint_name=name, position=position,
            tolerance_above=tolerance, tolerance_below=tolerance, weight=1.0,
        ) for name, position in zip(names, positions)],
    )]
    return request
