from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from bookshelf_guarded_control_ros.pose_motion_plan import (
    build_pose_motion_plan_request,
)


def test_pose_plan_request_uses_current_state_and_bounded_pose_constraints():
    pose = Pose()
    pose.position.x = 0.81
    pose.position.y = 0.09
    pose.position.z = 0.18
    pose.orientation.w = 1.0
    joint_state = JointState(
        name=[f"joint{index}" for index in range(1, 8)],
        position=[0.1 * index for index in range(7)],
    )

    request = build_pose_motion_plan_request(
        target_pose=pose,
        start_joint_state=joint_state,
        base_frame="link_base",
        planning_link="link_tcp",
        group_name="xarm7",
        workspace_min_xyz=[0.20, -0.60, 0.05],
        workspace_max_xyz=[1.00, 0.60, 1.00],
        planning_pipeline_id="",
        planner_id="",
        planning_attempts=3,
        allowed_planning_time_s=5.0,
        velocity_scaling=0.05,
        acceleration_scaling=0.05,
        position_tolerance_m=0.001,
        orientation_tolerance_rad=0.017,
        constraint_name="calibrated_preinsert_test",
        goal_joint_names=joint_state.name,
        maximum_goal_joint_delta_rad=1.5,
    )

    motion = request.motion_plan_request
    assert motion.group_name == "xarm7"
    assert list(motion.start_state.joint_state.name) == list(joint_state.name)
    assert list(motion.start_state.joint_state.position) == list(
        joint_state.position
    )
    assert motion.max_velocity_scaling_factor == 0.05
    assert motion.max_acceleration_scaling_factor == 0.05
    assert len(motion.goal_constraints) == 1

    constraints = motion.goal_constraints[0]
    assert constraints.name == "calibrated_preinsert_test"
    assert constraints.position_constraints[0].link_name == "link_tcp"
    assert constraints.orientation_constraints[0].link_name == "link_tcp"
    primitive = constraints.position_constraints[0].constraint_region.primitives[0]
    assert list(primitive.dimensions) == [0.002, 0.002, 0.002]
    assert constraints.orientation_constraints[0].orientation == pose.orientation
    assert [constraint.joint_name for constraint in constraints.joint_constraints] == list(
        joint_state.name
    )
    for index, constraint in enumerate(constraints.joint_constraints):
        assert constraint.position == joint_state.position[index]
        assert constraint.tolerance_above == 1.5
        assert constraint.tolerance_below == 1.5


def test_pose_plan_request_rejects_missing_current_goal_joint():
    pose = Pose()
    pose.orientation.w = 1.0
    joint_state = JointState(name=["joint1"], position=[0.0])

    try:
        build_pose_motion_plan_request(
            target_pose=pose,
            start_joint_state=joint_state,
            base_frame="link_base",
            planning_link="link_tcp",
            group_name="xarm7",
            workspace_min_xyz=[0.20, -0.60, 0.05],
            workspace_max_xyz=[1.00, 0.60, 1.00],
            planning_pipeline_id="",
            planner_id="",
            planning_attempts=3,
            allowed_planning_time_s=5.0,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
            position_tolerance_m=0.001,
            orientation_tolerance_rad=0.017,
            constraint_name="missing_joint_test",
            goal_joint_names=["joint1", "joint2"],
            maximum_goal_joint_delta_rad=1.5,
        )
    except ValueError as error:
        assert "joint2" in str(error)
    else:
        raise AssertionError("missing constrained joint was accepted")
