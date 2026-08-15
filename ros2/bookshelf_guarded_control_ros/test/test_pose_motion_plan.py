from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from bookshelf_guarded_control_ros.pose_motion_plan import (
    build_joint_motion_plan_request,
    build_pose_motion_plan_request,
    build_position_ik_request,
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
    assert list(constraints.joint_constraints) == []


def test_seeded_ik_and_validated_joint_goal_requests_are_separate():
    pose = Pose()
    pose.orientation.w = 1.0
    joint_state = JointState(
        name=[f"joint{index}" for index in range(1, 8)],
        position=[0.1 * index for index in range(7)],
    )

    ik_request = build_position_ik_request(
        target_pose=pose,
        start_joint_state=joint_state,
        base_frame="link_base",
        planning_link="link_tcp",
        group_name="xarm7",
        timeout_s=1.25,
        avoid_collisions=True,
    ).ik_request
    assert ik_request.group_name == "xarm7"
    assert ik_request.ik_link_name == "link_tcp"
    assert ik_request.pose_stamped.header.frame_id == "link_base"
    assert ik_request.robot_state.joint_state == joint_state
    assert ik_request.avoid_collisions is True
    assert ik_request.timeout.sec == 1
    assert ik_request.timeout.nanosec == 250_000_000

    motion = build_joint_motion_plan_request(
        target_joint_names=joint_state.name,
        target_joint_positions=[value + 0.1 for value in joint_state.position],
        start_joint_state=joint_state,
        group_name="xarm7",
        planning_pipeline_id="",
        planner_id="",
        planning_attempts=3,
        allowed_planning_time_s=5.0,
        velocity_scaling=0.05,
        acceleration_scaling=0.05,
        joint_tolerance_rad=0.001,
        constraint_name="validated_ik_test",
    ).motion_plan_request
    constraints = motion.goal_constraints[0]
    assert constraints.name == "validated_ik_test"
    assert list(constraints.position_constraints) == []
    assert list(constraints.orientation_constraints) == []
    assert [value.joint_name for value in constraints.joint_constraints] == list(
        joint_state.name
    )
    assert all(value.tolerance_above == 0.001 for value in constraints.joint_constraints)


def test_joint_goal_request_rejects_invalid_values():
    joint_state = JointState(name=["joint1"], position=[0.0])

    try:
        build_joint_motion_plan_request(
            target_joint_names=["joint1"],
            target_joint_positions=[float("nan")],
            start_joint_state=joint_state,
            group_name="xarm7",
            planning_pipeline_id="",
            planner_id="",
            planning_attempts=3,
            allowed_planning_time_s=5.0,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
            joint_tolerance_rad=0.001,
            constraint_name="invalid_joint_test",
        )
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite joint target was accepted")
