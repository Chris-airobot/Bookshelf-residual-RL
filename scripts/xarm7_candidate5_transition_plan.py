#!/usr/bin/env python3
"""Plan and validate the recorded-preinsert to Candidate 5 transition; never execute."""

import argparse
import json
import math
from pathlib import Path

from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetMotionPlan, GetStateValidity
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState

from bookshelf_simple_experiment_ros.moveit_requests import build_joint_motion_plan_request


JOINTS = [f"joint{i}" for i in range(1, 8)]
START = [1.1537, 1.6860, 4.9030, 1.4665, 3.5450, 0.6665, 4.4174]
GOAL = [0.142503, 0.767091, -0.019342, 1.377317, -3.100515, 0.983269, 3.154346]
LOWER = [-2 * math.pi, -2.059, -2 * math.pi, -0.19198, -2 * math.pi, -1.69297, -2 * math.pi]
UPPER = [2 * math.pi, 2.0944, 2 * math.pi, 3.927, 2 * math.pi, math.pi, 2 * math.pi]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/xarm7_candidate5_transition_plan.json")
    parser.add_argument("--planning-time-s", type=float, default=10.0)
    parser.add_argument("--publish-display", action="store_true")
    return parser.parse_args()


class PlanOnly(Node):
    def __init__(self, args):
        super().__init__("xarm7_candidate5_transition_plan_only")
        self.args = args
        self.plan = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.display = self.create_publisher(DisplayTrajectory, "/display_planned_path", qos)

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def state_valid(self, names, positions):
        request = GetStateValidity.Request()
        request.group_name = "xarm7"
        request.robot_state = RobotState(
            joint_state=JointState(name=list(names), position=list(positions))
        )
        response = self.call(self.validity, request)
        return bool(response and response.valid)


def seconds(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def main():
    args = arguments()
    rclpy.init()
    node = PlanOnly(args)
    try:
        if not node.plan.wait_for_service(5.0) or not node.validity.wait_for_service(5.0):
            raise RuntimeError("MoveIt planning/state-validity services are unavailable")
        start_state = JointState(name=JOINTS, position=START)
        request = build_joint_motion_plan_request(
            target_joint_names=JOINTS,
            target_joint_positions=GOAL,
            start_joint_state=start_state,
            group_name="xarm7",
            planning_attempts=5,
            allowed_planning_time_s=args.planning_time_s,
            velocity_scaling=0.05,
            acceleration_scaling=0.05,
            joint_tolerance_rad=0.001,
        )
        response = node.call(node.plan, request).motion_plan_response
        success = int(response.error_code.val) == int(MoveItErrorCodes.SUCCESS)
        names = list(response.trajectory.joint_trajectory.joint_names)
        points = response.trajectory.joint_trajectory.points
        collision_valid = bool(success and points)
        path_length = 0.0
        minimum_margin = float("inf")
        previous = None
        for point in points:
            positions = np.asarray(point.positions, dtype=float)
            if previous is not None:
                path_length += float(np.linalg.norm(positions - previous))
            previous = positions
            values = dict(zip(names, positions))
            arm = np.asarray([values[name] for name in JOINTS])
            minimum_margin = min(
                minimum_margin,
                min(min(value - low, high - value) for value, low, high in zip(arm, LOWER, UPPER)),
            )
            collision_valid = collision_valid and node.state_valid(names, positions)
        duration = seconds(points[-1].time_from_start) if points else 0.0
        valid = success and bool(points) and collision_valid and minimum_margin >= 0.0
        result = {
            "plan_only": True,
            "executed": False,
            "planning_success": bool(success),
            "moveit_error_code": int(response.error_code.val),
            "trajectory_points": int(len(points)),
            "trajectory_duration_s": float(duration),
            "joint_space_path_length_rad": float(path_length),
            "minimum_joint_limit_margin_rad": float(minimum_margin) if points else None,
            "collision_valid": bool(collision_valid),
            "valid_transition": bool(valid),
            "start_joints": START,
            "goal_joints": GOAL,
        }
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.publish_display and success:
            message = DisplayTrajectory()
            message.model_id = "UF_ROBOT"
            message.trajectory_start = response.trajectory_start
            message.trajectory = [response.trajectory]
            node.display.publish(message)
            rclpy.spin_once(node, timeout_sec=1.0)
        print(json.dumps(result, indent=2))
        print(f"saved: {output}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
