#!/usr/bin/env python3
"""Exercise Candidate 5 through MoveIt Servo on verified xArm fake hardware only."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListHardwareComponents
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


CANDIDATE_5 = [
    0.142503,
    0.767091,
    -0.019342,
    1.377317,
    -3.100515,
    0.983269,
    3.154346,
]
JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]
FAKE_HARDWARE_PLUGIN = "uf_robot_hardware/UFRobotFakeSystemHardware"
STATUS_NAMES = {
    -1: "INVALID",
    0: "NO_WARNING",
    1: "DECELERATE_FOR_APPROACHING_SINGULARITY",
    2: "HALT_FOR_SINGULARITY",
    3: "DECELERATE_FOR_COLLISION",
    4: "HALT_FOR_COLLISION",
    5: "JOINT_BOUND",
    6: "DECELERATE_FOR_LEAVING_SINGULARITY",
}
HALTING_STATUSES = {2, 4, 5}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-log",
        default=(
            "/home/riot/BookshelfFiles/experiment_logs/"
            "simple_policy_20260827_232929/policy_step.jsonl"
        ),
        help="Real rollout log supplying the exact accepted slot +X direction.",
    )
    parser.add_argument(
        "--output", default="/tmp/xarm7_candidate5_servo_diagnostic.json"
    )
    parser.add_argument("--distance-m", type=float, default=0.100)
    parser.add_argument("--control-rate-hz", type=float, default=30.0)
    parser.add_argument("--maximum-speed-m-s", type=float, default=0.025)
    parser.add_argument("--command-duration-s", type=float, default=0.20)
    parser.add_argument("--translation-tolerance-m", type=float, default=0.0005)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--stall-window-s", type=float, default=3.0)
    parser.add_argument("--stall-progress-m", type=float, default=0.0002)
    return parser.parse_args()


def slot_x_direction(log_path):
    with Path(log_path).expanduser().open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            transform = record.get("T_base_slot")
            if transform and "matrix" in transform:
                matrix = transform["matrix"]
                vector = [float(matrix[row][0]) for row in range(3)]
                norm = math.sqrt(sum(value * value for value in vector))
                if norm <= 0.0:
                    raise RuntimeError("T_base_slot has a zero-length +X axis")
                return [value / norm for value in vector]
    raise RuntimeError(f"no T_base_slot was found in {log_path}")


def pose_record(transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return {
        "translation_xyz": [
            float(translation.x),
            float(translation.y),
            float(translation.z),
        ],
        "quaternion_xyzw": [
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        ],
    }


def call(node, client, request, timeout_s):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        raise RuntimeError(f"service call timed out after {timeout_s:.1f} s")
    if future.exception() is not None:
        raise future.exception()
    return future.result()


class Candidate5ServoDiagnostic(Node):
    def __init__(self):
        super().__init__("xarm7_candidate5_servo_diagnostic")
        self.hardware = self.create_client(
            ListHardwareComponents, "/controller_manager/list_hardware_components"
        )
        self.start_servo = self.create_client(Trigger, "/servo_server/start_servo")
        self.initializer = ActionClient(
            self,
            FollowJointTrajectory,
            "/xarm7_traj_controller/follow_joint_trajectory",
        )
        self.twists = self.create_publisher(
            TwistStamped, "/servo_server/delta_twist_cmds", 10
        )
        self.create_subscription(Int8, "/servo_server/status", self._status, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_status = None
        self.status_transitions = []
        self.motion_start = None

    def _status(self, message):
        status = int(message.data)
        if status != self.latest_status:
            elapsed = 0.0 if self.motion_start is None else time.monotonic() - self.motion_start
            self.status_transitions.append(
                {
                    "elapsed_s": float(elapsed),
                    "status": status,
                    "name": STATUS_NAMES.get(status, "UNKNOWN"),
                }
            )
            self.latest_status = status

    def verify_fake_hardware(self):
        if not self.hardware.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("controller-manager hardware inspection is unavailable")
        response = call(self, self.hardware, ListHardwareComponents.Request(), 10.0)
        components = [
            {
                "name": str(component.name),
                "class_type": str(component.class_type),
            }
            for component in response.component
        ]
        if not any(item["class_type"] == FAKE_HARDWARE_PLUGIN for item in components):
            raise RuntimeError(
                "REFUSED: controller manager does not report the required fake "
                f"hardware plugin {FAKE_HARDWARE_PLUGIN}; found {components}"
            )
        return components

    def initialize_candidate(self):
        if not self.initializer.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("fake xArm trajectory controller is unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = CANDIDATE_5
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        future = self.initializer.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or not future.result().accepted:
            raise RuntimeError("fake trajectory controller rejected Candidate 5")
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        if not result_future.done():
            raise RuntimeError("fake Candidate 5 initialization timed out")
        error_code = int(result_future.result().result.error_code)
        if error_code != int(FollowJointTrajectory.Result.SUCCESSFUL):
            raise RuntimeError(
                f"fake Candidate 5 initialization failed with error {error_code}"
            )

    def lookup_tcp(self, timeout_s=5.0):
        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                return self.tf_buffer.lookup_transform(
                    "link_base",
                    "link_tcp",
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception as error:
                last_error = error
        raise RuntimeError(f"link_base -> link_tcp TF unavailable: {last_error}")

    def begin_servo(self):
        if not self.start_servo.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("MoveIt Servo start service is unavailable")
        response = call(self, self.start_servo, Trigger.Request(), 10.0)
        if not response.success:
            raise RuntimeError(f"MoveIt Servo did not start: {response.message}")

    def publish_twist(self, direction, speed):
        message = TwistStamped()
        message.header.frame_id = "link_base"
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(direction[0] * speed)
        message.twist.linear.y = float(direction[1] * speed)
        message.twist.linear.z = float(direction[2] * speed)
        self.twists.publish(message)


def xyz(transform):
    translation = transform.transform.translation
    return [float(translation.x), float(translation.y), float(translation.z)]


def projected_displacement(start_xyz, current_xyz, direction):
    return float(
        sum((current - start) * axis for current, start, axis in zip(current_xyz, start_xyz, direction))
    )


def run_motion(node, args, direction):
    start_transform = node.lookup_tcp()
    start_xyz = xyz(start_transform)
    node.begin_servo()
    node.motion_start = time.monotonic()
    deadline = node.motion_start + args.timeout_s
    next_tick = node.motion_start
    next_sample = node.motion_start
    progress_history = deque()
    samples = []
    stalled = False
    failure_reason = None
    progress = 0.0
    final_transform = start_transform

    while time.monotonic() < deadline:
        now = time.monotonic()
        rclpy.spin_once(node, timeout_sec=max(0.0, min(next_tick - now, 0.05)))
        now = time.monotonic()
        if now < next_tick:
            continue
        final_transform = node.lookup_tcp(timeout_s=0.5)
        progress = projected_displacement(start_xyz, xyz(final_transform), direction)
        remaining = max(args.distance_m - progress, 0.0)
        speed = min(args.maximum_speed_m_s, remaining / args.command_duration_s)
        if remaining <= args.translation_tolerance_m:
            break
        if node.latest_status in HALTING_STATUSES:
            failure_reason = STATUS_NAMES[node.latest_status]
            break

        node.publish_twist(direction, speed)
        progress_history.append((now, progress))
        while progress_history and now - progress_history[0][0] > args.stall_window_s:
            progress_history.popleft()
        if (
            len(progress_history) >= 2
            and now - progress_history[0][0] >= 0.95 * args.stall_window_s
            and progress - progress_history[0][1] < args.stall_progress_m
        ):
            stalled = True
            failure_reason = "TCP progress stalled while commands continued"
            break
        if now >= next_sample:
            samples.append(
                {
                    "elapsed_s": float(now - node.motion_start),
                    "tcp_translation_xyz": xyz(final_transform),
                    "forward_displacement_m": float(progress),
                    "commanded_speed_m_s": float(speed),
                    "servo_status": node.latest_status,
                    "servo_status_name": STATUS_NAMES.get(node.latest_status, "NOT_RECEIVED"),
                }
            )
            next_sample += 0.1
        next_tick += 1.0 / args.control_rate_hz

    for _ in range(5):
        node.publish_twist(direction, 0.0)
        rclpy.spin_once(node, timeout_sec=1.0 / args.control_rate_hz)

    final_transform = node.lookup_tcp()
    progress = projected_displacement(start_xyz, xyz(final_transform), direction)
    elapsed = time.monotonic() - node.motion_start
    passed = progress >= args.distance_m - args.translation_tolerance_m
    if not passed and failure_reason is None:
        failure_reason = "motion timeout before requested distance"
    return {
        "pass": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
        "failure_reason": failure_reason,
        "stalled": bool(stalled),
        "starting_tcp_pose": pose_record(start_transform),
        "final_tcp_pose": pose_record(final_transform),
        "insertion_direction_base": [float(value) for value in direction],
        "commanded_forward_distance_m": float(args.distance_m),
        "actual_forward_tcp_displacement_m": float(progress),
        "elapsed_s": float(elapsed),
        "servo_status_transitions": node.status_transitions,
        "samples": samples,
    }


def main():
    args = arguments()
    direction = slot_x_direction(args.run_log)
    rclpy.init()
    node = Candidate5ServoDiagnostic()
    result = None
    try:
        components = node.verify_fake_hardware()
        node.initialize_candidate()
        result = run_motion(node, args, direction)
        result.update(
            {
                "diagnostic_only": True,
                "fake_hardware_verified": True,
                "hardware_components": components,
                "candidate_5_joints": CANDIDATE_5,
                "source_run_log": str(Path(args.run_log).expanduser()),
                "control_rate_hz": float(args.control_rate_hz),
                "maximum_speed_m_s": float(args.maximum_speed_m_s),
                "command_duration_s": float(args.command_duration_s),
            }
        )
    except Exception as error:
        result = {
            "pass": False,
            "verdict": "FAIL",
            "diagnostic_only": True,
            "fake_hardware_verified": False,
            "failure_reason": str(error),
            "candidate_5_joints": CANDIDATE_5,
            "source_run_log": str(Path(args.run_log).expanduser()),
        }
    finally:
        node.destroy_node()
        rclpy.shutdown()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"{result['verdict']}: actual forward displacement "
        f"{result.get('actual_forward_tcp_displacement_m', 0.0):.6f} m / "
        f"{args.distance_m:.6f} m"
    )
    if result.get("failure_reason"):
        print(f"reason: {result['failure_reason']}")
    print(f"saved: {output}")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
