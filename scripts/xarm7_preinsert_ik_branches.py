#!/usr/bin/env python3
"""Rank xArm7 IK branches for a recorded preinsert TCP path (no motion commands)."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from ament_index_python import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK, GetStateValidity
import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINTS = [f"joint{i}" for i in range(1, 8)]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="policy_step.jsonl containing today's preinsert state")
    parser.add_argument("--output", default="/tmp/xarm7_preinsert_ik_branches.csv")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--path-samples", type=int, default=31)
    parser.add_argument("--insertion-distance-m", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ik-timeout-s", type=float, default=0.10)
    parser.add_argument("--skip-collision-check", action="store_true")
    return parser.parse_args()


def recorded_start(path):
    for line in Path(path).expanduser().open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("event") == "calculated":
            tcp = np.asarray(row["T_base_tcp"]["matrix"], dtype=float)
            slot = np.asarray(row["T_base_slot"]["matrix"], dtype=float)
            return tcp, slot[:3, 0] / np.linalg.norm(slot[:3, 0]), np.asarray(row["arm_joint_positions_rad"])
    raise ValueError("log contains no calculated policy step")


def robot_model():
    xacro = Path(get_package_share_directory("xarm_description")) / "urdf" / "xarm_device.urdf.xacro"
    xml = subprocess.run(
        ["xacro", str(xacro), "dof:=7", "robot_type:=xarm", "limited:=false", "add_gripper:=true"],
        check=True, capture_output=True, text=True,
    ).stdout
    model = pin.buildModelFromXML(xml)
    return model, model.getFrameId("link_tcp")


def pose(matrix):
    result = Pose()
    result.position.x, result.position.y, result.position.z = map(float, matrix[:3, 3])
    q = pin.Quaternion(matrix[:3, :3]).coeffs()  # xyzw
    result.orientation.x, result.orientation.y, result.orientation.z, result.orientation.w = map(float, q)
    return result


class Search(Node):
    def __init__(self, args, model, tcp_frame):
        super().__init__("xarm7_preinsert_ik_branch_search")
        self.args, self.model, self.data, self.tcp_frame = args, model, model.createData(), tcp_frame
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def solve(self, matrix, seed):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name, ik.ik_link_name = "xarm7", "link_tcp"
        ik.robot_state = RobotState(joint_state=JointState(name=JOINTS, position=seed.tolist()))
        ik.pose_stamped.header.frame_id = "link_base"
        ik.pose_stamped.pose = pose(matrix)
        ik.timeout.sec = int(self.args.ik_timeout_s)
        ik.timeout.nanosec = int((self.args.ik_timeout_s % 1.0) * 1e9)
        ik.avoid_collisions = False
        response = self.call(self.ik, request)
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        values = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        return np.asarray([values[name] for name in JOINTS]) if all(name in values for name in JOINTS) else None

    def collision_valid(self, joints):
        if self.args.skip_collision_check:
            return True
        request = GetStateValidity.Request()
        request.group_name = "xarm7"
        request.robot_state = RobotState(joint_state=JointState(name=JOINTS, position=joints.tolist()))
        response = self.call(self.validity, request)
        return bool(response and response.valid)

    def metrics(self, joints):
        q = pin.neutral(self.model)
        margins = []
        for name, value in zip(JOINTS, joints):
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = value
            margins.append(min(value - self.model.lowerPositionLimit[self.model.joints[jid].idx_q],
                               self.model.upperPositionLimit[self.model.joints[jid].idx_q] - value))
        jacobian = pin.computeFrameJacobian(
            self.model, self.data, q, self.tcp_frame, pin.ReferenceFrame.LOCAL
        )[:, :7]
        singular = np.linalg.svd(jacobian, compute_uv=False)
        return float(singular[0] / singular[-1]), float(min(margins))


def main():
    args = arguments()
    start, direction, recorded_seed = recorded_start(args.log)
    model, tcp_frame = robot_model()
    rng = np.random.default_rng(args.seed)
    lower = np.asarray([model.lowerPositionLimit[model.joints[model.getJointId(n)].idx_q] for n in JOINTS])
    upper = np.asarray([model.upperPositionLimit[model.joints[model.getJointId(n)].idx_q] for n in JOINTS])
    seeds = [recorded_seed] + [rng.uniform(lower, upper) for _ in range(max(args.candidates - 1, 0))]
    rclpy.init()
    node = Search(args, model, tcp_frame)
    if not node.ik.wait_for_service(5.0):
        raise RuntimeError("/compute_ik is unavailable; start the existing xArm MoveIt stack")
    if not args.skip_collision_check and not node.validity.wait_for_service(5.0):
        raise RuntimeError("/check_state_validity is unavailable (or use --skip-collision-check)")
    rows, unique = [], []
    try:
        for seed in seeds:
            initial = node.solve(start, seed)
            if initial is None or any(np.linalg.norm(initial - old) < 1e-3 for old in unique):
                continue
            unique.append(initial)
            conditions, margins, collision_ok, current, path_ok = [], [], True, initial, True
            for distance in np.linspace(0.0, args.insertion_distance_m, args.path_samples):
                target = start.copy(); target[:3, 3] += direction * distance
                current = node.solve(target, current)
                if current is None:
                    path_ok = False; break
                condition, margin = node.metrics(current)
                conditions.append(condition); margins.append(margin)
                if margin < 0.0:
                    path_ok = False; break
                collision_ok = collision_ok and node.collision_valid(current)
                if not collision_ok:
                    path_ok = False; break
            rows.append({
                "joints": initial, "initial_condition": conditions[0] if conditions else float("inf"),
                "max_condition": max(conditions, default=float("inf")),
                "final_condition": conditions[-1] if conditions else float("inf"),
                "joint_limit_margin": min(margins, default=-1.0),
                "collision_valid": collision_ok, "valid_path": path_ok and len(conditions) == args.path_samples,
            })
    finally:
        node.destroy_node(); rclpy.shutdown()
    rows.sort(key=lambda row: (not row["valid_path"], row["max_condition"]))
    output = Path(args.output).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["candidate_id", *JOINTS, "initial_condition", "max_condition", "final_condition",
              "joint_limit_margin", "collision_valid", "valid_path"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for candidate_id, row in enumerate(rows):
            values = {key: value for key, value in row.items() if key != "joints"}
            writer.writerow({"candidate_id": candidate_id, **dict(zip(JOINTS, row["joints"])), **values})
    print(f"wrote {len(rows)} unique candidates to {output}")
    for i, row in enumerate(rows[:10]):
        print(f"{i:3d} valid={row['valid_path']} max={row['max_condition']:.2f} final={row['final_condition']:.2f}")


if __name__ == "__main__":
    main()
