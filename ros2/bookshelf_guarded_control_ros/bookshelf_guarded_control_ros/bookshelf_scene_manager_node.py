#!/usr/bin/env python3
"""Manage coarse global-approach collision geometry without commanding motion."""

from __future__ import annotations

import json
from datetime import datetime
import hashlib
from pathlib import Path

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
import numpy as np
import rclpy
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .planning_scene_math import (
    GLOBAL_APPROACH,
    LOCAL_INSERTION,
    configured_box,
    local_handoff_error,
    shelf_box_from_slot,
    shelf_front_plane_error_m,
)
from .policy_tool_control_math import (
    make_transform,
    matrix_to_quaternion_xyzw,
)


def _pose_to_transform(pose) -> np.ndarray:
    return make_transform(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def _transform_to_pose(transform: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(transform[0, 3])
    pose.position.y = float(transform[1, 3])
    pose.position.z = float(transform[2, 3])
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    pose.orientation.x = float(quaternion[0])
    pose.orientation.y = float(quaternion[1])
    pose.orientation.z = float(quaternion[2])
    pose.orientation.w = float(quaternion[3])
    return pose


class BookshelfSceneManagerNode(Node):
    """Apply global/local planning-scene modes without any motion interface."""

    def __init__(self):
        super().__init__("bookshelf_scene_manager")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)

        self.latest_slot_pose = None
        self.latest_slot_pose_ns = None
        self.latest_activation_ready = False
        self.latest_activation_ns = None
        self.current_mode = None
        self.scene_applied = False
        self.apply_pending = False
        self.last_reason = "waiting for approved hardware measurements and slot pose"

        self.apply_client = self.create_client(
            ApplyPlanningScene,
            str(self.get_parameter("apply_planning_scene_service").value),
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.ready_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("ready_topic").value),
            10,
        )
        self.mode_publisher = self.create_publisher(
            String,
            str(self.get_parameter("mode_topic").value),
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("slot_pose_topic").value),
            self._slot_pose_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("activation_ready_topic").value),
            self._activation_callback,
            10,
        )
        self.create_service(
            SetBool,
            str(self.get_parameter("set_local_insertion_service").value),
            self._set_local_insertion_callback,
        )

        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.2)
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)
        self.get_logger().warning(
            "Bookshelf scene manager started without robot motion interfaces. "
            "It can only apply MoveIt collision objects."
        )
        self.get_logger().warning(
            "Local insertion removes the coarse bookshelf keep-out only after "
            "an explicit service request and all configured gates pass."
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("scene_config_path", "")
        self.declare_parameter("hardware_measurements_confirmed", False)
        self.declare_parameter("allow_local_insertion", False)
        self.declare_parameter("apply_planning_scene_service", "/apply_planning_scene")
        self.declare_parameter("slot_pose_topic", "/bookshelf_environment/static_slot_pose")
        self.declare_parameter(
            "activation_ready_topic", "/bookshelf_shadow/policy_activation_ready"
        )
        self.declare_parameter("slot_message_max_age_s", 1.0)
        self.declare_parameter("activation_max_age_s", 0.50)
        self.declare_parameter("maximum_shelf_front_plane_error_m", 0.005)

        self.declare_parameter("shelf_object_id", "bookshelf_global_keepout")
        self.declare_parameter("shelf_box_size_xyz", [0.20, 0.70, 0.40])
        self.declare_parameter(
            "shelf_box_center_offset_slot_xyz", [0.10, 0.0, 0.0]
        )

        self.declare_parameter("table_enabled", True)
        self.declare_parameter("table_object_id", "bookshelf_worktable")
        self.declare_parameter("table_box_size_xyz", [1.20, 1.20, 0.05])
        self.declare_parameter("table_box_center_base_xyz", [0.55, 0.0, -0.025])
        self.declare_parameter(
            "table_box_quaternion_base_xyzw", [0.0, 0.0, 0.0, 1.0]
        )

        self.declare_parameter("held_book_enabled", True)
        self.declare_parameter("held_book_object_id", "bookshelf_held_book")
        self.declare_parameter("held_book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter(
            "held_book_center_tcp_xyz", [0.0081243577, -0.0101568565, -0.0477412824]
        )
        self.declare_parameter(
            "held_book_quaternion_tcp_xyzw",
            [0.4756936127, 0.4680286672, 0.5317004003, 0.5214973038],
        )
        self.declare_parameter(
            "held_book_touch_links",
            [
                "link_tcp",
                "xarm_gripper_base_link",
                "left_finger",
                "right_finger",
                "left_inner_knuckle",
                "right_inner_knuckle",
                "left_outer_knuckle",
                "right_outer_knuckle",
            ],
        )

        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("status_topic", "/bookshelf_scene/status")
        self.declare_parameter("ready_topic", "/bookshelf_scene/ready")
        self.declare_parameter("mode_topic", "/bookshelf_scene/mode")
        self.declare_parameter(
            "set_local_insertion_service", "/bookshelf_scene/set_local_insertion"
        )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _slot_pose_callback(self, message: PoseStamped):
        if message.header.frame_id != self.base_frame:
            self.last_reason = (
                f"slot pose frame is {message.header.frame_id!r}, "
                f"expected {self.base_frame!r}"
            )
            return
        self.latest_slot_pose = message
        self.latest_slot_pose_ns = self._now_ns()

    def _activation_callback(self, message: Bool):
        self.latest_activation_ready = bool(message.data)
        self.latest_activation_ns = self._now_ns()

    def _fresh(self, timestamp_ns, maximum_age_s: float) -> bool:
        if timestamp_ns is None:
            return False
        maximum_age_s = float(maximum_age_s)
        if maximum_age_s <= 0.0:
            return True
        return (self._now_ns() - timestamp_ns) * 1.0e-9 <= maximum_age_s

    def _shelf_front_error(self) -> float:
        return shelf_front_plane_error_m(
            self.get_parameter("shelf_box_size_xyz").value,
            self.get_parameter("shelf_box_center_offset_slot_xyz").value,
        )

    def _global_input_error(self) -> str | None:
        if not bool(self.get_parameter("hardware_measurements_confirmed").value):
            return "hardware measurements are not confirmed"
        if self.latest_slot_pose is None:
            return "approved static slot pose is unavailable"
        if not self._fresh(
            self.latest_slot_pose_ns,
            float(self.get_parameter("slot_message_max_age_s").value),
        ):
            return "approved static slot pose is stale"
        try:
            error = self._shelf_front_error()
        except ValueError as exception:
            return f"invalid scene geometry: {exception}"
        tolerance = float(
            self.get_parameter("maximum_shelf_front_plane_error_m").value
        )
        if abs(error) > tolerance:
            return (
                "shelf box front face does not coincide with the slot mouth: "
                f"error={error:.6f} m"
            )
        return None

    def _set_local_insertion_callback(self, request, response):
        if not request.data:
            error = self._global_input_error()
            if error:
                response.success = False
                response.message = error
                self.last_reason = error
                self._publish_status()
                return response
            accepted = self._request_scene(GLOBAL_APPROACH)
            response.success = accepted
            response.message = (
                "global approach scene application requested"
                if accepted
                else self.last_reason
            )
            return response

        try:
            front_error = self._shelf_front_error()
        except ValueError as exception:
            response.success = False
            response.message = f"invalid scene geometry: {exception}"
            self.last_reason = response.message
            self._publish_status()
            return response
        error = local_handoff_error(
            hardware_measurements_confirmed=bool(
                self.get_parameter("hardware_measurements_confirmed").value
            ),
            allow_local_insertion=bool(
                self.get_parameter("allow_local_insertion").value
            ),
            activation_ready=self.latest_activation_ready,
            activation_fresh=self._fresh(
                self.latest_activation_ns,
                float(self.get_parameter("activation_max_age_s").value),
            ),
            global_scene_applied=(
                self.scene_applied and self.current_mode == GLOBAL_APPROACH
            ),
            shelf_front_plane_error=front_error,
            maximum_front_plane_error_m=float(
                self.get_parameter("maximum_shelf_front_plane_error_m").value
            ),
        )
        if error:
            response.success = False
            response.message = error
            self.last_reason = error
            self._publish_status()
            return response
        accepted = self._request_scene(LOCAL_INSERTION)
        response.success = accepted
        response.message = (
            "local insertion scene application requested"
            if accepted
            else self.last_reason
        )
        return response

    def _timer_callback(self):
        if not self.scene_applied and not self.apply_pending:
            error = self._global_input_error()
            if error is None:
                self._request_scene(GLOBAL_APPROACH)
            else:
                self.last_reason = error
        self._publish_status()

    def _request_scene(self, mode: str) -> bool:
        if self.apply_pending:
            self.last_reason = "a planning-scene update is already pending"
            return False
        if not self.apply_client.wait_for_service(timeout_sec=0.25):
            self.last_reason = "MoveIt apply_planning_scene service is unavailable"
            return False
        try:
            scene = self._planning_scene(mode)
        except ValueError as exception:
            self.last_reason = f"invalid planning-scene configuration: {exception}"
            return False
        request = ApplyPlanningScene.Request(scene=scene)
        self.apply_pending = True
        future = self.apply_client.call_async(request)
        future.add_done_callback(lambda value: self._apply_response(value, mode))
        return True

    def _planning_scene(self, mode: str) -> PlanningScene:
        if self.latest_slot_pose is None:
            raise ValueError("slot pose is unavailable")
        transform_base_slot = _pose_to_transform(self.latest_slot_pose.pose)
        shelf = shelf_box_from_slot(
            transform_base_slot,
            base_frame=self.base_frame,
            size_xyz=self.get_parameter("shelf_box_size_xyz").value,
            center_offset_slot_xyz=self.get_parameter(
                "shelf_box_center_offset_slot_xyz"
            ).value,
        )
        table = configured_box(
            frame_id=self.base_frame,
            size_xyz=self.get_parameter("table_box_size_xyz").value,
            center_xyz=self.get_parameter("table_box_center_base_xyz").value,
            quaternion_xyzw=self.get_parameter(
                "table_box_quaternion_base_xyzw"
            ).value,
            label="table_box",
        )
        held_book = configured_box(
            frame_id=self.tcp_frame,
            size_xyz=self.get_parameter("held_book_size_xyz").value,
            center_xyz=self.get_parameter("held_book_center_tcp_xyz").value,
            quaternion_xyzw=self.get_parameter(
                "held_book_quaternion_tcp_xyzw"
            ).value,
            label="held_book",
        )

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True

        shelf_object = self._collision_object(
            str(self.get_parameter("shelf_object_id").value),
            shelf,
            operation=(
                CollisionObject.ADD
                if mode == GLOBAL_APPROACH
                else CollisionObject.REMOVE
            ),
        )
        scene.world.collision_objects.append(shelf_object)
        if bool(self.get_parameter("table_enabled").value):
            scene.world.collision_objects.append(
                self._collision_object(
                    str(self.get_parameter("table_object_id").value),
                    table,
                    operation=CollisionObject.ADD,
                )
            )
        if bool(self.get_parameter("held_book_enabled").value):
            attached = AttachedCollisionObject()
            attached.link_name = self.tcp_frame
            attached.touch_links = [
                str(value)
                for value in self.get_parameter("held_book_touch_links").value
            ]
            attached.object = self._collision_object(
                str(self.get_parameter("held_book_object_id").value),
                held_book,
                operation=CollisionObject.ADD,
            )
            scene.robot_state.attached_collision_objects.append(attached)
        return scene

    @staticmethod
    def _collision_object(object_id: str, box, *, operation: int):
        message = CollisionObject()
        message.id = object_id
        message.header.frame_id = box.frame_id
        message.operation = int(operation)
        if operation == CollisionObject.ADD:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [float(value) for value in box.size_xyz]
            message.primitives = [primitive]
            message.primitive_poses = [_transform_to_pose(box.transform_frame_box)]
        return message

    def _apply_response(self, future, mode: str):
        self.apply_pending = False
        try:
            response = future.result()
            success = bool(response.success)
        except Exception as error:
            success = False
            self.last_reason = f"apply_planning_scene call failed: {error}"
        if success:
            self.current_mode = mode
            self.scene_applied = True
            self.last_reason = f"{mode} planning scene applied"
            self.get_logger().warning(self.last_reason)
        elif not self.last_reason.startswith("apply_planning_scene call failed"):
            self.last_reason = "MoveIt rejected the planning-scene update"
        self._publish_status()

    def _status(self) -> dict:
        mode = self.current_mode
        return {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "scene_applied": self.scene_applied,
            "apply_pending": self.apply_pending,
            "hardware_measurements_confirmed": bool(
                self.get_parameter("hardware_measurements_confirmed").value
            ),
            "allow_local_insertion": bool(
                self.get_parameter("allow_local_insertion").value
            ),
            "activation_ready": self.latest_activation_ready,
            "reason": self.last_reason,
            "scene_config": self._config_provenance(),
            "geometry": {
                "shelf_box_size_xyz": [
                    float(value)
                    for value in self.get_parameter("shelf_box_size_xyz").value
                ],
                "shelf_box_center_offset_slot_xyz": [
                    float(value)
                    for value in self.get_parameter(
                        "shelf_box_center_offset_slot_xyz"
                    ).value
                ],
                "table_box_size_xyz": [
                    float(value)
                    for value in self.get_parameter("table_box_size_xyz").value
                ],
                "table_box_center_base_xyz": [
                    float(value)
                    for value in self.get_parameter(
                        "table_box_center_base_xyz"
                    ).value
                ],
                "held_book_size_xyz": [
                    float(value)
                    for value in self.get_parameter("held_book_size_xyz").value
                ],
                "held_book_center_tcp_xyz": [
                    float(value)
                    for value in self.get_parameter(
                        "held_book_center_tcp_xyz"
                    ).value
                ],
            },
            "objects": {
                "bookshelf_keepout": bool(
                    self.scene_applied and mode == GLOBAL_APPROACH
                ),
                "table": bool(
                    self.scene_applied
                    and self.get_parameter("table_enabled").value
                ),
                "held_book": bool(
                    self.scene_applied
                    and self.get_parameter("held_book_enabled").value
                ),
            },
            "shelf_front_plane_error_m": (
                self._shelf_front_error() if self.latest_slot_pose is not None else None
            ),
            "hardware_commanded": False,
            "motion_interfaces": [],
        }

    def _config_provenance(self) -> dict:
        configured = str(self.get_parameter("scene_config_path").value)
        path = Path(configured).expanduser() if configured else None
        sha256 = None
        if path is not None and path.is_file():
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "path": str(path) if path is not None else None,
            "sha256": sha256,
        }

    def _publish_status(self):
        status = self._status()
        self.status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))
        ready = bool(self.scene_applied and self.current_mode is not None)
        self.ready_publisher.publish(Bool(data=ready))
        self.mode_publisher.publish(String(data=self.current_mode or "unconfigured"))


def main(args=None):
    rclpy.init(args=args)
    node = BookshelfSceneManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
