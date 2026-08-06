#!/usr/bin/env python3
"""Plan virtual-policy-tool steps without any trajectory execution interface."""

import rclpy

from .policy_tool_planner_base import PolicyToolPlannerBase


class PolicyToolPlanCheckerNode(PolicyToolPlannerBase):
    """A structurally plan-only node: no execution action client is created."""

    def __init__(self):
        super().__init__("policy_tool_plan_checker")
        self.get_logger().info("Policy-tool PLAN-ONLY checker started.")
        self.get_logger().info(
            "This process has no trajectory execution, controller, gripper, "
            "or robot-command client."
        )
        self.get_logger().info(
            "It may call MoveIt's planning service and publish target/trajectory diagnostics only."
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyToolPlanCheckerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
