#!/usr/bin/env python3

import os
import pickle
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from stable_baselines3 import PPO


class PolicyDryRunNode(Node):
    """
    Dry-run policy inference node.

    It subscribes to:
        /bookshelf_policy/observation

    It loads a trained SB3 PPO checkpoint and prints:
        6D action = [dx, dy, dz, dyaw, dpitch, g]

    By default, it does NOT publish actions to the robot.
    """

    def __init__(self):
        super().__init__("policy_dry_run_node")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("vecnormalize_path", "")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("publish_actions", False)

        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.vecnormalize_path = self.get_parameter("vecnormalize_path").value
        self.deterministic = bool(self.get_parameter("deterministic").value)
        self.publish_actions = bool(self.get_parameter("publish_actions").value)

        if not self.checkpoint_path:
            raise RuntimeError("Please provide checkpoint_path:=/path/to/model.zip")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.get_logger().info(f"Loading PPO checkpoint: {self.checkpoint_path}")
        self.model = PPO.load(self.checkpoint_path, env=None, print_system_info=False)

        self.vecnorm = None
        if self.vecnormalize_path:
            if os.path.exists(self.vecnormalize_path):
                self.get_logger().info(f"Loading VecNormalize stats: {self.vecnormalize_path}")
                with open(self.vecnormalize_path, "rb") as f:
                    self.vecnorm = pickle.load(f)
                self.vecnorm.training = False
                self.vecnorm.norm_reward = False
            else:
                self.get_logger().warn(f"VecNormalize path provided but not found: {self.vecnormalize_path}")

        self.sub = self.create_subscription(
            Float32MultiArray,
            "/bookshelf_policy/observation",
            self.obs_callback,
            10,
        )

        self.pub = self.create_publisher(
            Float32MultiArray,
            "/bookshelf_policy/action",
            10,
        )

        self.get_logger().info("Policy dry-run node started.")
        self.get_logger().info(f"deterministic={self.deterministic}, publish_actions={self.publish_actions}")

    def maybe_normalize_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        if self.vecnorm is None:
            return obs_batch

        # VecNormalize expects batched obs.
        return self.vecnorm.normalize_obs(obs_batch)

    def obs_callback(self, msg: Float32MultiArray):
        obs = np.array(msg.data, dtype=np.float32)

        if obs.shape[0] != 12:
            self.get_logger().warn(f"Expected 12D observation, got {obs.shape[0]}D.")
            return

        if not np.all(np.isfinite(obs)):
            self.get_logger().warn(f"Observation contains NaN/Inf: {obs}")
            return

        obs_batch = obs.reshape(1, -1)
        obs_in = self.maybe_normalize_obs(obs_batch)

        action, _ = self.model.predict(obs_in, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape[0] != 6:
            self.get_logger().warn(f"Expected 6D action, got {action.shape[0]}D: {action}")
            return

        action = np.clip(action, -1.0, 1.0)

        self.get_logger().info(
            "obs="
            + np.array2string(obs, precision=3, suppress_small=True)
            + " -> action="
            + np.array2string(action, precision=3, suppress_small=True)
        )

        if self.publish_actions:
            out = Float32MultiArray()
            out.data = [float(x) for x in action]
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyDryRunNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()