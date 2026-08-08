#!/usr/bin/env python3
"""Run the trained residual actor without exposing any robot-command interface."""

import json

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String

from .policy_activation import (
    PolicyActivationLimits,
    PolicyActivationTracker,
    activation_decision_dict,
    evaluate_policy_activation,
    load_activation_envelope,
)
from .policy_shadow_math import (
    MOTION_LABELS,
    NumpyActorBundle,
    NominalInsertConfig,
    POLICY_ACTION_LABELS,
    POLICY_ACTION_SIZE,
    POLICY_OBSERVATION_SIZE,
    ResidualMotionConfig,
    combine_motion_delta,
    compute_insert_nominal_delta,
    scale_residual_action,
    validate_shadow_inputs,
)


class PolicyShadowInferenceNode(Node):
    """Fail-closed, observation-gated policy inference for integration testing."""

    def __init__(self):
        super().__init__("policy_shadow_inference")
        self._declare_parameters()

        if not bool(self.get_parameter("deterministic").value):
            raise ValueError("Shadow deployment supports deterministic PPO inference only.")

        bundle_path = str(self.get_parameter("policy_bundle_path").value).strip()
        if not bundle_path:
            raise ValueError("policy_bundle_path is required.")
        self.bundle = NumpyActorBundle(bundle_path)
        self.bundle_sha256 = self.bundle.sha256
        envelope_path = str(self.get_parameter("activation_envelope_path").value).strip()
        self.activation_envelope = (
            load_activation_envelope(envelope_path) if envelope_path else None
        )
        self.activation_limits = self._activation_limits()
        self.activation_tracker = PolicyActivationTracker(
            int(self.get_parameter("activation_stable_samples").value)
        )
        self.motion_config = self._motion_config()
        self.nominal_config = self._nominal_config()

        self.latest_observation = None
        self.latest_observation_ns = None
        self.latest_raw_metrics = None
        self.latest_raw_metrics_ns = None
        self.observation_valid = False
        self.latest_valid_ns = None
        self.last_status_key = None

        self.inference_valid_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("inference_valid_topic").value),
            10,
        )
        self.policy_action_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("policy_action_topic").value),
            10,
        )
        self.residual_delta_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("residual_delta_topic").value),
            10,
        )
        self.nominal_delta_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("nominal_delta_topic").value),
            10,
        )
        self.final_delta_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("final_delta_topic").value),
            10,
        )
        self.debug_publisher = self.create_publisher(
            String,
            str(self.get_parameter("debug_topic").value),
            10,
        )
        self.activation_ready_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("activation_ready_topic").value),
            10,
        )
        self.activation_debug_publisher = self.create_publisher(
            String,
            str(self.get_parameter("activation_debug_topic").value),
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("observation_topic").value),
            self._observation_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("raw_metrics_topic").value),
            self._raw_metrics_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("observation_valid_topic").value),
            self._valid_callback,
            10,
        )

        rate = max(float(self.get_parameter("inference_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)

        self.get_logger().info("Residual PPO actor loaded in SHADOW-ONLY mode.")
        self.get_logger().info(
            "This node has no robot action, IK, trajectory, controller, gripper, "
            "or robot-control service clients."
        )
        self.get_logger().info(
            f"Portable actor bundle={self.bundle.path} sha256={self.bundle_sha256[:12]}"
        )

    def _declare_parameters(self):
        self.declare_parameter("policy_bundle_path", "")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("inference_rate_hz", 20.0)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("pair_max_skew_s", 0.10)
        self.declare_parameter("activation_envelope_path", "")
        self.declare_parameter("require_activation_envelope", False)
        self.declare_parameter("activation_stable_samples", 10)
        self.declare_parameter("maximum_abs_normalized_observation", 5.0)
        self.declare_parameter("activation_minimum_rear_to_mouth_m", -0.26)
        self.declare_parameter("activation_maximum_rear_to_mouth_m", -0.10)
        self.declare_parameter("activation_minimum_front_to_back_m", 0.12)
        self.declare_parameter("activation_maximum_front_to_back_m", 0.32)
        self.declare_parameter("activation_maximum_abs_lateral_error_m", 0.025)
        self.declare_parameter("activation_maximum_abs_vertical_error_m", 0.030)
        self.declare_parameter("activation_maximum_abs_yaw_error_rad", 0.3490658503988659)
        self.declare_parameter("activation_maximum_gripper_open", 0.25)
        self.declare_parameter("activation_required_mode", 0.0)
        self.declare_parameter("activation_mode_tolerance", 1.0e-6)

        self.declare_parameter("observation_topic", "/bookshelf_policy/observation_12d")
        self.declare_parameter("raw_metrics_topic", "/bookshelf_policy/raw_metrics")
        self.declare_parameter("observation_valid_topic", "/bookshelf_policy/observation_valid")

        self.declare_parameter("inference_valid_topic", "/bookshelf_shadow/inference_valid")
        self.declare_parameter("policy_action_topic", "/bookshelf_shadow/residual_policy_action")
        self.declare_parameter("residual_delta_topic", "/bookshelf_shadow/residual_delta")
        self.declare_parameter("nominal_delta_topic", "/bookshelf_shadow/nominal_delta")
        self.declare_parameter("final_delta_topic", "/bookshelf_shadow/final_delta")
        self.declare_parameter("debug_topic", "/bookshelf_shadow/policy_debug")
        self.declare_parameter(
            "activation_ready_topic", "/bookshelf_shadow/policy_activation_ready"
        )
        self.declare_parameter(
            "activation_debug_topic", "/bookshelf_shadow/policy_activation_debug"
        )

        self.declare_parameter("dx_action_scale", 0.0020)
        self.declare_parameter("dy_action_scale", 0.0010)
        self.declare_parameter("dz_action_scale", 0.0015)
        self.declare_parameter("dyaw_action_scale", 0.006108652381980153)
        self.declare_parameter("dpitch_action_scale", 0.005235987755982988)
        self.declare_parameter("release_threshold", 0.5)

        self.declare_parameter("nominal_insert_dx", 0.0010)
        self.declare_parameter("nominal_insert_dx_near_mouth", 0.0007)
        self.declare_parameter("nominal_lateral_gain", 0.25)
        self.declare_parameter("nominal_height_gain", 0.18)
        self.declare_parameter("nominal_insert_z_offset", 0.006)
        self.declare_parameter("nominal_yaw_gain", 0.14)
        self.declare_parameter("nominal_pitch_gain", 0.020)
        self.declare_parameter("nominal_align_lat_thresh", 0.006)
        self.declare_parameter("nominal_align_z_thresh", 0.010)
        self.declare_parameter("nominal_align_yaw_thresh", 0.10471975511965978)
        self.declare_parameter("nominal_align_tilt_x_thresh", 0.10)
        self.declare_parameter("nominal_unaligned_dx_scale", 0.0)
        self.declare_parameter("nominal_dy_limit", 0.0015)
        self.declare_parameter("nominal_dz_limit", 0.0018)
        self.declare_parameter("nominal_dyaw_limit", 0.006108652381980153)
        self.declare_parameter("nominal_dpitch_limit", 0.004363323129985824)
        self.declare_parameter("nominal_slow_rear_to_mouth", -0.035)

        self.declare_parameter("final_dx_limit", 0.0080)
        self.declare_parameter("final_dy_limit", 0.0030)
        self.declare_parameter("final_dz_limit", 0.0070)
        self.declare_parameter("final_dyaw_limit", 0.013962634015954637)
        self.declare_parameter("final_dpitch_limit", 0.010471975511965976)

    def _motion_config(self) -> ResidualMotionConfig:
        return ResidualMotionConfig(
            action_scales=(
                float(self.get_parameter("dx_action_scale").value),
                float(self.get_parameter("dy_action_scale").value),
                float(self.get_parameter("dz_action_scale").value),
                float(self.get_parameter("dyaw_action_scale").value),
                float(self.get_parameter("dpitch_action_scale").value),
            ),
            final_limits=(
                float(self.get_parameter("final_dx_limit").value),
                float(self.get_parameter("final_dy_limit").value),
                float(self.get_parameter("final_dz_limit").value),
                float(self.get_parameter("final_dyaw_limit").value),
                float(self.get_parameter("final_dpitch_limit").value),
            ),
            release_threshold=float(self.get_parameter("release_threshold").value),
        )

    def _activation_limits(self) -> PolicyActivationLimits:
        return PolicyActivationLimits(
            maximum_abs_normalized_observation=float(
                self.get_parameter("maximum_abs_normalized_observation").value
            ),
            minimum_rear_to_mouth_m=float(
                self.get_parameter("activation_minimum_rear_to_mouth_m").value
            ),
            maximum_rear_to_mouth_m=float(
                self.get_parameter("activation_maximum_rear_to_mouth_m").value
            ),
            minimum_front_to_back_m=float(
                self.get_parameter("activation_minimum_front_to_back_m").value
            ),
            maximum_front_to_back_m=float(
                self.get_parameter("activation_maximum_front_to_back_m").value
            ),
            maximum_abs_lateral_error_m=float(
                self.get_parameter("activation_maximum_abs_lateral_error_m").value
            ),
            maximum_abs_vertical_error_m=float(
                self.get_parameter("activation_maximum_abs_vertical_error_m").value
            ),
            maximum_abs_yaw_error_rad=float(
                self.get_parameter("activation_maximum_abs_yaw_error_rad").value
            ),
            maximum_gripper_open=float(
                self.get_parameter("activation_maximum_gripper_open").value
            ),
            required_mode=float(
                self.get_parameter("activation_required_mode").value
            ),
            mode_tolerance=float(
                self.get_parameter("activation_mode_tolerance").value
            ),
        )

    def _nominal_config(self) -> NominalInsertConfig:
        return NominalInsertConfig(
            insert_dx=float(self.get_parameter("nominal_insert_dx").value),
            insert_dx_near_mouth=float(
                self.get_parameter("nominal_insert_dx_near_mouth").value
            ),
            lateral_gain=float(self.get_parameter("nominal_lateral_gain").value),
            height_gain=float(self.get_parameter("nominal_height_gain").value),
            insert_z_offset=float(self.get_parameter("nominal_insert_z_offset").value),
            yaw_gain=float(self.get_parameter("nominal_yaw_gain").value),
            pitch_gain=float(self.get_parameter("nominal_pitch_gain").value),
            align_lat_thresh=float(self.get_parameter("nominal_align_lat_thresh").value),
            align_z_thresh=float(self.get_parameter("nominal_align_z_thresh").value),
            align_yaw_thresh=float(self.get_parameter("nominal_align_yaw_thresh").value),
            align_tilt_x_thresh=float(
                self.get_parameter("nominal_align_tilt_x_thresh").value
            ),
            unaligned_dx_scale=float(
                self.get_parameter("nominal_unaligned_dx_scale").value
            ),
            dy_limit=float(self.get_parameter("nominal_dy_limit").value),
            dz_limit=float(self.get_parameter("nominal_dz_limit").value),
            dyaw_limit=float(self.get_parameter("nominal_dyaw_limit").value),
            dpitch_limit=float(self.get_parameter("nominal_dpitch_limit").value),
            slow_rear_to_mouth=float(
                self.get_parameter("nominal_slow_rear_to_mouth").value
            ),
        )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _observation_callback(self, message: Float32MultiArray):
        self.latest_observation = np.asarray(message.data, dtype=np.float32)
        self.latest_observation_ns = self._now_ns()

    def _raw_metrics_callback(self, message: Float32MultiArray):
        self.latest_raw_metrics = np.asarray(message.data, dtype=np.float32)
        self.latest_raw_metrics_ns = self._now_ns()

    def _valid_callback(self, message: Bool):
        self.observation_valid = bool(message.data)
        self.latest_valid_ns = self._now_ns()

    def _fresh(self, timestamp_ns) -> bool:
        if timestamp_ns is None:
            return False
        maximum_age = float(self.get_parameter("message_max_age_s").value)
        if maximum_age <= 0.0:
            return True
        return (self._now_ns() - timestamp_ns) * 1.0e-9 <= maximum_age

    def _input_error(self):
        now_ns = self._now_ns()

        def age(timestamp_ns):
            if timestamp_ns is None:
                return None
            return (now_ns - timestamp_ns) * 1.0e-9

        skew = None
        if self.latest_observation_ns is not None and self.latest_raw_metrics_ns is not None:
            skew = abs(self.latest_observation_ns - self.latest_raw_metrics_ns) * 1.0e-9
        return validate_shadow_inputs(
            self.latest_observation,
            self.latest_raw_metrics,
            observation_valid=self.observation_valid,
            valid_age_s=age(self.latest_valid_ns),
            observation_age_s=age(self.latest_observation_ns),
            raw_metrics_age_s=age(self.latest_raw_metrics_ns),
            pair_skew_s=skew,
            maximum_age_s=float(self.get_parameter("message_max_age_s").value),
            maximum_pair_skew_s=float(self.get_parameter("pair_max_skew_s").value),
        )

    def _timer_callback(self):
        error = self._input_error()
        if error:
            self.activation_tracker.reset()
            self._publish_invalid(error)
            return

        try:
            normalized = self.bundle.normalize_observation(self.latest_observation)
            activation_evaluation = evaluate_policy_activation(
                self.latest_observation,
                normalized,
                self.latest_raw_metrics,
                limits=self.activation_limits,
                envelope=self.activation_envelope,
                require_envelope=bool(
                    self.get_parameter("require_activation_envelope").value
                ),
            )
            activation = self.activation_tracker.update(activation_evaluation)
            activation_debug = activation_decision_dict(activation)
            activation_debug.update(
                {
                    "shadow_only": True,
                    "hardware_commanded": False,
                    "envelope_source": (
                        self.activation_envelope.source
                        if self.activation_envelope is not None
                        else None
                    ),
                }
            )
            self._publish_activation(activation_debug)
            if not activation.ready:
                reason = (
                    "; ".join(activation.evaluation.reasons)
                    if activation.evaluation.reasons
                    else (
                        "waiting for stable activation samples "
                        f"({activation.consecutive_ready_samples}/"
                        f"{activation.required_stable_samples})"
                    )
                )
                self._publish_invalid(
                    f"policy activation not ready: {reason}",
                    details={
                        "policy_activation": activation_debug,
                        "vecnormalize_applied": True,
                        "observation_12d": self.latest_observation.round(7).tolist(),
                        "normalized_observation": normalized.round(7).tolist(),
                    },
                )
                return
            normalized, actor_mean, policy_action = self.bundle.predict(
                self.latest_observation
            )
            residual_delta = scale_residual_action(policy_action, self.motion_config)
            nominal_delta = compute_insert_nominal_delta(
                self.latest_raw_metrics,
                self.nominal_config,
            )
            final_delta = combine_motion_delta(
                nominal_delta,
                residual_delta,
                self.motion_config,
            )
        except (ValueError, FloatingPointError) as error:
            self._publish_invalid(f"inference error: {error}")
            return

        release_action = float(policy_action[-1])
        release_requested = release_action > self.motion_config.release_threshold

        self.inference_valid_publisher.publish(Bool(data=True))
        self.activation_ready_publisher.publish(Bool(data=True))
        self.policy_action_publisher.publish(
            Float32MultiArray(data=policy_action.tolist())
        )
        self.residual_delta_publisher.publish(
            Float32MultiArray(data=residual_delta.tolist())
        )
        self.nominal_delta_publisher.publish(
            Float32MultiArray(data=nominal_delta.tolist())
        )
        self.final_delta_publisher.publish(
            Float32MultiArray(data=final_delta.tolist())
        )

        debug = {
            "valid": True,
            "shadow_only": True,
            "command_interfaces": [],
            "bundle_sha256": self.bundle_sha256,
            "vecnormalize_applied": True,
            "deterministic": True,
            "policy_activation_ready": True,
            "policy_activation": activation_debug,
            "observation_12d": self.latest_observation.round(7).tolist(),
            "normalized_observation": normalized.round(7).tolist(),
            "actor_mean": actor_mean.round(7).tolist(),
            "policy_action": {
                label: round(float(value), 7)
                for label, value in zip(POLICY_ACTION_LABELS, policy_action)
            },
            "nominal_delta": {
                label: round(float(value), 7)
                for label, value in zip(MOTION_LABELS, nominal_delta)
            },
            "residual_delta": {
                label: round(float(value), 7)
                for label, value in zip(MOTION_LABELS, residual_delta)
            },
            "final_delta": {
                label: round(float(value), 7)
                for label, value in zip(MOTION_LABELS, final_delta)
            },
            "release_action": round(release_action, 7),
            "release_requested_diagnostic": release_requested,
            "release_executed": False,
        }
        self.debug_publisher.publish(String(data=json.dumps(debug, sort_keys=True)))
        self._log_status_once(
            "valid",
            "Valid shadow inference: VecNormalize -> PPO actor -> nominal/residual diagnostics.",
        )

    def _publish_activation(self, debug):
        self.activation_ready_publisher.publish(Bool(data=bool(debug["ready"])))
        self.activation_debug_publisher.publish(
            String(data=json.dumps(debug, sort_keys=True))
        )

    def _publish_invalid(self, reason: str, *, details=None):
        details = dict(details or {})
        activation = details.get("policy_activation")
        if activation is None:
            activation = {
                "ready": False,
                "instantaneous_ready": False,
                "consecutive_ready_samples": 0,
                "required_stable_samples": self.activation_tracker.required_stable_samples,
                "reasons": [reason],
                "normalized_outliers": {},
                "envelope_outliers": {},
                "geometry": {},
                "shadow_only": True,
                "hardware_commanded": False,
                "envelope_source": (
                    self.activation_envelope.source
                    if self.activation_envelope is not None
                    else None
                ),
            }
            details["policy_activation"] = activation
            self._publish_activation(activation)
        self.inference_valid_publisher.publish(Bool(data=False))
        self.activation_ready_publisher.publish(Bool(data=False))
        debug = {
            "valid": False,
            "shadow_only": True,
            "command_interfaces": [],
            "reason": reason,
            "release_executed": False,
        }
        debug.update(details)
        self.debug_publisher.publish(String(data=json.dumps(debug, sort_keys=True)))
        self._log_status_once(f"invalid:{reason}", f"Shadow inference invalid: {reason}", warning=True)

    def _log_status_once(self, key: str, message: str, warning: bool = False):
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if warning:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyShadowInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
