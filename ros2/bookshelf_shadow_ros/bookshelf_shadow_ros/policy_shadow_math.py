"""Pure NumPy helpers for read-only bookshelf policy shadow inference."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


POLICY_OBSERVATION_SIZE = 12
POLICY_ACTION_SIZE = 6
MOTION_ACTION_SIZE = 5

POLICY_ACTION_LABELS = (
    "residual_dx",
    "residual_dy",
    "residual_dz",
    "residual_dyaw",
    "residual_dpitch",
    "release",
)
MOTION_LABELS = ("dx", "dy", "dz", "dyaw", "dpitch")


def release_requested_for_mode(
    release_action: float,
    mode_observation: float,
    release_threshold: float,
) -> bool:
    """Interpret the learned release action only during INSERT mode."""

    values = np.asarray(
        [release_action, mode_observation, release_threshold], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("release action, mode, and threshold must be finite")
    return bool(
        math.isclose(float(mode_observation), 0.0, abs_tol=1.0e-4)
        and float(release_action) > float(release_threshold)
    )


@dataclass(frozen=True)
class ResidualMotionConfig:
    action_scales: tuple[float, ...] = (
        0.0020,
        0.0010,
        0.0015,
        math.radians(0.35),
        math.radians(0.30),
    )
    final_limits: tuple[float, ...] = (
        0.0080,
        0.0030,
        0.0070,
        math.radians(0.8),
        math.radians(0.6),
    )
    release_threshold: float = 0.5


@dataclass(frozen=True)
class NominalInsertConfig:
    insert_dx: float = 0.0010
    insert_dx_near_mouth: float = 0.0007
    lateral_gain: float = 0.25
    height_gain: float = 0.18
    insert_z_offset: float = 0.006
    yaw_gain: float = 0.14
    pitch_gain: float = 0.020
    align_lat_thresh: float = 0.006
    align_z_thresh: float = 0.010
    align_yaw_thresh: float = math.radians(6.0)
    align_tilt_x_thresh: float = 0.10
    unaligned_dx_scale: float = 0.0
    dy_limit: float = 0.0015
    dz_limit: float = 0.0018
    dyaw_limit: float = math.radians(0.35)
    dpitch_limit: float = math.radians(0.25)
    slow_rear_to_mouth: float = -0.035


@dataclass(frozen=True)
class NominalPushConfig:
    push_dx: float = 0.0008
    lateral_gain: float = 0.35
    height_gain: float = 0.30
    yaw_gain: float = 0.20
    pitch_gain: float = 0.08
    push_z_fraction_from_bottom: float = 0.20
    book_size: tuple[float, float, float] = (0.156, 0.034, 0.236)
    dy_limit: float = 0.0005
    dz_limit: float = 0.0010
    dyaw_limit: float = math.radians(0.35)
    dpitch_limit: float = math.radians(0.25)


def _vector(value, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def validate_shadow_inputs(
    observation,
    raw_metrics,
    *,
    observation_valid: bool,
    valid_age_s,
    observation_age_s,
    raw_metrics_age_s,
    pair_skew_s,
    maximum_age_s=0.50,
    maximum_pair_skew_s=0.10,
) -> str | None:
    """Validate paired shadow-inference inputs without running the actor."""

    if not observation_valid:
        return "upstream observation_valid is false"
    if maximum_age_s > 0.0:
        if valid_age_s is None or valid_age_s > maximum_age_s:
            return "observation_valid message is stale"
        if observation_age_s is None or observation_age_s > maximum_age_s:
            return "12D observation is missing or stale"
        if raw_metrics_age_s is None or raw_metrics_age_s > maximum_age_s:
            return "raw metrics are missing or stale"

    if observation is None:
        return "12D observation is missing or stale"
    if raw_metrics is None:
        return "raw metrics are missing or stale"
    observation = np.asarray(observation, dtype=np.float32)
    raw_metrics = np.asarray(raw_metrics, dtype=np.float32)
    if observation.shape != (POLICY_OBSERVATION_SIZE,):
        return f"expected 12D observation, got {observation.shape}"
    if raw_metrics.shape != (POLICY_OBSERVATION_SIZE,):
        return f"expected 12D raw metrics, got {raw_metrics.shape}"
    if not np.all(np.isfinite(observation)):
        return "12D observation contains non-finite values"
    if not np.all(np.isfinite(raw_metrics)):
        return "raw metrics contain non-finite values"
    if pair_skew_s is None or not math.isfinite(float(pair_skew_s)):
        return "observation/raw metric skew is non-finite"
    if float(pair_skew_s) > maximum_pair_skew_s:
        return f"observation/raw metric skew is {float(pair_skew_s):.3f} s"
    return None


def compute_insert_nominal_delta(
    raw_metrics,
    config: NominalInsertConfig = NominalInsertConfig(),
) -> np.ndarray:
    """Reproduce the simulator's nominal INSERT controller from raw 12D metrics."""

    raw = _vector(raw_metrics, POLICY_OBSERVATION_SIZE, "raw_metrics")
    mode_observation = float(raw[0])
    if not math.isclose(mode_observation, 0.0, abs_tol=1.0e-4):
        raise ValueError(
            "Exact nominal diagnostics currently support INSERT mode only "
            f"(mode observation 0.0), got {mode_observation:.4f}."
        )

    rear_to_mouth = float(raw[1])
    lat_err = float(raw[3])
    z_err = float(raw[4])
    yaw_err = float(raw[5])
    tilt_x = float(raw[10])

    aligned = (
        abs(lat_err) < config.align_lat_thresh
        and abs(z_err) < config.align_z_thresh
        and abs(yaw_err) < config.align_yaw_thresh
        and abs(tilt_x) < config.align_tilt_x_thresh
    )
    dx = (
        config.insert_dx_near_mouth
        if rear_to_mouth > config.slow_rear_to_mouth
        else config.insert_dx
    )
    if not aligned:
        dx *= config.unaligned_dx_scale

    delta = np.array(
        [
            dx,
            np.clip(config.lateral_gain * lat_err, -config.dy_limit, config.dy_limit),
            np.clip(
                -config.height_gain * (z_err - config.insert_z_offset),
                -config.dz_limit,
                config.dz_limit,
            ),
            np.clip(-config.yaw_gain * yaw_err, -config.dyaw_limit, config.dyaw_limit),
            np.clip(-config.pitch_gain * tilt_x, -config.dpitch_limit, config.dpitch_limit),
        ],
        dtype=np.float32,
    )
    return delta


def _book_vertical_half_extent(raw: np.ndarray, book_size) -> float:
    """Recover the oriented book's vertical half-extent from the 12D metrics."""

    depth, thickness, height = _vector(book_size, 3, "book_size")
    yaw = float(raw[5])
    up_x = float(raw[10])
    up_y = float(raw[11])
    up_horizontal_sq = up_x * up_x + up_y * up_y
    if up_horizontal_sq > 1.0 + 1.0e-5:
        raise ValueError("book up-axis components are inconsistent")
    up_z = math.sqrt(max(1.0 - up_horizontal_sq, 0.0))
    up = np.array([up_x, up_y, up_z], dtype=np.float64)

    horizontal_depth = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    projection = float(np.dot(horizontal_depth, up[:2]))
    denominator = math.sqrt(up_z * up_z + projection * projection)
    if denominator <= 1.0e-8:
        raise ValueError("book orientation cannot recover a depth axis")
    depth_axis = np.array(
        [
            horizontal_depth[0] * up_z / denominator,
            horizontal_depth[1] * up_z / denominator,
            -projection / denominator,
        ],
        dtype=np.float64,
    )
    thickness_axis = np.cross(up, depth_axis)
    return 0.5 * float(
        depth * abs(depth_axis[2])
        + thickness * abs(thickness_axis[2])
        + height * abs(up[2])
    )


def compute_push_nominal_delta(
    raw_metrics,
    config: NominalPushConfig = NominalPushConfig(),
) -> np.ndarray:
    """Reproduce the simulator's nominal PUSH controller from raw 12D metrics."""

    raw = _vector(raw_metrics, POLICY_OBSERVATION_SIZE, "raw_metrics")
    mode_observation = float(raw[0])
    if not math.isclose(mode_observation, 1.0, abs_tol=1.0e-4):
        raise ValueError(
            "Nominal push diagnostics require PUSH mode "
            f"(mode observation 1.0), got {mode_observation:.4f}."
        )

    lat_err = float(raw[3])
    yaw_err = float(raw[5])
    tool_to_book_y = float(raw[7])
    tool_to_book_z = float(raw[8])
    tilt_x = float(raw[10])
    vertical_half_extent = _book_vertical_half_extent(raw, config.book_size)
    desired_tool_z_from_book = (
        2.0 * config.push_z_fraction_from_bottom - 1.0
    ) * vertical_half_extent

    return np.array(
        [
            config.push_dx,
            np.clip(
                config.lateral_gain * (lat_err - tool_to_book_y),
                -config.dy_limit,
                config.dy_limit,
            ),
            np.clip(
                config.height_gain * (desired_tool_z_from_book - tool_to_book_z),
                -config.dz_limit,
                config.dz_limit,
            ),
            np.clip(-config.yaw_gain * yaw_err, -config.dyaw_limit, config.dyaw_limit),
            np.clip(-config.pitch_gain * tilt_x, -config.dpitch_limit, config.dpitch_limit),
        ],
        dtype=np.float32,
    )


def compute_policy_nominal_delta(
    raw_metrics,
    insert_config: NominalInsertConfig = NominalInsertConfig(),
    push_config: NominalPushConfig = NominalPushConfig(),
) -> np.ndarray:
    """Dispatch nominal motion using the simulator's INSERT/SCRIPTED/PUSH mode."""

    raw = _vector(raw_metrics, POLICY_OBSERVATION_SIZE, "raw_metrics")
    mode_observation = float(raw[0])
    if math.isclose(mode_observation, 0.0, abs_tol=1.0e-4):
        return compute_insert_nominal_delta(raw, insert_config)
    if math.isclose(mode_observation, 0.5, abs_tol=1.0e-4):
        return np.zeros(MOTION_ACTION_SIZE, dtype=np.float32)
    if math.isclose(mode_observation, 1.0, abs_tol=1.0e-4):
        return compute_push_nominal_delta(raw, push_config)
    raise ValueError(f"unsupported policy mode observation {mode_observation:.4f}")


def scale_residual_action(
    policy_action,
    config: ResidualMotionConfig = ResidualMotionConfig(),
) -> np.ndarray:
    """Apply the same action clamp and residual scales used by the simulator."""

    action = _vector(policy_action, POLICY_ACTION_SIZE, "policy_action")
    action = np.clip(action, -1.0, 1.0)
    scales = _vector(config.action_scales, MOTION_ACTION_SIZE, "action_scales")
    return (action[:MOTION_ACTION_SIZE] * scales).astype(np.float32)


def combine_motion_delta(
    nominal_delta,
    residual_delta,
    config: ResidualMotionConfig = ResidualMotionConfig(),
) -> np.ndarray:
    nominal = _vector(nominal_delta, MOTION_ACTION_SIZE, "nominal_delta")
    residual = _vector(residual_delta, MOTION_ACTION_SIZE, "residual_delta")
    limits = _vector(config.final_limits, MOTION_ACTION_SIZE, "final_limits")
    return np.clip(nominal + residual, -limits, limits).astype(np.float32)


class NumpyActorBundle:
    """Portable deterministic PPO actor and VecNormalize observation statistics."""

    SCHEMA_VERSION = 1
    REQUIRED_KEYS = {
        "schema_version",
        "observation_size",
        "action_size",
        "activation",
        "obs_mean",
        "obs_var",
        "obs_epsilon",
        "obs_clip",
        "action_low",
        "action_high",
        "policy_0_weight",
        "policy_0_bias",
        "policy_1_weight",
        "policy_1_bias",
        "action_weight",
        "action_bias",
        "metadata_json",
    }

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Policy bundle not found: {self.path}")

        with np.load(self.path, allow_pickle=False) as bundle:
            missing = self.REQUIRED_KEYS.difference(bundle.files)
            if missing:
                raise ValueError(f"Policy bundle is missing keys: {sorted(missing)}")
            values = {key: np.array(bundle[key], copy=True) for key in bundle.files}

        version = int(values["schema_version"].item())
        if version != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported policy bundle schema {version}; expected {self.SCHEMA_VERSION}."
            )
        if int(values["observation_size"].item()) != POLICY_OBSERVATION_SIZE:
            raise ValueError("Policy bundle does not use the trained 12D observation.")
        if int(values["action_size"].item()) != POLICY_ACTION_SIZE:
            raise ValueError("Policy bundle does not produce the trained 6D action.")
        if str(values["activation"].item()).lower() != "relu":
            raise ValueError("Only the checkpoint's ReLU actor is supported.")

        self.obs_mean = _vector(values["obs_mean"], POLICY_OBSERVATION_SIZE, "obs_mean")
        self.obs_var = _vector(values["obs_var"], POLICY_OBSERVATION_SIZE, "obs_var")
        if np.any(self.obs_var < 0.0):
            raise ValueError("obs_var contains negative values.")
        self.obs_epsilon = float(values["obs_epsilon"].item())
        self.obs_clip = float(values["obs_clip"].item())
        if self.obs_epsilon <= 0.0 or self.obs_clip <= 0.0:
            raise ValueError("Observation epsilon and clip must be positive.")

        self.action_low = _vector(values["action_low"], POLICY_ACTION_SIZE, "action_low")
        self.action_high = _vector(values["action_high"], POLICY_ACTION_SIZE, "action_high")
        if np.any(self.action_low >= self.action_high):
            raise ValueError("Policy action limits are invalid.")

        self.policy_0_weight = self._matrix(values["policy_0_weight"], (256, 12), "policy_0_weight")
        self.policy_0_bias = _vector(values["policy_0_bias"], 256, "policy_0_bias")
        self.policy_1_weight = self._matrix(values["policy_1_weight"], (256, 256), "policy_1_weight")
        self.policy_1_bias = _vector(values["policy_1_bias"], 256, "policy_1_bias")
        self.action_weight = self._matrix(values["action_weight"], (6, 256), "action_weight")
        self.action_bias = _vector(values["action_bias"], 6, "action_bias")

        try:
            self.metadata = json.loads(str(values["metadata_json"].item()))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid policy bundle metadata: {error}") from error

    @staticmethod
    def _matrix(value, shape: tuple[int, int], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
        return array

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def normalize_observation(self, observation) -> np.ndarray:
        observation = _vector(
            observation,
            POLICY_OBSERVATION_SIZE,
            "observation",
        )
        normalized = (observation - self.obs_mean) / np.sqrt(
            self.obs_var + self.obs_epsilon
        )
        return np.clip(normalized, -self.obs_clip, self.obs_clip).astype(np.float32)

    def predict(self, observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized observation, actor mean, and environment-clamped action."""

        normalized = self.normalize_observation(observation)
        hidden = np.maximum(
            self.policy_0_weight @ normalized + self.policy_0_bias,
            0.0,
        )
        hidden = np.maximum(
            self.policy_1_weight @ hidden + self.policy_1_bias,
            0.0,
        )
        actor_mean = self.action_weight @ hidden + self.action_bias
        sb3_action = np.clip(actor_mean, self.action_low, self.action_high)
        environment_action = np.clip(sb3_action, -1.0, 1.0)
        return (
            normalized.astype(np.float32),
            actor_mean.astype(np.float32),
            environment_action.astype(np.float32),
        )
