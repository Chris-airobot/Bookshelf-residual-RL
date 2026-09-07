"""Measured PUSH success gate from the July 8 training configuration.

Training env.yaml SHA256: 9a342fa44f9b6314a6062dce9166022f36f00e6af870fa4ed3504dd8640f2c88.
See bookshelf_env_v5._get_dones before commit 1a5586e: July used a seating
band; later evaluation removed its upper depth bounds. Keep the stricter band
and stop on overshoot. ROS book axes are (depth, thickness, height), whereas
Isaac's local book axes were (depth, height, thickness).
"""

from dataclasses import dataclass
import math

import numpy as np

from .policy_observation_math import _book_corners_in_slot
from .policy_tool_math import validated_transform


@dataclass(frozen=True)
class JulyPushSuccess:
    rear_min_m: float = -0.012
    rear_max_m: float = 0.002
    front_clear_min_m: float = -0.003
    front_clear_max_m: float = 0.055
    front_epsilon_m: float = 0.0002
    lateral_epsilon_m: float = 0.0015
    z_threshold_m: float = 0.015
    yaw_threshold_rad: float = math.radians(8.0)
    upright_dot_min: float = 0.85
    consecutive_samples: int = 4


def seating_metrics(transform_slot_book, book_size, slot_depth_m, slot_width_m):
    """All July geometric gates, in accepted slot coordinates, without clipping."""
    transform = validated_transform(transform_slot_book)
    if not np.all(np.isfinite([slot_depth_m, slot_width_m, *book_size])):
        raise ValueError("nonfinite PUSH geometry")
    if slot_depth_m <= 0 or slot_width_m <= 0 or min(book_size) <= 0:
        raise ValueError("nonpositive PUSH geometry")
    cfg = JulyPushSuccess()
    corners = _book_corners_in_slot(transform, book_size)
    rear, front = float(corners[:, 0].min()), float(corners[:, 0].max())
    clearance = float(slot_depth_m) - front
    extent = float(np.abs(corners[:, 1]).max())
    yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    z = float(transform[2, 3])
    upright = abs(float(transform[2, 2]))
    remaining = max(0.0, cfg.rear_min_m - rear,
                    clearance - cfg.front_clear_max_m - cfg.front_epsilon_m)
    gates = {
        "rear_ok": cfg.rear_min_m <= rear <= cfg.rear_max_m,
        "front_ok": cfg.front_clear_min_m - cfg.front_epsilon_m <= clearance
        <= cfg.front_clear_max_m + cfg.front_epsilon_m,
        "lateral_ok": extent <= 0.5 * slot_width_m + cfg.lateral_epsilon_m,
        "z_ok": abs(z) < cfg.z_threshold_m,
        "yaw_ok": abs(yaw) < cfg.yaw_threshold_rad,
        "upright_ok": upright > cfg.upright_dot_min,
    }
    return {
        **gates,
        "success_geometry": all(gates.values()),
        "depth_overshoot": rear > cfg.rear_max_m
        or clearance < cfg.front_clear_min_m - cfg.front_epsilon_m,
        "book_depth_m": front,
        "rear_to_mouth_m": rear,
        "front_to_back_m": clearance,
        "lateral_extent_m": extent,
        "lateral_limit_m": 0.5 * slot_width_m + cfg.lateral_epsilon_m,
        "z_m": z,
        "yaw_rad": yaw,
        "upright_dot": upright,
        "target_min_depth_m": max(
            slot_depth_m - cfg.front_clear_max_m - cfg.front_epsilon_m,
            front - rear + cfg.rear_min_m,
        ),
        "remaining_depth_m": remaining,
        "target_rear_band_m": [cfg.rear_min_m, cfg.rear_max_m],
        "target_front_clear_band_m": [cfg.front_clear_min_m - cfg.front_epsilon_m,
                                      cfg.front_clear_max_m + cfg.front_epsilon_m],
    }


class PushCompletion:
    """PUSH-only budget and fresh-measurement hold; never infers success from travel."""

    def __init__(self, maximum_travel_m, timeout_s, marker_max_age_s):
        limits = [maximum_travel_m, timeout_s, marker_max_age_s]
        if not np.all(np.isfinite(limits)) or min(limits) <= 0:
            raise ValueError("PUSH safety limits must be finite and positive")
        self.maximum_travel_m = float(maximum_travel_m)
        self.timeout_s = float(timeout_s)
        self.marker_max_age_s = float(marker_max_age_s)
        self.last_stamp_ns = None
        self.success_samples = 0

    def limit_reason(self, elapsed_s, tcp_path_m):
        if not np.all(np.isfinite([elapsed_s, tcp_path_m])) or min(elapsed_s, tcp_path_m) < 0:
            return "SAFETY_STOP"
        if elapsed_s >= self.timeout_s:
            return "TIMEOUT"
        if tcp_path_m >= self.maximum_travel_m:
            return "MAX_TRAVEL"
        return None

    def observe(self, metrics, stamp_ns, now_ns):
        age = (now_ns - stamp_ns) * 1.0e-9
        if stamp_ns <= 0 or age < 0 or age > self.marker_max_age_s:
            raise ValueError(f"PUSH book TF is stale/future/unstamped (age={age:.3f}s)")
        if self.last_stamp_ns is not None and stamp_ns < self.last_stamp_ns:
            raise ValueError("PUSH book TF timestamp regressed")
        if metrics["depth_overshoot"]:
            raise ValueError("PUSH book has passed July's safe seating band")
        if stamp_ns != self.last_stamp_ns:
            self.success_samples = self.success_samples + 1 if metrics["success_geometry"] else 0
            self.last_stamp_ns = stamp_ns
        return self.success_samples >= JulyPushSuccess().consecutive_samples
