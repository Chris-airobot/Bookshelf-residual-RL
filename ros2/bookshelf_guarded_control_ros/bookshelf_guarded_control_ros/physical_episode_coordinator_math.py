"""Pure contracts for the physical bookshelf episode coordinator."""

from __future__ import annotations

import numpy as np


HARDWARE_AUTHORIZATION_TOKEN = "I_APPROVE_XARM_FULL_EPISODE"


def validate_episode_operation(operation: str, authorization_token: str) -> str:
    """Validate the fail-closed calculate/control boundary."""

    value = str(operation).strip().lower()
    if value not in ("calculate", "control"):
        raise ValueError("operation must be calculate or control")
    if value == "control" and str(authorization_token) != HARDWARE_AUTHORIZATION_TOKEN:
        raise ValueError(
            "control requires the exact xArm full-episode authorization token"
        )
    return value


def trailing_depth_target_reached(
    current_depth_m: float,
    target_depth_m: float,
    tolerance_m: float,
) -> bool:
    """Return whether the live book trailing edge reached its insertion target."""

    values = np.asarray(
        [current_depth_m, target_depth_m, tolerance_m], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("push target values must be finite")
    if tolerance_m < 0.0:
        raise ValueError("push target tolerance must be nonnegative")
    return float(current_depth_m) >= float(target_depth_m) - float(tolerance_m)
