"""Pure safety checks for the global calibrated pre-insertion target."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .policy_tool_control_math import validated_transform


@dataclass(frozen=True)
class PreinsertTargetLimits:
    maximum_translation_m: float = 0.75
    maximum_rotation_rad: float = math.radians(5.0)
    workspace_min_xyz: tuple[float, float, float] = (0.20, -0.60, 0.05)
    workspace_max_xyz: tuple[float, float, float] = (1.00, 0.60, 1.00)


def rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def preinsert_target_metrics(transform_base_tcp_current, transform_base_tcp_target):
    current = validated_transform(transform_base_tcp_current)
    target = validated_transform(transform_base_tcp_target)
    relative = np.linalg.inv(current) @ target
    return {
        "translation_m": float(np.linalg.norm(relative[:3, 3])),
        "rotation_rad": rotation_angle_rad(relative[:3, :3]),
    }


def preinsert_target_error(
    transform_base_tcp_current,
    transform_base_tcp_target,
    *,
    limits: PreinsertTargetLimits = PreinsertTargetLimits(),
) -> str | None:
    current = validated_transform(transform_base_tcp_current)
    target = validated_transform(transform_base_tcp_target)
    metrics = preinsert_target_metrics(current, target)
    if metrics["translation_m"] > float(limits.maximum_translation_m):
        return (
            "pre-insertion TCP translation exceeds limit: "
            f"{metrics['translation_m']:.6f} m > "
            f"{float(limits.maximum_translation_m):.6f} m"
        )
    if metrics["rotation_rad"] > float(limits.maximum_rotation_rad):
        return (
            "pre-insertion TCP rotation exceeds limit: "
            f"{math.degrees(metrics['rotation_rad']):.3f} deg > "
            f"{math.degrees(float(limits.maximum_rotation_rad)):.3f} deg"
        )
    lower = np.asarray(limits.workspace_min_xyz, dtype=np.float64)
    upper = np.asarray(limits.workspace_max_xyz, dtype=np.float64)
    position = target[:3, 3]
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        return "pre-insertion workspace bounds are invalid"
    if np.any(position < lower) or np.any(position > upper):
        return f"pre-insertion TCP target is outside workspace bounds: {position.tolist()}"
    return None


def target_identifier(transform_base_tcp_target) -> str:
    target = validated_transform(transform_base_tcp_target)
    rounded = np.round(target, decimals=10).astype("<f8", copy=False)
    return hashlib.sha256(rounded.tobytes()).hexdigest()
