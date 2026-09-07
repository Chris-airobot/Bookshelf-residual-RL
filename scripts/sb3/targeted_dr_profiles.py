#!/usr/bin/env python3
"""Targeted sim-to-real fine-tuning profiles for the July six-action policy."""

from __future__ import annotations

import math


PROFILE_NAMES = (
    "disabled",
    "obs_moderate",
    "combined_moderate",
    "strong_tail",
    "combined_moderate_actuation",
    "obs_moderate_curriculum",
    "obs_gripper_no_grasp",
)

EVALUATION_PROFILE_NAMES = (
    "nominal",
    "observation_moderate",
    "strong_tail",
    "physical_like",
    "gripper_only",
    "combined_moderate",
    "combined_moderate_actuation",
)


def _symmetric_xyz(x: float, y: float, z: float):
    return (-x, -y, -z), (x, y, z)


def _set_observation_bias(cfg, xyz_m: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> None:
    cfg.enable_policy_book_observation_bias = True
    cfg.policy_book_observation_translation_bias_min, cfg.policy_book_observation_translation_bias_max = (
        _symmetric_xyz(*xyz_m)
    )
    rpy = tuple(math.radians(value) for value in rpy_deg)
    cfg.policy_book_observation_rpy_bias_min = tuple(-value for value in rpy)
    cfg.policy_book_observation_rpy_bias_max = rpy


def _set_fixed_observation_bias(
    cfg, xyz_m: tuple[float, float, float], rpy_deg: tuple[float, float, float]
) -> None:
    cfg.enable_policy_book_observation_bias = True
    cfg.policy_book_observation_translation_bias_min = tuple(xyz_m)
    cfg.policy_book_observation_translation_bias_max = tuple(xyz_m)
    rpy = tuple(math.radians(value) for value in rpy_deg)
    cfg.policy_book_observation_rpy_bias_min = rpy
    cfg.policy_book_observation_rpy_bias_max = rpy


def _set_gripper_nuisance(cfg, deterministic_values: tuple[float, ...] = ()) -> None:
    cfg.enable_insert_gripper_observation_nuisance = True
    cfg.insert_gripper_observation_min = 0.0
    cfg.insert_gripper_observation_max = 0.45
    cfg.insert_gripper_observation_deterministic_values = deterministic_values


def _set_actual_grasp(cfg, xyz_m: tuple[float, float, float], yaw_deg: float) -> None:
    # Disable the global-step curriculum so these per-job bounds stay fixed.
    cfg.enable_residual_reset_curriculum = False
    cfg.book_grasp_translation_jitter_min, cfg.book_grasp_translation_jitter_max = _symmetric_xyz(*xyz_m)
    cfg.book_grasp_x_jitter, cfg.book_grasp_y_jitter, cfg.book_grasp_z_jitter = xyz_m
    cfg.book_grasp_yaw_jitter = math.radians(yaw_deg)


def apply_targeted_dr_profile(cfg, profile: str) -> dict[str, object]:
    """Apply one explicit DR profile and return its concise resolved contract."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown targeted DR profile: {profile!r}")
    if profile == "disabled":
        return {"profile": profile}

    # Preserve the July physical-grasp distribution unless a profile explicitly
    # requests broader actual-grasp randomization.
    _set_actual_grasp(cfg, (0.003, 0.003, 0.0015), 3.0)
    _set_gripper_nuisance(cfg)

    if profile in ("obs_moderate", "obs_moderate_curriculum", "obs_gripper_no_grasp", "combined_moderate", "combined_moderate_actuation"):
        _set_observation_bias(cfg, (0.005, 0.010, 0.004), (3.0, 3.0, 8.0))
    elif profile == "strong_tail":
        _set_observation_bias(cfg, (0.007, 0.012, 0.006), (5.0, 5.0, 12.0))

    if profile in ("combined_moderate", "combined_moderate_actuation"):
        _set_actual_grasp(cfg, (0.005, 0.008, 0.003), 8.0)
    elif profile == "obs_gripper_no_grasp":
        _set_actual_grasp(cfg, (0.0, 0.0, 0.0), 0.0)
    elif profile == "strong_tail":
        _set_actual_grasp(cfg, (0.010, 0.010, 0.005), 12.0)

    if profile == "combined_moderate_actuation":
        cfg.enable_insert_action_realization_dr = True
        cfg.insert_action_realization_scale_min = 0.65
        cfg.insert_action_realization_scale_max = 0.90

    if profile == "obs_moderate_curriculum":
        cfg.enable_targeted_dr_curriculum = True

    return {
        "profile": profile,
        "observation_translation_min_m": tuple(cfg.policy_book_observation_translation_bias_min),
        "observation_translation_max_m": tuple(cfg.policy_book_observation_translation_bias_max),
        "observation_rpy_min_rad": tuple(cfg.policy_book_observation_rpy_bias_min),
        "observation_rpy_max_rad": tuple(cfg.policy_book_observation_rpy_bias_max),
        "actual_grasp_translation_min_m": tuple(cfg.book_grasp_translation_jitter_min),
        "actual_grasp_translation_max_m": tuple(cfg.book_grasp_translation_jitter_max),
        "actual_grasp_yaw_rad": float(cfg.book_grasp_yaw_jitter),
        "insert_gripper_observation_range": (
            float(cfg.insert_gripper_observation_min),
            float(cfg.insert_gripper_observation_max),
        ),
        "insert_action_realization_scale_range": (
            float(cfg.insert_action_realization_scale_min),
            float(cfg.insert_action_realization_scale_max),
        ),
    }


def apply_evaluation_dr_profile(cfg, profile: str) -> dict[str, object]:
    """Apply one common robustness-evaluation distribution."""
    if profile not in EVALUATION_PROFILE_NAMES:
        raise ValueError(f"unknown evaluation DR profile: {profile!r}")
    if profile == "nominal":
        return {"profile": profile}

    # Keep each nuisance isolated unless the profile explicitly says combined.
    if profile == "observation_moderate":
        _set_observation_bias(cfg, (0.005, 0.010, 0.004), (3.0, 3.0, 8.0))
    elif profile == "strong_tail":
        _set_observation_bias(cfg, (0.007, 0.012, 0.006), (5.0, 5.0, 12.0))
    elif profile == "physical_like":
        # Measured release trial: the fixed TCP->book estimate was about
        # 6.7 mm lower in book Y and 5.5 deg lower in yaw than marker truth.
        _set_fixed_observation_bias(cfg, (0.0, -0.007, 0.0), (0.0, 0.0, -6.0))
        _set_gripper_nuisance(cfg, (0.388235,))
    elif profile == "gripper_only":
        _set_gripper_nuisance(cfg, (0.0, 0.1, 0.2, 0.3, 0.35, 0.388235, 0.425, 0.45))
    elif profile == "combined_moderate":
        apply_targeted_dr_profile(cfg, "combined_moderate")
    elif profile == "combined_moderate_actuation":
        apply_targeted_dr_profile(cfg, "combined_moderate_actuation")

    return {
        "profile": profile,
        "policy_book_observation_bias": bool(getattr(cfg, "enable_policy_book_observation_bias", False)),
        "insert_gripper_observation_nuisance": bool(
            getattr(cfg, "enable_insert_gripper_observation_nuisance", False)
        ),
        "insert_action_realization_dr": bool(getattr(cfg, "enable_insert_action_realization_dr", False)),
    }
