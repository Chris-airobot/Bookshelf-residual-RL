#!/usr/bin/env python3
"""Overnight RL training sweep variant definitions and application helper."""

from __future__ import annotations

VARIANTS = (
    'july',
    'lat4',
    'lat4_yaw075',
    'quad_match',
    'wall05',
    'wall_depth',
    'lat4_sat001',
    'sc_lat4',
)

SCRATCH_VARIANTS = ('sc_lat4',)

MODE = {
    name: ('scratch' if name in SCRATCH_VARIANTS else 'finetune')
    for name in VARIANTS
}


def apply_overnight_variant(env_cfg, variant: str) -> dict[str, object]:
    """Apply one overnight sweep variant to env_cfg and return its summary."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown overnight sweep variant: {variant!r}')

    if variant == 'july':
        pass
    elif variant in ('lat4', 'sc_lat4'):
        env_cfg.insert_lat_penalty_scale = 4.0
    elif variant == 'lat4_yaw075':
        env_cfg.insert_lat_penalty_scale = 4.0
        env_cfg.insert_yaw_penalty_scale = 0.75
    elif variant == 'quad_match':
        env_cfg.insert_lat_penalty_scale = 0.0
        env_cfg.insert_lat_penalty_quadratic = True
        env_cfg.insert_lat_quadratic_coef_per_m2 = 2000.0
    elif variant == 'wall05':
        env_cfg.insert_wall_margin_penalty_enable = True
        env_cfg.insert_wall_margin_penalty_scale_per_m = 20.0
        env_cfg.insert_wall_margin_m = 0.0005
        env_cfg.insert_variant_ramp_transitions = 2_000_000
    elif variant == 'wall_depth':
        env_cfg.insert_wall_margin_penalty_enable = True
        env_cfg.insert_wall_margin_penalty_scale_per_m = 20.0
        env_cfg.insert_wall_margin_m = 0.0005
        env_cfg.insert_wall_margin_depth_gate_enable = True
        env_cfg.insert_wall_depth_gate_range_m = 0.08
        env_cfg.insert_wall_depth_gate_floor = 0.25
        env_cfg.insert_variant_ramp_transitions = 2_000_000
    elif variant == 'lat4_sat001':
        env_cfg.insert_lat_penalty_scale = 4.0
        env_cfg.insert_action_saturation_penalty_enable = True
        env_cfg.insert_action_saturation_coef = 0.001
        env_cfg.insert_variant_ramp_transitions = 2_000_000

    return {
        'variant': variant,
        'mode': MODE[variant],
        'insert_lat_penalty_scale': float(
            getattr(env_cfg, 'insert_lat_penalty_scale', 0.8)
        ),
        'insert_yaw_penalty_scale': float(
            getattr(env_cfg, 'insert_yaw_penalty_scale', 0.5)
        ),
        'insert_lat_penalty_quadratic': bool(
            getattr(env_cfg, 'insert_lat_penalty_quadratic', False)
        ),
        'insert_lat_quadratic_coef_per_m2': float(
            getattr(env_cfg, 'insert_lat_quadratic_coef_per_m2', 0.0)
        ),
        'insert_wall_margin_penalty_enable': bool(
            getattr(env_cfg, 'insert_wall_margin_penalty_enable', False)
        ),
        'insert_wall_margin_penalty_scale_per_m': float(
            getattr(env_cfg, 'insert_wall_margin_penalty_scale_per_m', 0.0)
        ),
        'insert_wall_margin_m': float(
            getattr(env_cfg, 'insert_wall_margin_m', 0.0)
        ),
        'insert_wall_margin_depth_gate_enable': bool(
            getattr(env_cfg, 'insert_wall_margin_depth_gate_enable', False)
        ),
        'insert_wall_depth_gate_range_m': float(
            getattr(env_cfg, 'insert_wall_depth_gate_range_m', 0.08)
        ),
        'insert_wall_depth_gate_floor': float(
            getattr(env_cfg, 'insert_wall_depth_gate_floor', 0.25)
        ),
        'insert_action_saturation_penalty_enable': bool(
            getattr(env_cfg, 'insert_action_saturation_penalty_enable', False)
        ),
        'insert_action_saturation_coef': float(
            getattr(env_cfg, 'insert_action_saturation_coef', 0.0)
        ),
        'insert_variant_ramp_transitions': int(
            getattr(env_cfg, 'insert_variant_ramp_transitions', 0)
        ),
    }
