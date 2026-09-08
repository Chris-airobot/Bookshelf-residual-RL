#!/usr/bin/env python3
"""Tests for batch 2 (K1-K8) overnight RL sweep variant configuration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VARIANTS_PATH = _REPO_ROOT / 'scripts' / 'sb3' / 'overnight_sweep_variants.py'


def _load_variants():
    spec = importlib.util.spec_from_file_location(
        'overnight_sweep_variants', _VARIANTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m = _load_variants()
VARIANTS = _m.VARIANTS
SCRATCH_VARIANTS = _m.SCRATCH_VARIANTS
MODE = _m.MODE
apply_overnight_variant = _m.apply_overnight_variant


def _make_july_cfg() -> SimpleNamespace:
    """Return dummy configuration initialized to July defaults."""
    return SimpleNamespace(
        insert_lat_penalty_scale=0.8,
        insert_yaw_penalty_scale=0.5,
        insert_lat_penalty_quadratic=False,
        insert_lat_quadratic_coef_per_m2=0.0,
        insert_wall_margin_penalty_enable=False,
        insert_wall_margin_penalty_scale_per_m=0.0,
        insert_wall_margin_m=0.0,
        insert_wall_margin_depth_gate_enable=False,
        insert_wall_depth_gate_range_m=0.08,
        insert_wall_depth_gate_floor=0.25,
        insert_action_saturation_penalty_enable=False,
        insert_action_saturation_coef=0.0,
        insert_variant_ramp_transitions=0,
        slot_lateral_clearance_min=0.0030,
        slot_lateral_clearance_max=0.0030,
    )


def test_k1_vs_k5_sc_lat4():
    """Verify K1 and K5 reuse sc_lat4 with identical effect and mode."""
    # Seed difference (2026 vs 123) is a submit-time argument, not in variant.
    cfg_k1 = _make_july_cfg()
    cfg_k5 = _make_july_cfg()
    res_k1 = apply_overnight_variant(cfg_k1, 'sc_lat4')
    res_k5 = apply_overnight_variant(cfg_k5, 'sc_lat4')
    assert vars(cfg_k1) == vars(cfg_k5)
    assert res_k1 == res_k5
    assert res_k1['mode'] == 'scratch'
    assert MODE['sc_lat4'] == 'scratch'


def test_k1_vs_k2():
    """Verify K2 adds yaw penalty scale 0.75 relative to K1."""
    cfg_k1 = _make_july_cfg()
    cfg_k2 = _make_july_cfg()
    apply_overnight_variant(cfg_k1, 'sc_lat4')
    res_k2 = apply_overnight_variant(cfg_k2, 'sc_lat4_yaw075')

    assert cfg_k1.insert_lat_penalty_scale == 4.0
    assert cfg_k1.insert_yaw_penalty_scale == 0.5

    assert cfg_k2.insert_lat_penalty_scale == 4.0
    assert cfg_k2.insert_yaw_penalty_scale == 0.75

    diff_fields = {
        k for k in vars(cfg_k1) if getattr(cfg_k1, k) != getattr(cfg_k2, k)
    }
    assert diff_fields == {'insert_yaw_penalty_scale'}
    assert res_k2['mode'] == 'scratch'


def test_sc_wall05_variant():
    """Verify sc_wall05 enables wall penalty without ramp or depth gate."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'sc_wall05')

    assert cfg.insert_wall_margin_penalty_enable is True
    assert cfg.insert_wall_margin_penalty_scale_per_m == 20.0
    assert cfg.insert_wall_margin_m == 0.0005
    assert cfg.insert_variant_ramp_transitions == 0
    assert cfg.insert_wall_margin_depth_gate_enable is False
    assert cfg.insert_lat_penalty_scale == 0.8
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert res['mode'] == 'scratch'


def test_sc_quad_match_variant():
    """Verify sc_quad_match replaces linear with quadratic lateral penalty."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'sc_quad_match')

    assert cfg.insert_lat_penalty_scale == 0.0
    assert cfg.insert_lat_penalty_quadratic is True
    assert cfg.insert_lat_quadratic_coef_per_m2 == 2000.0
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert cfg.insert_wall_margin_penalty_enable is False
    assert cfg.insert_action_saturation_penalty_enable is False
    assert res['mode'] == 'scratch'

    # Numeric equivalence documentation
    assert 2000.0 * (0.002**2) == pytest.approx(0.008)
    assert 2000.0 * (0.0005**2) == pytest.approx(0.0005)


def test_sc_lat2_softwall_dr_variant():
    """Verify sc_lat2_softwall_dr sets softer wall, gate, and clearance."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'sc_lat2_softwall_dr')

    assert cfg.insert_lat_penalty_scale == 2.0
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert cfg.insert_wall_margin_penalty_enable is True
    assert cfg.insert_wall_margin_penalty_scale_per_m == 10.0
    assert cfg.insert_wall_margin_m == 0.0005
    assert cfg.insert_wall_margin_depth_gate_enable is True
    assert cfg.insert_wall_depth_gate_range_m == 0.08
    assert cfg.insert_wall_depth_gate_floor == 0.25
    assert cfg.insert_variant_ramp_transitions == 0
    assert cfg.slot_lateral_clearance_min == 0.0030
    assert cfg.slot_lateral_clearance_max == 0.0045

    assert res['slot_lateral_clearance_min'] == 0.0030
    assert res['slot_lateral_clearance_max'] == 0.0045
    assert res['mode'] == 'scratch'


def test_sc_wall05_ramp_vs_sc_wall05():
    """Verify sc_wall05_ramp matches sc_wall05 except for 5M transitions."""
    cfg_ramp = _make_july_cfg()
    cfg_noramp = _make_july_cfg()
    res_ramp = apply_overnight_variant(cfg_ramp, 'sc_wall05_ramp')
    res_noramp = apply_overnight_variant(cfg_noramp, 'sc_wall05')

    assert cfg_ramp.insert_wall_margin_penalty_enable is True
    assert cfg_ramp.insert_wall_margin_penalty_scale_per_m == 20.0
    assert cfg_ramp.insert_wall_margin_m == 0.0005
    assert cfg_ramp.insert_wall_margin_depth_gate_enable is False
    assert cfg_ramp.insert_lat_penalty_scale == 0.8
    assert cfg_ramp.insert_yaw_penalty_scale == 0.5
    assert cfg_ramp.insert_variant_ramp_transitions == 5_000_000

    assert cfg_noramp.insert_variant_ramp_transitions == 0

    diff_fields = {
        k for k in res_ramp
        if k != 'variant' and res_ramp[k] != res_noramp[k]
    }
    assert diff_fields == {'insert_variant_ramp_transitions'}


def test_sc_lat4_sat001_variant():
    """Verify sc_lat4_sat001 sets lat 4.0 and sat coef 0.001 with no ramp."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'sc_lat4_sat001')

    assert cfg.insert_lat_penalty_scale == 4.0
    assert cfg.insert_action_saturation_penalty_enable is True
    assert cfg.insert_action_saturation_coef == 0.001
    assert cfg.insert_variant_ramp_transitions == 0
    assert cfg.insert_wall_margin_penalty_enable is False
    assert res['mode'] == 'scratch'


def test_scratch_variants_mode_is_scratch():
    """Verify MODE mapping is scratch for all 7 batch 2 scratch variants."""
    expected_scratch = (
        'sc_lat4',
        'sc_lat4_yaw075',
        'sc_wall05',
        'sc_quad_match',
        'sc_lat2_softwall_dr',
        'sc_wall05_ramp',
        'sc_lat4_sat001',
    )
    for v in expected_scratch:
        assert MODE[v] == 'scratch', f'Expected scratch mode for {v}'
        assert v in SCRATCH_VARIANTS


def test_unknown_variant_raises():
    """Verify unknown variant name raises ValueError."""
    cfg = _make_july_cfg()
    with pytest.raises(ValueError, match='unknown overnight sweep variant'):
        apply_overnight_variant(cfg, 'sc_non_existent')


def test_j1_to_j8_variants_unchanged():
    """Verify apply_overnight_variant regression for all batch 1 variants."""
    j1_to_j8_expected = {
        'july': {
            'variant': 'july',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 0.8,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': False,
            'insert_wall_margin_penalty_scale_per_m': 0.0,
            'insert_wall_margin_m': 0.0,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 0,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'lat4': {
            'variant': 'lat4',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 4.0,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': False,
            'insert_wall_margin_penalty_scale_per_m': 0.0,
            'insert_wall_margin_m': 0.0,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 0,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'lat4_yaw075': {
            'variant': 'lat4_yaw075',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 4.0,
            'insert_yaw_penalty_scale': 0.75,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': False,
            'insert_wall_margin_penalty_scale_per_m': 0.0,
            'insert_wall_margin_m': 0.0,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 0,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'quad_match': {
            'variant': 'quad_match',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 0.0,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': True,
            'insert_lat_quadratic_coef_per_m2': 2000.0,
            'insert_wall_margin_penalty_enable': False,
            'insert_wall_margin_penalty_scale_per_m': 0.0,
            'insert_wall_margin_m': 0.0,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 0,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'wall05': {
            'variant': 'wall05',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 0.8,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': True,
            'insert_wall_margin_penalty_scale_per_m': 20.0,
            'insert_wall_margin_m': 0.0005,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 2_000_000,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'wall_depth': {
            'variant': 'wall_depth',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 0.8,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': True,
            'insert_wall_margin_penalty_scale_per_m': 20.0,
            'insert_wall_margin_m': 0.0005,
            'insert_wall_margin_depth_gate_enable': True,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 2_000_000,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'lat4_sat001': {
            'variant': 'lat4_sat001',
            'mode': 'finetune',
            'insert_lat_penalty_scale': 4.0,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': False,
            'insert_wall_margin_penalty_scale_per_m': 0.0,
            'insert_wall_margin_m': 0.0,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': True,
            'insert_action_saturation_coef': 0.001,
            'insert_variant_ramp_transitions': 2_000_000,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
        'sc_lat4': {
            'variant': 'sc_lat4',
            'mode': 'scratch',
            'insert_lat_penalty_scale': 4.0,
            'insert_yaw_penalty_scale': 0.5,
            'insert_lat_penalty_quadratic': False,
            'insert_lat_quadratic_coef_per_m2': 0.0,
            'insert_wall_margin_penalty_enable': False,
            'insert_wall_margin_penalty_scale_per_m': 0.0,
            'insert_wall_margin_m': 0.0,
            'insert_wall_margin_depth_gate_enable': False,
            'insert_wall_depth_gate_range_m': 0.08,
            'insert_wall_depth_gate_floor': 0.25,
            'insert_action_saturation_penalty_enable': False,
            'insert_action_saturation_coef': 0.0,
            'insert_variant_ramp_transitions': 0,
            'slot_lateral_clearance_min': 0.003,
            'slot_lateral_clearance_max': 0.003,
        },
    }

    for name, expected in j1_to_j8_expected.items():
        cfg = _make_july_cfg()
        res = apply_overnight_variant(cfg, name)
        assert res == expected, (
            f'Regression in variant {name}: {res} != {expected}'
        )
        assert MODE[name] == expected['mode']
