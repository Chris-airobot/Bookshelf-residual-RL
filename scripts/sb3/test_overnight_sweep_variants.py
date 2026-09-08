#!/usr/bin/env python3
"""Tests for overnight RL sweep variant configuration mapping."""

import copy
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
    )


def test_lat4_variant():
    """Verify lat4 variant scales lateral penalty to 4.0."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'lat4')
    assert cfg.insert_lat_penalty_scale == 4.0
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert not cfg.insert_lat_penalty_quadratic
    assert not cfg.insert_wall_margin_penalty_enable
    assert not cfg.insert_action_saturation_penalty_enable
    assert cfg.insert_variant_ramp_transitions == 0
    assert res['variant'] == 'lat4'
    assert res['mode'] == 'finetune'
    assert res['insert_lat_penalty_scale'] == 4.0


def test_lat4_yaw075_variant():
    """Verify lat4_yaw075 variant scales lateral to 4.0 and yaw to 0.75."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'lat4_yaw075')
    assert cfg.insert_lat_penalty_scale == 4.0
    assert cfg.insert_yaw_penalty_scale == 0.75
    assert not cfg.insert_lat_penalty_quadratic
    assert not cfg.insert_wall_margin_penalty_enable
    assert not cfg.insert_action_saturation_penalty_enable
    assert cfg.insert_variant_ramp_transitions == 0
    assert res['mode'] == 'finetune'


def test_quad_match_variant():
    """Verify quad_match variant replaces linear with quadratic lateral."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'quad_match')
    assert cfg.insert_lat_penalty_scale == 0.0
    assert cfg.insert_lat_penalty_quadratic is True
    assert cfg.insert_lat_quadratic_coef_per_m2 == 2000.0
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert cfg.insert_variant_ramp_transitions == 0
    assert res['insert_lat_penalty_quadratic'] is True
    assert res['insert_lat_quadratic_coef_per_m2'] == 2000.0


def test_wall05_variant():
    """Verify wall05 variant enables wall margin penalty with 2M ramp."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'wall05')
    assert cfg.insert_wall_margin_penalty_enable is True
    assert cfg.insert_wall_margin_penalty_scale_per_m == 20.0
    assert cfg.insert_wall_margin_m == 0.0005
    assert cfg.insert_variant_ramp_transitions == 2_000_000
    assert cfg.insert_wall_margin_depth_gate_enable is False
    assert cfg.insert_lat_penalty_scale == 0.8
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert res['insert_wall_margin_penalty_enable'] is True
    assert res['insert_wall_margin_m'] == 0.0005


def test_wall_depth_variant():
    """Verify wall_depth variant adds depth gate to wall margin penalty."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'wall_depth')
    assert cfg.insert_wall_margin_penalty_enable is True
    assert cfg.insert_wall_margin_penalty_scale_per_m == 20.0
    assert cfg.insert_wall_margin_m == 0.0005
    assert cfg.insert_wall_margin_depth_gate_enable is True
    assert cfg.insert_wall_depth_gate_range_m == 0.08
    assert cfg.insert_wall_depth_gate_floor == 0.25
    assert cfg.insert_variant_ramp_transitions == 2_000_000
    assert cfg.insert_lat_penalty_scale == 0.8
    assert cfg.insert_yaw_penalty_scale == 0.5
    assert res['insert_wall_margin_depth_gate_enable'] is True


def test_lat4_sat001_variant():
    """Verify lat4_sat001 enables saturation penalty on 5 motion dims."""
    cfg = _make_july_cfg()
    res = apply_overnight_variant(cfg, 'lat4_sat001')
    assert cfg.insert_lat_penalty_scale == 4.0
    assert cfg.insert_action_saturation_penalty_enable is True
    assert cfg.insert_action_saturation_coef == 0.001
    assert cfg.insert_variant_ramp_transitions == 2_000_000
    assert cfg.insert_wall_margin_penalty_enable is False
    assert res['insert_action_saturation_penalty_enable'] is True
    assert res['insert_action_saturation_coef'] == 0.001


def test_sc_lat4_variant():
    """Verify sc_lat4 has identical field effect to lat4 with scratch mode."""
    cfg = _make_july_cfg()
    ref_cfg = _make_july_cfg()
    res_sc = apply_overnight_variant(cfg, 'sc_lat4')
    res_lat4 = apply_overnight_variant(ref_cfg, 'lat4')
    assert vars(cfg) == vars(ref_cfg)
    assert res_sc['mode'] == 'scratch'
    assert res_lat4['mode'] == 'finetune'
    assert res_sc['insert_lat_penalty_scale'] == 4.0


def test_unknown_variant_raises():
    """Verify unknown variant raises ValueError."""
    cfg = _make_july_cfg()
    with pytest.raises(ValueError, match='unknown overnight sweep variant'):
        apply_overnight_variant(cfg, 'non_existent_variant')


def test_july_variant_mutates_nothing():
    """Verify july variant leaves configuration untouched."""
    cfg = _make_july_cfg()
    expected = copy.deepcopy(vars(cfg))
    res = apply_overnight_variant(cfg, 'july')
    assert vars(cfg) == expected
    assert res['variant'] == 'july'
    assert res['mode'] == 'finetune'


def test_mode_map():
    """Verify MODE mapping for scratch vs finetune variants."""
    assert MODE['sc_lat4'] == 'scratch'
    finetune_variants = [k for k, v in MODE.items() if v == 'finetune']
    assert len(finetune_variants) == 7
    batch1_finetune = (
        'july',
        'lat4',
        'lat4_yaw075',
        'quad_match',
        'wall05',
        'wall_depth',
        'lat4_sat001',
    )
    for name in batch1_finetune:
        assert MODE[name] == 'finetune'
