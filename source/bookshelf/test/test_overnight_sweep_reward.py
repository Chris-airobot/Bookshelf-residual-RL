#!/usr/bin/env python3
"""Tests for overnight sweep reward helper math."""

import importlib.util
import math
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'bookshelf'
    / 'tasks'
    / 'direct'
    / 'bookshelf'
    / 'overnight_sweep_reward.py'
)


def _load_osr():
    spec = importlib.util.spec_from_file_location(
        'overnight_sweep_reward', _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


osr = _load_osr()


def test_lat4_linear():
    """Verify LAT4 linear penalty math and symmetry."""
    scale = 4.0
    lat_err = 0.002
    penalty_pos = scale * lat_err
    penalty_neg = scale * abs(-lat_err)
    assert penalty_pos == pytest.approx(0.008)
    assert penalty_pos == pytest.approx(penalty_neg)


def test_quadratic_lat_penalty():
    """Verify quadratic penalty at 2mm, 0.5mm, and negative error."""
    coef = 2000.0
    assert osr.quadratic_lat_penalty(0.002, coef) == pytest.approx(0.008)
    assert osr.quadratic_lat_penalty(0.0005, coef) == pytest.approx(
        0.0005, rel=1e-3
    )
    assert osr.quadratic_lat_penalty(-0.002, coef) == pytest.approx(0.008)


def test_wall_margin_penalty_inside_and_outside():
    """Verify wall penalty is 0 inside margin and positive beyond."""
    inner_half = 0.010
    margin = 0.0005
    scale = 20.0
    # Threshold is inner_half - margin = 0.0095
    # Inside margin (0.009 <= 0.0095) -> 0
    assert osr.wall_margin_penalty(0.009, inner_half, margin, scale) == 0.0

    # Beyond margin: (0.0115 > 0.0095)
    # -> 20 * (0.0115 - 0.0095) = 20 * 0.002 = 0.04
    expected = scale * (0.0115 - (inner_half - margin))
    assert osr.wall_margin_penalty(
        0.0115, inner_half, margin, scale
    ) == pytest.approx(expected)
    assert osr.wall_margin_penalty(
        0.0115, inner_half, margin, scale
    ) == pytest.approx(0.04)


def test_wall_extent_responds_to_yaw():
    """Verify book corner lat_extent responds to yaw and center-Y."""
    torch = pytest.importorskip('torch', exc_type=ImportError)

    # Book box dimensions: length=0.20, width=0.03, height=0.25
    lx, ly, lz = 0.20, 0.03, 0.25
    corners_local = torch.tensor([
        [-lx / 2, -ly / 2, -lz / 2],
        [-lx / 2, -ly / 2, lz / 2],
        [-lx / 2, ly / 2, -lz / 2],
        [-lx / 2, ly / 2, lz / 2],
        [lx / 2, -ly / 2, -lz / 2],
        [lx / 2, -ly / 2, lz / 2],
        [lx / 2, ly / 2, -lz / 2],
        [lx / 2, ly / 2, lz / 2],
    ], dtype=torch.float32)

    cy = 0.0

    def compute_lat_extent(y_offset, yaw_rad):
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # Rotated y coordinates
        corner_y = (
            corners_local[:, 0] * sin_y
            + corners_local[:, 1] * cos_y
            + y_offset
        )
        return torch.abs(corner_y - cy).max().item()

    ext_yaw0 = compute_lat_extent(0.0, 0.0)
    ext_yaw5 = compute_lat_extent(0.0, math.radians(5.0))
    ext_yaw10 = compute_lat_extent(0.0, math.radians(10.0))
    assert ext_yaw5 > ext_yaw0
    assert ext_yaw10 > ext_yaw5

    # Center-Y offset at fixed yaw
    ext_y_offset = compute_lat_extent(0.005, math.radians(5.0))
    assert ext_y_offset > ext_yaw5


def test_depth_gate():
    """Verify depth gate values and monotonicity."""
    range_m = 0.08
    floor = 0.25
    assert osr.wall_depth_gate(0.20, range_m, floor) == pytest.approx(0.25)
    assert osr.wall_depth_gate(0.0, range_m, floor) == pytest.approx(1.0)
    assert osr.wall_depth_gate(-0.05, range_m, floor) == pytest.approx(1.0)

    d_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    gates = [osr.wall_depth_gate(d, range_m, floor) for d in d_values]
    for i in range(len(gates) - 1):
        assert gates[i] >= gates[i + 1]


def test_action_saturation_penalty():
    """Verify pre-clamp saturation penalty on 5 motion dims.

    Release dimension must be excluded.
    """
    torch = pytest.importorskip('torch', exc_type=ImportError)

    coef = 0.001
    # 6 actions: 5 motion + 1 release
    # Case 1: all within [-1, 1]
    raw_valid = torch.zeros((1, 6), dtype=torch.float32)
    raw_valid[0, :5] = torch.tensor([0.5, -0.9, 1.0, -0.2, 0.0])
    raw_valid[0, 5] = 5.0  # huge release dim
    pen1 = osr.saturation_penalty(raw_valid[:, :5], coef)
    assert pen1.item() == 0.0

    # Case 2: one motion dim = 2.0 -> hinge = (2-1)**2 = 1, mean over 5 = 0.2
    # penalty = 0.001 * 0.2 = 2e-4
    raw_saturated = torch.zeros((1, 6), dtype=torch.float32)
    raw_saturated[0, 0] = 2.0
    raw_saturated[0, 5] = 99.0  # release dim must be excluded
    pen2 = osr.saturation_penalty(raw_saturated[:, :5], coef)
    assert pen2.item() == pytest.approx(2e-4)


def test_ramp_fraction():
    """Verify curriculum ramp fraction behavior."""
    assert osr.ramp_fraction(0, 2_000_000) == 0.0
    assert osr.ramp_fraction(1_000_000, 2_000_000) == 0.5
    assert osr.ramp_fraction(2_000_000, 2_000_000) == 1.0
    assert osr.ramp_fraction(9_999_999, 2_000_000) == 1.0
    assert osr.ramp_fraction(5, 0) == 1.0
    assert osr.ramp_fraction(5, -100) == 1.0
    assert osr.ramp_fraction(5, None) == 1.0
