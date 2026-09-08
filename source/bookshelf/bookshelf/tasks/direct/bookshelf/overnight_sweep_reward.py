#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure helper functions for overnight RL sweep reward variants."""

from __future__ import annotations


def _relu(x):
    """Rectified linear unit supporting torch tensors and scalar floats."""
    c = getattr(x, 'clamp', None)
    return c(min=0.0) if c is not None else (x if x > 0.0 else 0.0 * x)


def _clip01(x):
    """Clip between 0.0 and 1.0 supporting torch tensors and scalar floats."""
    c = getattr(x, 'clamp', None)
    return c(0.0, 1.0) if c is not None else min(1.0, max(0.0, x))


def quadratic_lat_penalty(lat_err_m, coef_per_m2):
    """Compute symmetric quadratic lateral error penalty in metres.

    J4: symmetric. lat_err_m in METRES. penalty = coef_per_m2 * lat_err_m**2.
    """
    return coef_per_m2 * (lat_err_m * lat_err_m)


def wall_margin_excess(lat_extent_m, inner_half_m, wall_margin_m):
    """Compute lateral extent excess beyond shelf inner half margin in metres.

    relu(lat_extent - (inner_half - margin)) in metres.
    """
    return _relu(lat_extent_m - (inner_half_m - wall_margin_m))


def wall_margin_penalty(
    lat_extent_m, inner_half_m, wall_margin_m, wall_scale_per_m
):
    """Compute wall-margin penalty.

    J2 base: metres * (per-metre scale) -> dimensionless per-step penalty.
    """
    excess = wall_margin_excess(lat_extent_m, inner_half_m, wall_margin_m)
    return wall_scale_per_m * excess


def wall_depth_gate(d_m, range_m=0.08, floor=0.25):
    """Compute depth gate multiplier for wall margin penalty.

    J7: g(d) = floor + (1-floor) * clip(1 - d/range, 0, 1).
    d = mouth_x - front_x (leading edge). far before mouth -> floor;
    at/after entry -> 1.0; monotonic.
    """
    return floor + (1.0 - floor) * _clip01(1.0 - d_m / range_m)


def saturation_penalty(raw_motion_actions, coef):
    """Compute penalty for pre-clamp action saturation across 5 motion dims.

    J6: coef * mean_over_dims( relu(|a_raw| - 1)**2 ).
    raw_motion_actions = 5 motion dims, PRE-CLAMP.
    torch: shape (N,5) -> (N,). Release dim must NOT be included by caller.
    """
    a = raw_motion_actions
    hinge = _relu(a.abs() - 1.0)
    return coef * (hinge * hinge).mean(dim=-1)


def ramp_fraction(aggregate_transitions, ramp_transitions):
    """Compute curriculum ramp multiplier in [0.0, 1.0].

    ramp_transitions <= 0 -> always 1.0.
    """
    if ramp_transitions is None or ramp_transitions <= 0:
        return 1.0
    return _clip01(float(aggregate_transitions) / float(ramp_transitions))
