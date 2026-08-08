#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO-only comparison config for Bookshelf-PPO-Direct-v0.

This uses the same residual-task environment, object geometry, reset-noise
curriculum, clearance, PPO config, and evaluation path as the residual method,
but disables the nominal geometric controller. The learned PPO action therefore
acts as the full local Cartesian command instead of a residual correction.
"""

from isaaclab.utils import configclass

from .bookshelf_residual_env_cfg import BookshelfEnvCfg as ResidualBookshelfEnvCfg


@configclass
class BookshelfEnvCfg(ResidualBookshelfEnvCfg):
    """PPO-only baseline: same task settings, no nominal insertion controller."""

    enable_nominal_controller = False

