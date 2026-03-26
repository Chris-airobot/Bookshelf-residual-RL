# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Bookshelf-Direct-v0",
    entry_point=f"{__name__}.bookshelf_env_v0:BookshelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bookshelf_env_cfg_v0:BookshelfEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Bookshelf-Direct-v1",
    entry_point=f"{__name__}.bookshelf_env_v1:BookshelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bookshelf_env_cfg_v1:BookshelfEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bookshelf-Direct-v2",
    entry_point=f"{__name__}.bookshelf_env_v2:BookshelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bookshelf_env_cfg_v2:BookshelfEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bookshelf-Direct-v3",
    entry_point=f"{__name__}.bookshelf_env_v3:BookshelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bookshelf_env_cfg_v3:BookshelfEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Bookshelf-Direct-v4",
    entry_point=f"{__name__}.bookshelf_env_v4:BookshelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bookshelf_env_cfg_v4:BookshelfEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# Insertion-only v4: robot keeps grasping, geometry-only success at slot mouth.
gym.register(
    id="Bookshelf-Direct-v4-insert-only",
    entry_point=f"{__name__}.bookshelf_env_book_only:BookshelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bookshelf_env_cfg_book_only:BookshelfEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)