# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget


def open_gripper(robot, world):
    """Set gripper to open position (must be called after world.reset())."""
    gripper_open_position = 0.04
    dof_names = getattr(robot, "dof_names", None)
    if dof_names is None:
        print("Warning: dof_names is None, cannot open gripper")
        return
    
    finger_joint_indices = []
    for i, name in enumerate(dof_names):
        if "finger" in name.lower():
            finger_joint_indices.append(i)
    
    if len(finger_joint_indices) >= 2:
        # Set both fingers to open position
        finger_positions = np.array([gripper_open_position] * len(finger_joint_indices))
        robot.set_joint_positions(finger_positions, finger_joint_indices)
        world.step(render=False)
        print(f"✓ Opened gripper (finger positions: {finger_positions})")
    else:
        print(f"⚠ Could not find finger joints. Found {len(finger_joint_indices)} finger joints.")


my_world = World(stage_units_in_meters=1.0)
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
open_gripper(my_franka, my_world)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()
reset_needed = False
frame = 0
dof_names = getattr(my_franka, "dof_names", None)  # 方便把数值对应到关节名
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            open_gripper(my_franka, my_world)
            reset_needed = False
        q = my_franka.get_joint_positions()  # np.ndarray
        if frame % 30 == 0:  # 每 30 帧打印一次，避免刷屏
            if dof_names is not None:
                print({n: float(v) for n, v in zip(dof_names, q)})
            else:
                print(q)
        observations = my_world.get_observations()
        actions, succ = my_controller.compute_inverse_kinematics(
            target_position=observations[target_name]["position"],
            target_orientation=observations[target_name]["orientation"],
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")

simulation_app.close()