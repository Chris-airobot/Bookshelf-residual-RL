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


INITIAL_JOINT_POSITIONS = {
    "panda_joint1": 1.7190992832183838,
    "panda_joint2": 1.5082764625549316,
    "panda_joint3": -1.2342585325241089,
    "panda_joint4": -2.7785983085632324,
    "panda_joint5": -2.776813268661499,
    "panda_joint6": 2.04762601852417,
    "panda_joint7": -0.9243860244750977,
    "panda_finger_joint1": 0.0030379709787666798,
    "panda_finger_joint2": 0.0035099773667752743,
}

def _initial_joint_arrays(robot):
    """Return indices and values for the named joints in INITIAL_JOINT_POSITIONS."""
    dof_names = getattr(robot, "dof_names", None)
    if dof_names is None:
        raise RuntimeError("dof_names is None, cannot set initial joint positions")

    name_to_index = {name: i for i, name in enumerate(dof_names)}
    joint_indices = []
    joint_positions = []
    missing_names = []
    for name, value in INITIAL_JOINT_POSITIONS.items():
        joint_idx = name_to_index.get(name)
        if joint_idx is None:
            missing_names.append(name)
            continue
        joint_indices.append(joint_idx)
        joint_positions.append(value)

    if missing_names:
        raise RuntimeError(f"Missing joint names: {missing_names}")

    joint_indices = np.array(joint_indices, dtype=np.int32)
    joint_positions = np.array(joint_positions, dtype=np.float32)
    return dof_names, name_to_index, joint_indices, joint_positions


def hold_initial_joint_positions(robot):
    """Force the robot state and targets to the provided joint configuration."""
    dof_names, name_to_index, joint_indices, joint_positions = _initial_joint_arrays(robot)
    robot.set_joint_positions(joint_positions, joint_indices)
    if hasattr(robot, "set_joint_position_targets"):
        robot.set_joint_position_targets(joint_positions, joint_indices)
    if hasattr(robot, "set_joint_velocities"):
        robot.set_joint_velocities(np.zeros_like(joint_positions), joint_indices)
    return dof_names, name_to_index


def set_initial_joint_positions(robot):
    """Set robot joints to the provided initial configuration after reset."""
    dof_names, name_to_index = hold_initial_joint_positions(robot)
    q = robot.get_joint_positions()
    print("Initialized joints:")
    print({name: float(q[name_to_index[name]]) for name in INITIAL_JOINT_POSITIONS if name in name_to_index})


def move_target_to_end_effector(robot, target, world):
    """Move the draggable cuboid to the current end-effector pose."""
    hold_initial_joint_positions(robot)
    world.step(render=False)
    ee_position, ee_orientation = robot.end_effector.get_world_pose()
    target.set_world_pose(position=ee_position, orientation=ee_orientation)
    world.step(render=False)
    hold_initial_joint_positions(robot)
    print("Moved target cuboid to end-effector pose:")
    print({"position": ee_position.tolist(), "orientation": ee_orientation.tolist()})


def initialize_robot_and_target(robot, target, world):
    """Force the requested robot pose, then align the cuboid to that pose."""
    set_initial_joint_positions(robot)
    move_target_to_end_effector(robot, target, world)


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
my_target = my_world.scene.get_object(target_name)
initialize_robot_and_target(my_franka, my_target, my_world)
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
            initialize_robot_and_target(my_franka, my_target, my_world)
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
        frame += 1

simulation_app.close()