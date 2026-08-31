# Canonical xArm7 Physical Experiment

## Supported layout

Keep one Bookshelf source repository:

```text
~/Chris/bookshelf-unified/                 source of truth
  .ros2_ws/                                generated build/install/log output
~/Chris/bookshelf_xarm_sim_ws/src/xarm_ros2/  official xArm dependency (Alienware)
```

On Riot, the official xArm dependency may remain in its existing
`~/Chris/ros2_ws` underlay. It is a dependency, not another Bookshelf checkout.
Do not build Bookshelf packages into named overlays such as
`install_release_geometry` or `install_physical_episode`.

## Build once

From the repository:

```bash
cd ~/Chris/bookshelf-unified
scripts/ros2/build_xarm_experiment.sh
```

The script builds the three Bookshelf ROS packages into only
`.ros2_ws/install`. It also verifies that the full physical launch was
installed. The official xArm underlay must include `xarm_moveit_config`,
`xarm_moveit_servo`, and `xarm_planner`.

## Capture operator poses

With the robot already positioned by the operator and `/joint_states` active:

```bash
scripts/ros2/capture_operator_joint_pose.sh scan
scripts/ros2/capture_operator_joint_pose.sh loading
```

These commands only record the current joint state under
`~/BookshelfFiles/experiment_configs/operator_joint_poses/`; they never command
the robot.

## Calculate-only check

This starts the complete camera, marker, policy, logging, and hardware state
pipeline, but the coordinator creates no gripper or motion command interfaces:

```bash
cd ~/Chris/bookshelf-unified
scripts/ros2/run_xarm_experiment.sh calculate
```

Run this on the machine hosting the physical xArm and RealSense camera. A
development machine may build the canonical overlay, but the launcher exits
before starting hardware when `realsense2_camera` is unavailable.

Use this first after every hardware or camera restart. Continue only when the
marker is visible and both observation and inference validity are true.

## Authorized physical episode

Stop the calculate-only launch first. After reviewing its geometry and the
physical release boundary, run:

```bash
cd ~/Chris/bookshelf-unified

BOOKSHELF_AUTHORIZATION_TOKEN=I_APPROVE_XARM_FULL_EPISODE \
BOOKSHELF_PHYSICAL_RELEASE_BOUNDARY_CONFIRMED=true \
scripts/ros2/run_xarm_experiment.sh control
```

The control command remains fail-closed unless both explicit values are
present. Keep the emergency stop available and do not run another MoveIt or
Servo owner at the same time.

## Optional overrides

The scripts discover the reviewed files under `~/BookshelfFiles`. Override a
path only when intentionally testing another reviewed artifact:

```bash
export BOOKSHELF_APPROVED_CONFIG=/absolute/path/trial_static_slot.yaml
export BOOKSHELF_POLICY_BUNDLE=/absolute/path/policy.npz
export BOOKSHELF_ACTIVATION_ENVELOPE=/absolute/path/envelope.json
```

Other supported overrides are `BOOKSHELF_ROBOT_IP`, `BOOKSHELF_SHOW_RVIZ`,
`BOOKSHELF_RECORD_CAMERA`, `BOOKSHELF_START_IMMEDIATELY`, and
`BOOKSHELF_EXPERIMENT_OUTPUT_ROOT`.

If the official xArm install is in a nonstandard location, set a colon-separated
underlay list before building or running:

```bash
export BOOKSHELF_ROS_UNDERLAYS=/absolute/path/to/xarm/install/setup.bash
```

## Legacy cleanup

After the canonical build and calculate-only launch succeed, the old generated
policy workspace is no longer needed:

```text
~/Chris/bookshelf_xarm_policy_sim_ws/
```

Do not remove `~/Chris/bookshelf_xarm_sim_ws`; it contains the official xArm
source dependency used on Alienware.
