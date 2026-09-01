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

The script builds the five Bookshelf ROS packages used by the workflow into only
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

## Complete operator sequence

The loading pose is also the hold/return pose. The complete keyboard sequence
is:

```text
G  verify/preview the saved scan joint trajectory; inspect, then E executes
S  freeze the detected slot
L  verify/preview the loading/hold joint trajectory; inspect, then E executes
O  open the gripper
C  close the gripper after the operator loads the book
P  calculate and plan preinsert
E  execute the reviewed preinsert trajectory
I  start PPO INSERT -> release -> retreat -> close empty -> PUSH
H  after PUSH, verify/preview return-to-loading; E executes and opens
Q  quit the operator console
```

Every key is state-gated. PUSH ends in
`PUSH_COMPLETE_WAITING_RETURN`; there is no automatic return. `H` is accepted
only in that state. A failed return does not open the gripper and does not enter
`READY_FOR_NEXT_BOOK`.

At startup, and after `READY_FOR_NEXT_BOOK`, press `G`, inspect its displayed
collision-checked direct joint trajectory, then press `E`. Slot freeze `S` is
accepted only after that reviewed motion reaches `SCAN`. `G`, `L`, and `H`
preview the exact direct controller trajectory; `P` retains robust IK and
MoveIt planning. None execute by themselves.

## Alienware full fake-hardware rehearsal

This uses the recorded RGB-D bag, unchanged slot detector, official fake xArm7
MoveIt/gripper, reviewed preinsert planner, and existing PPO/post-insert
controller. It starts no physical hardware:

```bash
cd ~/Chris/bookshelf-unified
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source .ros2_ws/install/local_setup.bash
ros2 launch bookshelf_simple_experiment_ros \
  offline_full_sequence_rehearsal.launch.py
```

The captured scan and loading files must exist under
`~/BookshelfFiles/experiment_configs/operator_joint_poses/` on Alienware.

## Riot full-sequence shadow

This starts the real xArm connection, live joint states/TF, RealSense,
perception, planning, and PPO calculations. The shadow override prevents the
preinsert, loading/return, gripper, Servo-start, and Twist command owners from
creating or using actuator interfaces:

```bash
cd ~/Chris/bookshelf-unified
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source ~/Chris/ros2_ws/install/setup.bash
source .ros2_ws/install/local_setup.bash
ros2 launch bookshelf_simple_experiment_ros \
  real_experiment_operator.launch.py \
  allow_execution:=true shadow_full_sequence:=true
```

The stationary robot state remains authoritative. Later phases are explicitly
labelled logical shadow transitions; no replacement joint states or TF are
published.

## Authorized physical episode

Stop the shadow or calculate-only launch first. After reviewing its geometry
and the physical release boundary, start the complete state-gated workflow:

```bash
cd ~/Chris/bookshelf-unified
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source ~/Chris/ros2_ws/install/setup.bash
source .ros2_ws/install/local_setup.bash
ros2 launch bookshelf_simple_experiment_ros \
  real_experiment_operator.launch.py \
  allow_execution:=true shadow_full_sequence:=false
```

Keep the emergency stop available and do not run another MoveIt or Servo owner
at the same time.

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
