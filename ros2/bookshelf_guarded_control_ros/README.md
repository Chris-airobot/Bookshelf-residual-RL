# Bookshelf Guarded Control ROS

This package contains two separate control paths. MoveIt is used for the global
approach to the pre-insertion pose. The physical policy launch uses MoveIt Servo
for continuous Cartesian corrections during local insertion. It does not
contain a policy network; it consumes the nominal-plus-residual delta produced
by `bookshelf_shadow_ros`.

```text
/bookshelf_shadow/final_delta
  + live validity and provenance
  + T_base_slot
  + T_base_link_eef
  + calibrated T_link_eef_link_tcp
  + T_link_tcp_virtual_policy_tool
        |
        v
target virtual policy-tool pose in slot coordinates
        |
        v
target link_eef pose in link_base
        |
        v
bounded TwistStamped commands to MoveIt Servo (local insertion only)
```

`direct_policy_servo` checks fresh observations, inference, calibration
provenance, per-step displacement, workspace bounds, and competing local
controllers. It converts each accepted TCP target to `link_eef`, publishes
bounded base-frame twists, and publishes zero velocity when an input becomes
invalid or stale. It has no planning or gripper interface.

## Two intentionally separate processes

`policy_tool_plan_checker` has no trajectory execution action client. It asks
MoveIt for a path and publishes the target poses, path, and JSON report. The
committed checker accepts the derived, physically unverified tool transform for
inspection but keeps `execution_ready=false` while the planning scene is not
confirmed complete.

`guarded_policy_tool_executor` is the only process in this package capable of
owning an execution action client. It creates that client only when every
static startup gate is already open. Its committed configuration therefore
creates no command interface because all of these gates start closed:

- `allow_unverified_policy_tool: false`
- `planning_scene_complete: false`
- `dry_run: true`
- `allow_execution: false`
- `approval_token: DISABLED`

It has no gripper interface. A valid plan is short-lived, current joints must
still match its start state, known prototype executor nodes must be absent, and
one exact approval token authorizes at most one trajectory.

Every local MoveIt result also passes a joint-trajectory sanity check before it
can become `plan_valid`. The check requires exactly `joint1` through `joint7`,
finite waypoint positions and velocities, strictly increasing timestamps, a
first waypoint close to the planned start state, bounded adjacent waypoint and
endpoint motion, bounded cumulative joint-space path length, and bounded
duration. The measured statistics and rejection reasons are included under
`trajectory_sanity` in `/bookshelf_guarded/plan_report`.

## Important planning-scene boundary

MoveIt can only collision-check geometry present in its active planning scene.
The global free-space approach and the contact-rich local insertion use two
explicit scene modes:

- `global_approach`: table and attached book are present, and one coarse box
  keeps the robot outside the complete bookshelf volume.
- `local_insertion`: an explicit, gated handoff removes only the coarse shelf
  box. The table and attached book remain while the residual policy selects
  small Cartesian targets.

`bookshelf_scene_manager` applies these objects but has no motion interface. It
starts fail-closed, automatically applies only the global scene after physical
measurements are confirmed, and rejects the local handoff unless activation is
fresh and ready. The executor additionally requires fresh
`/bookshelf_scene/status` reporting `local_insertion`; the static
`planning_scene_complete` review gate remains separate.

All physical dimensions and transition settings live in one YAML file. The
repository copy remains unapproved:

```bash
ros2 launch bookshelf_guarded_control_ros \
  bookshelf_scene_manager.launch.py \
  scene_config:=/path/to/reviewed_bookshelf_scene.yaml
```

Copy `config/bookshelf_scene_physical.yaml`, update its marked shelf/table
measurements, inspect the resulting objects in RViz, and only then change
`hardware_measurements_confirmed`. Enabling `allow_local_insertion` is a second
independent review decision.

## Unified read-only shadow rehearsal

Use the promoted stationary configuration as the single source for the frozen
slot, live marker book, held-book collision reference, and policy-tool frame:

```bash
APPROVED_CONFIG=/path/to/reviewed_trial/trial_static_slot.yaml

ros2 launch bookshelf_guarded_control_ros \
  physical_experiment_shadow_rehearsal.launch.py \
  trial_name:=shadow_rehearsal_001 \
  approved_config:="$APPROVED_CONFIG" \
  policy_bundle:=/path/to/bookshelf_shadow_actor.npz \
  activation_envelope:=/path/to/activation_envelope.json \
  show_rviz:=false
```

The launch validates the approved configuration, provenance, policy bundle,
and 12-channel activation envelope before including any runtime components.
The observation bringup owns the only RGB-D slot detector; shadow inference
reuses its topics. This compatibility launch also starts the hardware bringup,
which creates MoveIt/controller interfaces, but it sends no plan, trajectory,
gripper, or execution request. The two-launch workflow below is preferred
because it makes hardware ownership explicit. Policy activation may remain
false until the robot reaches the local insertion region.

## Two-launch physical operation

The preferred operator interface separates hardware ownership from policy
deployment:

```bash
# Terminal 1: the only xArm/MoveIt/camera owner.
ros2 launch bookshelf_policy_ros physical_hardware_bringup.launch.py \
  robot_ip:=192.168.1.209 show_rviz:=false

# Terminal 2: consumes existing hardware topics; never starts hardware.
ros2 launch bookshelf_guarded_control_ros physical_policy_deployment.launch.py \
  trial_name:=policy_shadow_001 \
  approved_config:=/path/to/trial_static_slot.yaml \
  policy_bundle:=/path/to/bookshelf_shadow_actor.npz \
  activation_envelope:=/path/to/activation_envelope.json \
  operation:=calculate
```

The policy launch explicitly disables nested hardware and marker-detector
bringup. It starts the slot diagnostic, logging, observation adapter, policy
calculation, and audit exactly once. The observation adapter continuously uses
the live marker-derived book pose; policy deployment does not compare it with a
recorded fixed-grasp reference. `calculate` creates no robot-command client.
`control` starts only `direct_policy_servo`; each local policy step is sent to
the Servo server already owned by the hardware launch.
The Servo server reuses the active `xarm7_traj_controller`; it does not change
xArm mode or create another robot connection. Do not send a separate MoveIt
trajectory goal while `operation:=control` is running.
Stop both terminals before restarting either stack.

## Build and test

Run these on the target machine. The package depends on the installed
`bookshelf_shadow_ros` package and the xArm MoveIt installation.

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_shadow_ws/install_standard/setup.bash

cd /home/riot/Chris/bookshelf-unified

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=$PWD/ros2/bookshelf_guarded_control_ros \
python3 -m pytest \
  ros2/bookshelf_guarded_control_ros/test \
  -q

colcon \
  --log-base /home/riot/Chris/bookshelf_guarded_control_ws/log \
  build \
  --base-paths ros2/bookshelf_guarded_control_ros \
  --build-base /home/riot/Chris/bookshelf_guarded_control_ws/build \
  --install-base /home/riot/Chris/bookshelf_guarded_control_ws/install \
  --packages-select bookshelf_guarded_control_ros
```

## Plan-only launch

Start the robot, MoveIt, and the calibrated candidate shadow pipeline first.
Then source this package and launch only the checker:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/setup.bash

ros2 launch bookshelf_guarded_control_ros policy_tool_plan_only.launch.py
```

Inspect `/bookshelf_guarded/plan_report`. A first report may have
`valid=true`, but it must retain `execution_ready=false` until the planning
scene is independently completed and checked.

Do not launch `guarded_policy_tool_single_step.launch.py` merely to inspect a
plan. Enabling that process requires a separate reviewed experiment-specific
configuration; the committed file is intentionally non-executable.

Do not run the older `policy_to_robot_node`, `cartesian_action_executor_node`,
or `action_executor_node` alongside this package. They are explicitly blocked
because they use prototype frame/action mappings and do not provide this
package's planning, provenance, freshness, or approval gates.
