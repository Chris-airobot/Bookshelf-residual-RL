# Bookshelf Real-Robot Experiment Runbook

Last stationary preflight: 2026-08-08

This runbook is for the xArm7 bookshelf insertion experiment on the Riot PC.
It separates global motion, local residual-policy activation, plan-only checking,
and explicitly approved single-step execution.

For the current scan-first physical sequence, use
`REAL_ROBOT_EXPERIMENT_COMMANDS_2026-08-09.md`. It captures the unobstructed
slot, requires RViz approval, and creates one trial-specific slot configuration
before the book is attached. The package-default August 4 slot must not be used
as the current physical reference.

The stationary preflight verified that:

- the unified source checkout builds on Riot;
- the far robot pose is rejected by the local-policy activation gate;
- the plan-only layer fails closed;
- no execution node is needed for the shadow and planning checks;
- automatic logging records a finalized compressed ROS bag, manifest, event log,
  and ROS graph.

Physical execution itself has not yet been validated.

## Safety Rules

1. Keep a trained operator beside the robot with immediate access to the stop
   control.
2. Clear people and loose objects from the robot workspace.
3. Use the traditional planner for global motion to the pre-insertion pose.
   The PPO policy is only a local insertion controller.
4. Do not launch the guarded executor until the activation gate and plan-only
   checker both pass.
5. The first physical test is one low-scale Cartesian step only.
6. Keep release and all gripper commands disabled for the first motion test.
7. Stop immediately if the target TCP, planned path, or physical motion is not
   consistent with the expected insertion direction.
8. Never run from one of the old repository checkouts.

## Fixed Paths

Run all experiment commands from the unified Riot checkout:

```bash
REPO=/home/riot/Chris/bookshelf-unified
UNIFIED_INSTALL=/home/riot/Chris/bookshelf_unified_ws/install
HARDWARE_INSTALL=/home/riot/Chris/ros2_ws/install_depth_fix

POLICY=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz
ENVELOPE=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json
LOG_ROOT=/home/riot/BookshelfFiles/experiment_logs
```

Expected asset hashes:

```text
Policy bundle:
75773dde0edabebcb525469c2e2b1cf868d7724f45a9f661f994cd8847a0ab19

Activation-envelope JSON:
82213b44217c52a300917ff3e0a3d1f247d22c127f54ce03aae3490a97fb1be3

Envelope simulator-source hash:
3a11e503691711aeed5c2d0563f90ffc594fcdfb3e3c689b3a9e486d00f4568f
```

## 1. Source And Asset Preflight

Run this before starting ROS:

```bash
cd /home/riot/Chris/bookshelf-unified

git branch --show-current
git rev-parse HEAD
git rev-parse origin/combined/bookshelf-20260808

sha256sum \
  /home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  /home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json
```

The two Git hashes must match. The known modified `data/bc/*.pt` files are a
Git LFS checkout issue and must not be staged or used as the residual policy.

## 2. Start Hardware And Camera

### Riot Terminal 1

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash

ros2 launch bookshelf_policy_ros \
  marker_vision_bringup.launch.py \
  show_rviz:=false
```

This terminal provides robot state, TF, MoveIt, and camera topics. Starting the
bringup must not command movement.

## 3. Start Automatic Experiment Logging

Start logging before global motion so the full trial is recorded.

### Riot Terminal 2

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/setup.bash

mkdir -p /home/riot/BookshelfFiles/experiment_logs

TRIAL_NAME=physical_trial_001

ros2 launch bookshelf_shadow_ros \
  experiment_logging.launch.py \
  trial_name:="$TRIAL_NAME" \
  output_root:=/home/riot/BookshelfFiles/experiment_logs \
  repository_path:=/home/riot/Chris/bookshelf-unified \
  policy_bundle:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  activation_envelope:=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  record_camera:=true \
  minimum_free_space_gb:=5.0
```

Use a new trial name for every attempt. This launch is subscriber-only and
cannot move the robot.

## 4. Capture And Freeze The Current Slot

Follow sections 4 through 7 of
`REAL_ROBOT_EXPERIMENT_COMMANDS_2026-08-09.md`. The resulting file is:

```text
/home/riot/BookshelfFiles/experiment_logs/environment_checks/<TRIAL_NAME>/trial_static_slot.yaml
```

It must come from a valid read-only capture and explicit RViz approval. Use
that same file as `check_config`, `target_config`, and `adapter_config`.

## 5. Start The Shadow Observation And Policy Pipeline

### Riot Terminal 3

```bash
cd /home/riot/Chris/bookshelf-unified

source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/setup.bash

ros2 launch bookshelf_shadow_ros \
  policy_calibrated_static_shadow.launch.py \
  adapter_config:=/home/riot/BookshelfFiles/experiment_logs/environment_checks/physical_trial_001/trial_static_slot.yaml \
  policy_bundle:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  activation_envelope:=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  enable_audit:=false
```

At the far pose, `policy_activation_ready` must remain false. This is expected.

## 6. Confirm That No Executor Is Running

### Riot Terminal 4

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/setup.bash

ros2 node list | grep -E \
  "guarded_policy_tool_executor|policy_to_robot|cartesian_action_executor|action_executor" \
  && echo "STOP: an execution node is already running" \
  || echo "PASS: no execution node"

ros2 node list | sort | uniq -c | sort -nr
```

Stop and cleanly restart the system if a command-capable node is duplicated.

## 7. Move Globally To The Pre-Insertion Pose

Use the validated traditional-planner workflow to move the robot to the
physical pre-insertion pose. Do not use PPO output for this movement.

The stored pre-insertion pose must be checked for the current shelf, book,
grasp, and robot setup before it is executed. Do not assume an old saved pose is
still correct.

After global motion finishes, leave the robot stationary.

## 8. Check Local-Policy Activation

### Riot Terminal 4

```bash
ros2 topic echo /bookshelf_shadow/policy_activation_ready
```

Proceed only after it reports `true` stably. Then inspect the detailed result:

```bash
ros2 topic echo --once \
  /bookshelf_shadow/policy_activation_debug \
  --field data
```

Required conditions:

- `ready` is true;
- `instantaneous_ready` is true;
- `consecutive_ready_samples` is at least 10;
- `reasons` is empty;
- `normalized_outliers` is empty;
- `envelope_outliers` is empty;
- `hardware_commanded` is false.

If any condition fails, return to the global planner. Do not launch the local
executor.

## 9. Run Plan-Only Checking

### Riot Terminal 5

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/setup.bash

ros2 launch bookshelf_guarded_control_ros \
  policy_tool_plan_only.launch.py
```

### Riot Terminal 4

```bash
ros2 topic echo --once /bookshelf_guarded/plan_valid

ros2 topic echo --once \
  /bookshelf_guarded/plan_report \
  --field data

ros2 topic echo --once /bookshelf_guarded/target_policy_tool
ros2 topic echo --once /bookshelf_guarded/target_tcp

timeout 5 ros2 run tf2_ros tf2_echo \
  link_base link_tcp
```

Do not proceed unless all of the following are true:

- `plan_valid` is true;
- activation is still ready;
- collision and reachability checks passed;
- the target TCP is a small one-step displacement from the current TCP;
- the direction agrees with the intended insertion correction;
- no release or gripper command is requested;
- `hardware_commanded` remains false.

Stop the plan-only launch before starting the guarded single-step executor.

## 10. One-Time Executor Configuration Review

The physical executor configuration must be reviewed before experiment day.
Do not use the default file as implicit permission to move.

Inspect the exact gates with:

```bash
cd /home/riot/Chris/bookshelf-unified

rg -n \
  "approval|token|execution|enable|command_scale|max_|release|gripper|activation" \
  ros2/bookshelf_guarded_control_ros/config \
  ros2/bookshelf_guarded_control_ros/bookshelf_guarded_control_ros
```

Create a separately reviewed configuration at:

```text
/home/riot/BookshelfFiles/experiment_configs/guarded_policy_tool_executor_physical.yaml
```

For the first physical test it must enforce:

- one step maximum;
- low command scale, initially 0.1;
- activation-ready required;
- fresh observation and TF required;
- valid plan required;
- workspace and displacement limits enabled;
- release disabled;
- gripper commands disabled;
- an explicit one-session approval token.

The exact approval fields must be taken from the source inspection above. Do
not guess their names or values beside the robot.

## 11. Guarded Single-Step Execution

Do not run this section until the executor configuration review is complete,
the operator is ready, activation is true, and the plan-only target has been
approved.

Because Terminal 2 is already recording the whole experiment, disable the
second integrated logger to avoid duplicate bags:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/setup.bash

EXECUTOR_CONFIG=/home/riot/BookshelfFiles/experiment_configs/guarded_policy_tool_executor_physical.yaml

test -f "$EXECUTOR_CONFIG" \
  || { echo "STOP: reviewed executor config is missing"; exit 1; }

ros2 launch bookshelf_guarded_control_ros \
  guarded_policy_tool_single_step.launch.py \
  executor_config:="$EXECUTOR_CONFIG" \
  enable_logging:=false \
  trial_name:=physical_trial_001 \
  experiment_output_root:=/home/riot/BookshelfFiles/experiment_logs \
  repository_path:=/home/riot/Chris/bookshelf-unified \
  policy_bundle:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  activation_envelope:=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  record_camera:=true
```

The operator must watch the robot throughout this step. Stop immediately after
the single action or at the first unexpected motion.

## 12. Shutdown Order

Stop processes using `Ctrl+C` in this order:

1. guarded single-step executor;
2. plan-only checker, if still running;
3. shadow policy pipeline;
4. automatic experiment logger;
5. hardware and camera bringup.

Stopping the logger after the policy processes allows it to capture their final
states and finalize the compressed bag.

## 13. Verify The Trial Record

```bash
source /opt/ros/humble/setup.bash

LOG_ROOT=/home/riot/BookshelfFiles/experiment_logs

RUN_DIR=$(find "$LOG_ROOT" \
  -mindepth 1 -maxdepth 1 -type d \
  -printf '%T@ %p\n' | \
  sort -n | tail -1 | cut -d' ' -f2-)

echo "Run directory: $RUN_DIR"

find "$RUN_DIR" -maxdepth 2 -type f \
  -printf '%s %p\n' | sort -n

ros2 bag info "$RUN_DIR/rosbag"
```

Required files:

- `manifest.json`
- `events.jsonl`
- `ros_graph.json`
- `rosbag/metadata.yaml`
- `rosbag/rosbag_0.db3.zstd`

Preserve the complete run directory. Do not overwrite or rename individual
files inside it.

## Stop Conditions

Stop the experiment without execution if any of these occurs:

- source, policy, or envelope hashes differ;
- duplicate command-capable nodes are present;
- camera, joint-state, or TF topics are stale;
- slot or book calibration is inconsistent with the physical scene;
- activation is false or unstable;
- normalized observations are outside the simulator envelope;
- target TCP is not a small local correction;
- plan-only checking fails;
- collision or reachability checking is unavailable;
- the reviewed executor configuration or approval token is missing;
- automatic logging is not active;
- the operator cannot immediately stop the robot.
