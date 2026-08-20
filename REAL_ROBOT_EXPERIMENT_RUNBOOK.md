# Bookshelf Real-Robot Experiment Runbook

Last offline stationary calibration and shadow replay: 2026-08-19

This runbook is for the xArm7 bookshelf insertion experiment on the Riot PC.
It separates the MoveIt global approach from direct residual-policy Cartesian
control during local insertion.

The current approved configuration is candidate `53e7fe80d56d`. It binds the
frozen View A slot, continuous marker-based held-book pose, policy-tool frame,
and coarse physical scene. Do not use the package-default August 4 slot or an
older trial configuration.

The stationary preflight verified that:

- the unified source checkout builds on Riot;
- the far robot pose is rejected by the local-policy activation gate;
- policy calculation runs without a robot-command client;
- local Cartesian targets can be inspected before control;
- automatic logging records a finalized compressed ROS bag, manifest, event log,
  and ROS graph.

Physical execution itself has not yet been validated.

## Safety Rules

1. Keep a trained operator beside the robot with immediate access to the stop
   control.
2. Clear people and loose objects from the robot workspace.
3. Use the traditional planner for global motion to the pre-insertion pose.
   The PPO policy is only a local insertion controller.
4. Run `operation:=calculate` first and inspect the calculated TCP direction.
5. Start `operation:=control` only after MoveIt global motion has stopped.
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
git rev-parse origin/main

sha256sum \
  /home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  /home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json
```

The two Git hashes must match. The known modified `data/bc/*.pt` files are a
Git LFS checkout issue and must not be staged or used as the residual policy.

## 2. Start Hardware, Then Deploy The Shadow Policy

Start from a clean ROS graph. Do not leave an older xArm, MoveIt, camera, or
marker-vision launch running. The hardware and policy stacks have separate
ownership and must each be launched exactly once.

### Riot Terminal 1

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

ros2 launch bookshelf_policy_ros \
  physical_hardware_bringup.launch.py \
  robot_ip:=192.168.1.209 \
  show_rviz:=false
```

This is the only launch that may own the xArm, MoveIt, RealSense, hand-eye TF,
and calibrated marker-book detector. It creates hardware-capable MoveIt
interfaces but sends no motion, gripper, or policy goal. Use `show_rviz:=true`
only from the Riot graphical desktop; keep it false over SSH.

Wait for `/joint_states`, RGB, aligned depth, `link_base -> link_tcp`, and
`link_tcp -> target_book_center` before starting Terminal 2.

### Riot Terminal 2

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

APPROVED_CONFIG=/home/riot/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml
TRIAL_NAME=policy_shadow_$(date +%Y%m%d_%H%M%S)

test -f "$APPROVED_CONFIG" \
  || { echo "STOP: approved configuration is missing"; exit 1; }

ros2 launch bookshelf_guarded_control_ros \
  physical_policy_deployment.launch.py \
  trial_name:="$TRIAL_NAME" \
  approved_config:="$APPROVED_CONFIG" \
  repository_path:=/home/riot/Chris/bookshelf-unified \
  policy_bundle:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  activation_envelope:=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  operation:=calculate \
  record_camera:=true \
  record_raw_replay_inputs:=false \
  capture_condition:=book_attached \
  capture_duration_s:=0.0
```

This launch reuses the live hardware topics. It owns the sole RGB-D slot
detector, frozen-slot diagnostics, held-book gate, logger, policy adapter,
policy calculation and policy audit. In the default `calculate` operation it starts no
planning-scene manager, planner, or executor. Policy output remains diagnostic
and cannot move the robot.

The only other operation is `control`. It starts the policy-to-MoveIt-Servo
bridge. The Servo server and trajectory controller remain in the hardware
launch; the policy launch does not create a driver, planner, or gripper client.

### Riot Terminal 3

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

ros2 node list | grep -E \
  "guarded_policy_tool_executor|policy_to_robot|cartesian_action_executor|action_executor" \
  && echo "STOP: an execution node is already running" \
  || echo "PASS: no execution node"

test "$(ros2 node list | grep -c '^/rgbd_slot_detector$')" -eq 1 \
  && echo "PASS: exactly one slot detector" \
  || echo "STOP: slot detector ownership is invalid"

ros2 topic echo --once /bookshelf_environment/static_slot_check_passed
ros2 topic echo --once /bookshelf_scene/held_book_pose_check_passed
ros2 topic echo --once /bookshelf_policy/observation_valid
ros2 topic echo --once /bookshelf_shadow/inference_valid
ros2 topic echo --once /bookshelf_shadow/policy_activation_ready
```

The held-book and observation checks must become true with the approved
physical setup. The live frozen-slot comparison may remain false when the held
book occludes the slot. `policy_activation_ready` and `inference_valid` may
both remain false while the robot is outside the local insertion region; that
is the expected fail-closed state. Stop if any bookshelf execution node exists
or if the detector count is not exactly one.

## 3. Move Globally To The Pre-Insertion Pose

Use the validated traditional-planner workflow to move the robot to the
physical pre-insertion pose. Do not use PPO output for this movement.

The stored pre-insertion pose must be checked for the current shelf, book,
grasp, and robot setup before it is executed. Do not assume an old saved pose is
still correct.

After global motion finishes, leave the robot stationary.

## 4. Check Local-Policy Activation

### Riot Terminal 2

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

If any condition fails, return to the global planner. Do not start local
control.

## 5. Start Local Policy Servo Control

Stop the Terminal 2 `calculate` launch. Do not send another MoveIt goal after
this point. Restart the same Terminal 2 command with:

```bash
operation:=control
```

This starts `direct_policy_servo`. It starts the existing MoveIt Servo server
only after fresh observation, inference, calibration provenance, slot pose, and
EEF TF are available. Every policy step is bounded, converted through the
approved virtual policy-tool and TCP transforms, and sent as a capped base-frame
velocity command for `link_eef`. It never changes xArm mode or commands the
gripper.

Monitor the local controller in a third terminal:

```bash
ros2 topic echo /bookshelf_control/command_valid
ros2 topic echo /bookshelf_control/status --field data
ros2 topic echo /bookshelf_control/target_tcp
ros2 topic echo /servo_server/status
```

`hardware_commanded` in the status changes to true only after a nonzero twist
is published. Stop immediately at unexpected motion or a Servo halt status.

## 8. Shutdown Order

Stop processes using `Ctrl+C` in this order:

1. direct policy servo, if running;
2. policy calculation launch;
3. physical hardware launch.

Stopping the unified rehearsal last allows its integrated logger to capture the
final policy states and finalize the compressed bag.

## 9. Verify The Trial Record

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
- observation or inference validity is false or unstable;
- target TCP is not a small local correction;
- `/servo_server/start_servo` or `/servo_server/delta_twist_cmds` is unavailable;
- MoveIt Servo reports a collision, singularity, or joint-limit halt;
- automatic logging is not active;
- the operator cannot immediately stop the robot.

## Appendix: Offline Stationary A/B/C Calibration

This workflow processes the three stationary observation bags without starting
MoveIt, a controller, a gripper interface, or a robot-command node:

- View A: close slot view, no book;
- View B: independent slot view, no book;
- View C: book attached in the intended rigid grasp at a safe stationary pose.

The command replays only raw RGB, aligned depth, CameraInfo, `/tf`, and
`/tf_static`. Recorded detector outputs and controller/action topics are not
replayed. A and B are processed independently and must agree before a fused
slot candidate is written. C produces `T_link_eef_book`, captures the fixed
`T_link_eef_link_tcp`, and derives the TCP book and virtual policy-tool
candidate transforms.

### Alienware Terminal

```bash
source /opt/ros/humble/setup.bash

cd /home/chris/Chris/bookshelf-unified

colcon --log-base /tmp/bookshelf_stationary_pipeline_log \
  build \
  --base-paths ros2/bookshelf_shadow_ros \
  --build-base /tmp/bookshelf_stationary_pipeline_build \
  --install-base /tmp/bookshelf_stationary_pipeline_install \
  --packages-select bookshelf_shadow_ros

source /tmp/bookshelf_stationary_pipeline_install/setup.bash

ROOT=/home/chris/BookshelfFiles/real_robot_bags/stationary_captures_20260817_abc
OUT=/home/chris/BookshelfFiles/evaluation_results/stationary_capture_pipeline/$(date +%Y-%m-%d_%H-%M-%S)

ros2 run bookshelf_shadow_ros stationary_capture_pipeline \
  --view-a-run "$ROOT/2026-08-17_23-04-44_slot_close_view_a_20260817_230444" \
  --view-b-run "$ROOT/2026-08-17_23-07-06_slot_close_view_b_20260817_230706" \
  --book-run "$ROOT/2026-08-17_23-17-21_book_attached_safe_pose_20260817_231721" \
  --repository /home/chris/Chris/bookshelf-unified \
  --slot-minimum-confidence 0.55 \
  --output-dir "$OUT"
```

The `0.55` threshold is specific to the recorded alternate View B, whose
stable detector confidence lies around `0.58-0.61`. It does not bypass the
120-sample robust filter or the independent A/B pose and width agreement gate.

The expected final files are:

- `capture_input_audit.json`;
- `view_a/static_slot_capture_candidate.json`;
- `view_b/static_slot_capture_candidate.json`;
- `static_slot_cross_view_candidate.json`;
- `book/marker_book_calibration_summary.json`;
- `book/eef_tcp_context.json`;
- `stationary_calibration_candidate.yaml`;
- `stationary_calibration_bundle_candidate.json`.

Even when every candidate is valid, the generated bundle states
`candidate_selected=false`, `execution_authorized=false`, and
`hardware_commanded=false`. Human visual review and the existing separate
promotion gates remain mandatory.
