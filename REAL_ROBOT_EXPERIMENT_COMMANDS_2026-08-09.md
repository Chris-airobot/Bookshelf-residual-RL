# Real-Robot Experiment Commands — 2026-08-09

This is the consolidated operator sequence for the xArm7 bookshelf experiment
from the unified repository and unified ROS install. Do not source the retired
`/tmp/bookshelf_marker_install` overlay.

The sequence intentionally separates:

1. hardware, camera, MoveIt, and logging;
2. read-only static-slot capture, human review, and trial-specific freezing;
3. automatic global target calculation and collision-aware MoveIt planning;
4. post-positioning shadow activation and plan-only evidence.

It does **not** authorize or launch `guarded_policy_tool_single_step`, any
approval publication, a gripper command, or trajectory execution. The global
bridge produces a path for review; it never sends that path to the robot.

## Safety stop conditions

- A trained operator must remain beside the robot with immediate stop access.
- Stop if the shelf, camera, book grasp, TF, target, or planned path looks
  inconsistent.
- Never run `policy_to_robot`, `cartesian_action_executor`, or
  `action_executor`.
- Never execute a MoveIt plan until a human has reviewed the entire path and
  the physical clearance.
- The MoveIt scene does not provide independent absolute shelf-pose ground
  truth. RViz review must be compared with the physical setup.

## 0. Fixed paths

```bash
REPO=/home/riot/Chris/bookshelf-unified
HARDWARE_INSTALL=/home/riot/Chris/ros2_ws/install_depth_fix
UNIFIED_INSTALL=/home/riot/Chris/bookshelf_unified_ws/install
POLICY=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz
ENVELOPE=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json
LOG_ROOT=/home/riot/BookshelfFiles/experiment_logs
TRIAL_NAME=physical_trial_001
```

Choose a new `TRIAL_NAME` for every attempt.

## 1. Source and asset preflight

```bash
cd /home/riot/Chris/bookshelf-unified

git branch --show-current
git rev-parse HEAD
git rev-parse origin/combined/bookshelf-20260808

sha256sum \
  /home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  /home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json

ping -c 3 192.168.1.209
df -h /home/riot/BookshelfFiles/experiment_logs
```

Expected branch and hashes:

```text
combined/bookshelf-20260808
policy:   75773dde0edabebcb525469c2e2b1cf868d7724f45a9f661f994cd8847a0ab19
envelope: 82213b44217c52a300917ff3e0a3d1f247d22c127f54ce03aae3490a97fb1be3
```

The two Git commit hashes must match. Do not stage, restore, or use the known
modified `data/bc/*.pt` Git LFS files.

## 2. Terminal 1 — hardware, camera, TF, and MoveIt

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

ros2 pkg prefix bookshelf_policy_ros
ros2 pkg prefix bookshelf_shadow_ros
```

`bookshelf_shadow_ros` must resolve under
`/home/riot/Chris/bookshelf_unified_ws/install`. Then launch headlessly over
SSH:

```bash
mkdir -p /tmp/bookshelf_hardware_ros_logs

ROS_LOG_DIR=/tmp/bookshelf_hardware_ros_logs \
ros2 launch bookshelf_policy_ros marker_vision_bringup.launch.py \
  enable_robot_control:=true \
  enable_calibrated_book_detection:=false \
  enable_legacy_three_book_detection:=false \
  show_rviz:=false
```

This starts the xArm hardware interface, robot state, TF, MoveIt, and RealSense
camera. It does not start a bookshelf policy executor. A local operator who
needs RViz may deliberately use `show_rviz:=true` from the Riot desktop; keep
it false over SSH.

## 3. Terminal 2 — automatic logging before any movement

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

mkdir -p /home/riot/BookshelfFiles/experiment_logs
TRIAL_NAME=physical_trial_001

ros2 launch bookshelf_shadow_ros experiment_logging.launch.py \
  trial_name:="$TRIAL_NAME" \
  output_root:=/home/riot/BookshelfFiles/experiment_logs \
  repository_path:=/home/riot/Chris/bookshelf-unified \
  policy_bundle:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  activation_envelope:=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  record_camera:=true \
  minimum_free_space_gb:=5.0
```

Leave this running and record the automatic experiment directory printed on
screen. Logging includes TF, joint state, slot-check results, MoveIt displayed
plans, monitored planning scene, policy diagnostics, plan-only results, and
trajectory action evidence.

## 4. Terminal 3 — capture and freeze the unobstructed slot

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

ros2 pkg prefix bookshelf_policy_ros
ros2 pkg prefix bookshelf_shadow_ros
```

Expected prefixes in this terminal include:

```text
/home/riot/Chris/bookshelf_unified_ws/install/bookshelf_shadow_ros
```

Launch the read-only detector and robust capture while the slot is unobstructed:

```bash
TRIAL_NAME=physical_trial_001
CAPTURE_DIR=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME/slot_capture

ros2 launch bookshelf_shadow_ros static_slot_capture.launch.py \
  output_dir:="$CAPTURE_DIR" \
  repository_path:=/home/riot/Chris/bookshelf-unified \
  target_samples:=120
```

If exactly one `/rgbd_slot_detector` is already running, do not duplicate it:

```bash
ros2 launch bookshelf_shadow_ros static_slot_capture.launch.py \
  start_live_detector:=false \
  output_dir:="$CAPTURE_DIR" \
  repository_path:=/home/riot/Chris/bookshelf-unified \
  target_samples:=120
```

### RViz displays

In the existing MoveIt RViz, set `Fixed Frame` to `link_base` and add:

- `MarkerArray`: `/bookshelf_environment/static_slot_candidate_markers`
- `Image`: `/slot_detector/debug_image`

The image shows the camera frame, detector ROI, opening mask, detected slot
boundaries, centre line, width, and confidence. The debug-image publisher is
RELIABLE with depth one so it matches the default Humble RViz Image display.

After 120 accepted samples, inspect:

- the green candidate outline matches the physical opening;
- the arrow points into the shelf along slot local `+X`;
- the annotated RGB-D image selects the intended opening;
- the reported residuals and inlier fraction are stable.

In another terminal, check the result:

```bash
ros2 topic echo /bookshelf_environment/static_slot_capture_ready --once
ros2 topic echo \
  /bookshelf_environment/static_slot_capture_status \
  --field data --once

python3 -m json.tool \
  "$CAPTURE_DIR/static_slot_capture_candidate.json"
```

The capture is still unapproved and cannot affect the policy. Stop the capture
launch after visual review. Only when the RViz candidate matches the physical
slot, explicitly create this trial's single shared parameter file:

```bash
cd /home/riot/Chris/bookshelf-unified

CANDIDATE=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME/slot_capture/static_slot_capture_candidate.json
TRIAL_SLOT_CONFIG=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME/trial_static_slot.yaml

python3 scripts/promote_static_slot_capture.py \
  --candidate "$CANDIDATE" \
  --output "$TRIAL_SLOT_CONFIG" \
  --approval-token VISUALLY_APPROVED_STATIC_SLOT

python3 -m json.tool \
  "${TRIAL_SLOT_CONFIG%.yaml}.provenance.json"
```

This command does not launch ROS or command hardware. It creates one config
containing consistent slot values for the environment check, pre-insertion
target, and policy observation adapter. It never edits the package defaults.

## 5. Terminal 3 — verify the frozen trial slot

Restart the detector and compare it with the newly frozen trial reference:

```bash
TRIAL_NAME=physical_trial_001
TRIAL_SLOT_CONFIG=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME/trial_static_slot.yaml

ros2 launch bookshelf_shadow_ros static_slot_environment_check.launch.py \
  check_config:="$TRIAL_SLOT_CONFIG" \
  output_dir:=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME/frozen_check
```

In RViz, replace the capture MarkerArray with
`/bookshelf_environment/slot_markers`. Cyan is the frozen candidate; green is
the live estimate agreeing with it; red indicates disagreement.

### One-command software readiness report

While the hardware launch and frozen-slot check are running, generate the
fail-closed readiness report. This command only reads files, ROS graph state,
four live messages, and one TF. It creates no publishers, planners,
controllers, gripper clients, or execution clients.

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

TRIAL_NAME=physical_trial_001
TRIAL_DIR=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME

ros2 run bookshelf_shadow_ros physical_experiment_preflight \
  --repository /home/riot/Chris/bookshelf-unified \
  --trial-slot-config "$TRIAL_DIR/trial_static_slot.yaml" \
  --policy-bundle /home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  --activation-envelope /home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  --expected-shadow-prefix /home/riot/Chris/bookshelf_unified_ws/install/bookshelf_shadow_ros \
  --expected-guarded-prefix /home/riot/Chris/bookshelf_unified_ws/install/bookshelf_guarded_control_ros \
  --check-ros \
  --require-frozen-check \
  --output "$TRIAL_DIR/software_preflight.json"
```

Expected final lines:

```text
Software ready: True
Execution authorized: False
Hardware commanded: False
```

`Software ready` means the frozen slot, provenance, policy assets, committed
fail-closed executor defaults, hardware topics, camera TF, and frozen live
check are coherent. It deliberately does not authorize movement.

## 6. Terminal 4 — read-only system and slot checks

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash
```

Confirm that no executor exists:

```bash
ros2 node list | grep -E \
  "guarded_policy_tool_executor|policy_to_robot|cartesian_action_executor|action_executor" \
  && echo "STOP: execution node is running" \
  || echo "PASS: no execution node"

ros2 node list | sort | uniq -c | sort -nr
```

Confirm logging, state, camera, and TF:

```bash
ros2 node list | grep -E "bookshelf_experiment_logger|rosbag2_recorder"
ros2 topic echo /joint_states --once
timeout 5 ros2 topic hz /camera/color/image_raw
timeout 5 ros2 topic hz /camera/aligned_depth_to_color/image_raw
timeout 5 ros2 run tf2_ros tf2_echo link_base camera_color_optical_frame
```

Confirm the annotated image and live detection:

```bash
ros2 topic info -v /slot_detector/debug_image
ros2 topic echo /slot_detector/debug_image --once --field header
ros2 topic echo /slot_detector/confidence --once
ros2 topic echo /slot_detector/slot_width --once
ros2 topic echo /slot_detector/slot_pose --once
```

Confirm the environment result:

```bash
ros2 topic echo /bookshelf_environment/static_slot_check_passed --once
ros2 topic echo \
  /bookshelf_environment/static_slot_check_status \
  --field data --once

python3 -m json.tool \
  /home/riot/BookshelfFiles/experiment_logs/environment_checks/physical_trial_001/frozen_check/static_slot_environment_check.json
```

Accept the slot for this trial only when the live outline is green, RViz agrees
with the physical shelf, the label reaches `PASS 30/30`, and the Boolean topic
is true. The fixed checks are confidence at least 0.60, translation error at
most 10 mm, rotation error at most 5 degrees, and width error at most 5 mm. Do
not weaken a limit to make the check pass.

## 7. Prepare the reviewed global collision scene

The automatic bridge requires a trial-specific scene file. Create it from the
repository template before the experiment, then replace the measured shelf and
table values. Set `hardware_measurements_confirmed: true` only after those
values and the held-book box have been checked in RViz. Keep
`allow_local_insertion: false` during the global approach.

```bash
mkdir -p /home/riot/BookshelfFiles/experiment_configs

SCENE_CONFIG=/home/riot/BookshelfFiles/experiment_configs/physical_trial_001_bookshelf_scene.yaml

cp --no-clobber \
  /home/riot/Chris/bookshelf-unified/ros2/bookshelf_guarded_control_ros/config/bookshelf_scene_physical.yaml \
  "$SCENE_CONFIG"

test -f "$SCENE_CONFIG" \
  && echo "SCENE_CONFIG_FOUND" \
  || echo "STOP: scene config is missing"
```

Do not edit the package template. The external file is the reviewed record for
this physical setup.

## 8. Automatic global pre-insertion plan only

Keep the logger and frozen environment check running. Leave the robot
stationary while starting this command. The bridge launches the collision-scene
manager, calculates the target from this trial's frozen slot, preserves the
current `link_tcp` orientation, and submits one MoveIt planning request.

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

TRIAL_NAME=physical_trial_001
TRIAL_DIR=/home/riot/BookshelfFiles/experiment_logs/environment_checks/$TRIAL_NAME
SCENE_CONFIG=/home/riot/BookshelfFiles/experiment_configs/${TRIAL_NAME}_bookshelf_scene.yaml

ros2 launch bookshelf_guarded_control_ros \
  calibrated_preinsert_plan_only.launch.py \
  target_config:="$TRIAL_DIR/trial_static_slot.yaml" \
  scene_config:="$SCENE_CONFIG" \
  output_dir:="$TRIAL_DIR/preinsert_plan"
```

Inspect the result from the monitoring terminal:

```bash
ros2 topic echo /bookshelf_scene/status --field data --once
ros2 topic echo /bookshelf_shadow/current_tcp_pose --once
ros2 topic echo /bookshelf_preinsert/target_tcp --once
ros2 topic echo /bookshelf_preinsert/plan_valid --once
ros2 topic echo /bookshelf_preinsert/plan_report --field data --once

python3 -m json.tool \
  "$TRIAL_DIR/preinsert_plan/calibrated_preinsert_plan_report.json"
```

Require all of the following:

- scene mode is `global_approach`;
- shelf keep-out, table, and held-book collision objects are active;
- `plan_valid` is true and `path_planned` is true;
- trajectory sanity passed;
- `preserved_tcp_orientation_change_deg` is effectively zero;
- `execution_ready`, `execution_authorized`, and `hardware_commanded` are false.

The path is published on `/display_planned_path` for RViz review. Inspect the
complete path against the physical shelf, table, camera, cable, gripper, and
held book. This bridge cannot execute the path. Physical global movement is a
separate operator-approved MoveIt step; do not treat `plan_valid` as permission
to move.

After the separately approved global movement reaches pre-insertion, leave the
robot stationary before continuing.

## 9. Start shadow activation only after reaching pre-insertion

In a new terminal:

```bash
cd /home/riot/Chris/bookshelf-unified

source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

ros2 launch bookshelf_shadow_ros policy_calibrated_static_shadow.launch.py \
  adapter_config:=/home/riot/BookshelfFiles/experiment_logs/environment_checks/physical_trial_001/trial_static_slot.yaml \
  policy_bundle:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz \
  activation_envelope:=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json \
  enable_audit:=false
```

Collect observation, activation, and TF evidence:

```bash
ros2 topic echo /bookshelf_policy/observation_valid --once
ros2 topic echo /bookshelf_policy/observation_12d --once
ros2 topic echo /bookshelf_policy/adapter_debug --field data --once
ros2 topic echo /bookshelf_shadow/policy_activation_ready
ros2 topic echo /bookshelf_shadow/policy_activation_debug --field data --once

timeout 5 ros2 run tf2_ros tf2_echo link_base link_eef
timeout 5 ros2 run tf2_ros tf2_echo link_base link_tcp
timeout 5 ros2 run tf2_ros tf2_echo link_base calibration_detected_book
```

Require at least 10 consecutive ready samples, no reasons, no normalized
outliers, and no envelope outliers. Do not weaken the envelope.

## 10. Plan-only local-policy evidence

In a new terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash

ros2 launch bookshelf_guarded_control_ros policy_tool_plan_only.launch.py
```

Inspect from the monitoring terminal:

```bash
ros2 topic echo /bookshelf_guarded/plan_valid --once
ros2 topic echo /bookshelf_guarded/plan_report --field data --once
ros2 topic echo /bookshelf_guarded/target_policy_tool --once
ros2 topic echo /bookshelf_guarded/target_tcp --once
timeout 5 ros2 run tf2_ros tf2_echo link_base link_tcp
```

This is the end of the currently authorized sequence. Stop the plan-only launch
after collecting evidence. Do **not** launch the guarded executor.

## 11. Implementation and offline verification commands

The executor now has an atomic, hard one-trajectory-submission allowance for
the complete process lifetime. Submission failure, rejection, or result failure
does not restore the allowance.

The repository physical-executor YAML is deliberately fail closed and contains
only a token placeholder. The live one-session token belongs only in the
separately reviewed external file and must never be committed or printed.

Run offline tests with no hardware command path:

```bash
cd /home/riot/Chris/bookshelf-unified
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash

PYTHONPATH=ros2/bookshelf_guarded_control_ros:$PYTHONPATH \
/usr/bin/python3 -m pytest -q ros2/bookshelf_guarded_control_ros/test

PYTHONPATH=ros2/bookshelf_shadow_ros:$PYTHONPATH \
/usr/bin/python3 -m pytest -q ros2/bookshelf_shadow_ros/test

python3 -m compileall -q \
  ros2/bookshelf_guarded_control_ros/bookshelf_guarded_control_ros \
  ros2/bookshelf_shadow_ros/bookshelf_shadow_ros \
  ros2/bookshelf_shadow_ros/launch

git diff --check -- \
  ros2/bookshelf_guarded_control_ros \
  ros2/bookshelf_shadow_ros \
  experiment_configs \
  REAL_ROBOT_EXPERIMENT_COMMANDS_2026-08-09.md
```

No test, environment-check, target-calculation, shadow, or plan-only command in
this document authorizes robot execution.
