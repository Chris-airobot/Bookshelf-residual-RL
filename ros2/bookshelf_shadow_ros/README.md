# Bookshelf Shadow ROS

Read-only integration pipeline for the real D435 and xArm bookshelf setup:

```text
RGB + aligned depth + CameraInfo
  -> markerless slot detector
  -> 12D simulator-compatible observation
  -> saved VecNormalize statistics
  -> deterministic PPO actor
  -> nominal/residual/final diagnostics
```

Nothing in this package creates a MoveIt client, trajectory action client,
controller client, gripper client, or robot-command publisher. Policy outputs
remain under `/bookshelf_shadow/*`.

## Current validation boundary

The portable actor is verified numerically against SB3 before export. The
markerless adapter currently uses an approximate fixed `link_eef -> book`
transform from
`config/policy_observation_adapter_markerless_smoke.yaml`. It is suitable for
data-flow and inference smoke testing only. Calibrate that transform and the
slot target offset before treating the 12D values as geometrically exact.

The fixed gripper-to-book estimate is valid only while the book is rigidly held.
Post-release book tracking remains outside this markerless smoke test.

## Calibrate the grasped book frame from the recorded marker bag

The recorded calibration view uses `DICT_ARUCO_ORIGINAL`, marker ID 0, a
39 mm black square, and the measured mount in
`config/real_book_aruco0_mount.yaml`. The calibrator is subscriber-only. It
detects the marker, checks RGB reprojection and aligned depth, composes
`T_eef_camera * T_camera_marker * T_marker_book`, rejects inconsistent frames,
and averages the remaining rigid-grasp poses.

Build the local package without an editable install:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

colcon build \
  --base-paths ros2/bookshelf_shadow_ros \
  --build-base /tmp/bookshelf_shadow_build \
  --install-base /tmp/bookshelf_shadow_install \
  --packages-select bookshelf_shadow_ros
```

Local PC terminal 1:

```bash
source /opt/ros/humble/setup.bash
source /tmp/bookshelf_shadow_install/setup.bash

OUTPUT_DIR=/home/chris/RL/bookshelf/logs/marker_book_calibration/marker_book_grasp_01

ros2 launch bookshelf_shadow_ros marker_book_bag_calibration.launch.py \
  output_dir:="$OUTPUT_DIR" \
  target_samples:=250
```

Local PC terminal 2:

```bash
source /opt/ros/humble/setup.bash

BAG_DIR=/home/chris/RL/bookshelf/data/real_robot_audits/marker_book_grasp_01_2026-08-05/marker_book_grasp_01/rosbag
ros2 bag play "$BAG_DIR"
```

After playback, stop terminal 1 cleanly if it is still waiting. Inspect the
saved `debug/*.png` cuboid overlays first. A valid run writes:

```text
marker_book_samples.csv
marker_book_calibration_summary.json
eef_book_calibration.yaml
debug/sample_*.png
```

`eef_book_calibration.yaml` is emitted only when at least 30 robust inliers
remain. It can then be passed after the normal markerless adapter parameter
file so its measured `link_eef -> book` values override
`approximate_smoke_only`. This calibration is grasp-specific: moving the book
inside the gripper invalidates it.

## Calibrated static-slot shadow test

`policy_calibrated_static_shadow.launch.py` combines two recorded quantities:

- the static slot pose and width from the unobstructed 2026-08-04 RGB-D run;
- the measured rigid `link_eef -> book` transform from the 2026-08-05 marker run.

The slot configuration is labelled
`measured_rgbd_static_no_absolute_ground_truth`. Its repeatability was measured,
but its absolute pose has not been checked against an independent physical
reference. The book transform is labelled
`measured_aruco_original_id0_static_grasp` and is valid only while that grasp is
unchanged.

This launch deliberately starts no RGB-D detector. It is for testing the
calibrated geometry, 12D observation, VecNormalize, nominal controller, and PPO
actor while the marker blocks the camera's view of the slot. It also starts no
executor, IK, trajectory, gripper, or robot-command node.

Local PC terminal 1:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash
source /tmp/bookshelf_shadow_install/setup.bash

AUDIT_DIR=$PWD/logs/calibrated_static_shadow/$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$AUDIT_DIR"

ros2 launch bookshelf_shadow_ros policy_calibrated_static_shadow.launch.py \
  policy_bundle:=$PWD/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz \
  audit_output_dir:="$AUDIT_DIR" \
  audit_samples:=300 \
  use_sim_time:=true
```

Local PC terminal 2:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

BAG_DIR=$PWD/data/real_robot_audits/marker_book_grasp_01_2026-08-05/marker_book_grasp_01/rosbag

ros2 bag play "$BAG_DIR" \
  --clock \
  --topics /joint_states /tf /tf_static
```

The 20-second bag provides the real robot state and transforms. The configured
slot and book calibrations provide the remaining geometry. A complete run
writes `policy_stream_samples.csv` and `policy_stream_summary.json` under
`$AUDIT_DIR`. The summary must report:

```text
hardware_commanded: false
slot_pose_sources: configured_static
static_slot_transform_statuses: measured_rgbd_static_no_absolute_ground_truth
eef_book_transform_statuses: measured_aruco_original_id0_static_grasp
```

For a live Riot shadow test, keep `use_sim_time:=false` (the default), launch
the normal robot-state bringup, and use the same calibrated-static launch. Do
not use this static-slot mode after the bookshelf or neighbouring books move.

## Book-frame convention

The real-world adapter must provide a semantic policy book frame:

```text
+X = book depth / insertion direction
+Y = book thickness / lateral direction
+Z = book height / up direction
```

Its size parameter is therefore `[depth, thickness, height]`, currently
`[0.156, 0.034, 0.236]` m. Isaac's cuboid root instead stores dimensions as
`[depth, height, thickness]` and uses a standing rotation. The offline tests
apply and verify the fixed simulator-root-to-policy-book axis conversion.
Do not publish the raw Isaac cuboid-root convention as the calibrated real
book frame.

## Calibrated pre-insertion target report

`calibrated_preinsert_target.launch.py` solves the rigid transform needed to
place the measured book grasp at the nominal pre-insertion pose:

```text
T_base_eef_target = T_base_slot * T_slot_book_target * inverse(T_eef_book)
```

The target book is aligned with the slot, lifted by the trained controller's
6 mm insertion offset, and placed with its front face 30 mm before the shelf
mouth. The node publishes `PoseStamped` diagnostics for RViz and writes
`calibrated_preinsert_target_report.json`. It never creates IK, planning,
trajectory, gripper, controller, or robot-command interfaces.

The report also propagates an assumed +/-2 mm translation and +/-2 degree
rigid-grasp calibration uncertainty through 2,000 deterministic samples. This
is a sensitivity study, not an independently measured calibration accuracy.
It explicitly records that IK reachability, collision freedom, and execution
safety remain unchecked.

At the 30 mm standoff, `rear_to_mouth` and `front_to_back` exceed their 80 mm
policy scales. Their clipping is expected. Any other clipped target channel is
reported separately as an unexpected geometry/configuration issue.

Build and run the pure geometry tests:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
python3 -m pytest \
  ros2/bookshelf_shadow_ros/test/test_calibrated_preinsert_target_math.py \
  -q

colcon \
  --log-base /tmp/bookshelf_target_log \
  build \
  --base-paths ros2/bookshelf_shadow_ros \
  --build-base /tmp/bookshelf_target_build \
  --install-base /tmp/bookshelf_target_install \
  --packages-select bookshelf_shadow_ros
```

Generate the target report without any camera, rosbag, or robot state:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash
source /tmp/bookshelf_target_install/setup.bash

TARGET_DIR=$PWD/logs/calibrated_preinsert_target/$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$TARGET_DIR"

ros2 launch bookshelf_shadow_ros calibrated_preinsert_target.launch.py \
  output_dir:="$TARGET_DIR"
```

The static report is written immediately. Press `Ctrl+C` after it appears.
To add a comparison against the recorded marker-grasp robot pose, launch with
`use_sim_time:=true tf_max_age_s:=0.0`, then replay only clock and transforms:

```bash
ros2 bag play \
  /home/chris/RL/bookshelf/data/real_robot_audits/marker_book_grasp_01_2026-08-05/marker_book_grasp_01/rosbag \
  --clock \
  --topics /tf /tf_static
```

The comparison is diagnostic only. A large current-to-target delta is expected
when the recorded pose is not the trained policy's pre-insertion pose, and it
must not be sent to the real robot.

## Policy tool-frame audit

The simulator policy does not observe the `panda_hand` body origin. Its tool
point is shifted by `ik_body_offset_pos=[0, 0, 0.107]` m, while the nominal
book reset offset is `[0, 0, 0.075]` m. Their nominal separation is therefore
about 32 mm. Using real `link_eef` directly produced a measured tool-to-book
distance near 125 mm, so the calibrated policy path now fails closed until an
explicit `T_link_eef_policy_tool` is validated.

`policy_tool_frame_audit.launch.py` replays the recorded robot TF tree and:

- explicitly evaluates the xArm `link_tcp` tool-centre frame;
- discovers frames containing `finger`, `gripper`, `knuckle`, `eef`, `tool`, or `tcp`;
- evaluates configured xArm frame names;
- constructs position-only midpoints for matching left/right finger frames;
- ranks candidates against a conservative 20-50 mm training distance;
- never selects a candidate or authorizes execution automatically.

Build and test on the Alienware PC:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
python3 -m pytest \
  ros2/bookshelf_shadow_ros/test/test_policy_tool_frame_audit.py \
  ros2/bookshelf_shadow_ros/test/test_calibrated_preinsert_target_math.py \
  -q

colcon \
  --log-base /tmp/bookshelf_tool_audit_log \
  build \
  --base-paths ros2/bookshelf_shadow_ros \
  --build-base /tmp/bookshelf_tool_audit_build \
  --install-base /tmp/bookshelf_tool_audit_install \
  --packages-select bookshelf_shadow_ros
```

Launch the audit in terminal 1:

```bash
source /tmp/bookshelf_tool_audit_install/setup.bash

AUDIT_DIR=$PWD/logs/policy_tool_frame_audit/$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$AUDIT_DIR"

ros2 launch bookshelf_shadow_ros policy_tool_frame_audit.launch.py \
  output_dir:="$AUDIT_DIR" \
  use_sim_time:=true \
  tf_max_age_s:=0.0
```

Replay only recorded transforms and clock in terminal 2:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

BAG_DIR=$PWD/data/real_robot_audits/marker_book_grasp_01_2026-08-05/marker_book_grasp_01/rosbag
ros2 bag play "$BAG_DIR" --clock --topics /tf /tf_static
```

The output is `policy_tool_frame_audit.json`. Distance agreement is only a
ranking signal: candidate frame semantics and gripper geometry must still be
reviewed before copying a transform into the calibrated adapter configuration.

## Extract the simulator policy-tool transform

The configured 107 mm IK offset and 75 mm book offset are not sufficient to
recover the trained control frame because the book is spawned from the
simulated finger midpoint. The residual environment captures the resulting
translation and orientation after every reset in `_book_offset_tool` and
`_book_rel_quat_tool`. `extract_sim_policy_tool_transform.py` samples those
captured values directly and converts the Isaac cuboid-root axes into the
semantic policy-book frame.

Run the final-training reset distribution on the Alienware PC:

```bash
cd /home/chris/RL/bookshelf

SIM_TOOL_DIR=$PWD/logs/sim_policy_tool_transform/$(date +%Y-%m-%d_%H-%M-%S)

PYTHONPATH=$PWD/source/bookshelf:$PWD/ros2/bookshelf_shadow_ros \
~/isaacsim/python.sh scripts/extract_sim_policy_tool_transform.py \
  --task Bookshelf-Residual-Direct-v0 \
  --num_envs 256 \
  --resets 8 \
  --seed 42 \
  --profile training_final \
  --output_dir "$SIM_TOOL_DIR" \
  --headless
```

For a zero-noise reference, repeat with `--profile nominal` and a different
output directory. Each run writes a JSON summary and compressed NPZ containing
all sampled `T_policy_book_policy_tool` transforms.

Combine the simulator representative with the measured xArm grasp and the
latest TF audit:

```bash
cd /home/chris/RL/bookshelf

SIM_SUMMARY=$(find logs/sim_policy_tool_transform \
  -name sim_policy_tool_transform_summary.json \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

REAL_AUDIT=$(find logs/policy_tool_frame_audit \
  -name policy_tool_frame_audit.json \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

OUTPUT=$PWD/logs/derived_xarm_policy_tool/$(date +%Y-%m-%d_%H-%M-%S)/xarm_policy_tool_transform.json

PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
python3 scripts/derive_xarm_policy_tool_transform.py \
  --sim_summary "$SIM_SUMMARY" \
  --real_audit "$REAL_AUDIT" \
  --tcp_frame link_tcp \
  --output "$OUTPUT"
```

The derived transform is labelled `derived_unverified_sim_to_xarm`. The script
does not modify adapter configuration, start IK, or authorize execution.

## Shadow-only policy-tool candidate checks

`policy_observation_adapter_policy_tool_candidate.yaml` contains the nominal
derived transform as a dedicated experiment candidate. It is deliberately
labelled `derived_unverified_sim_to_xarm_nominal_2026_08_06`, and its only
consumer is the read-only shadow pipeline. The default calibrated adapter still
uses the identity transform and fails closed.

The candidate represents:

```text
T_link_eef_policy_tool = T_link_eef_link_tcp * T_link_tcp_policy_tool
translation = [0.0091193265, -0.0469589570, 0.1226008212] m
quaternion  = [-0.7018277662, -0.0327964714, -0.0029489766, 0.7115851892]
```

No result below authorizes IK, planning, collision checking, trajectory
execution, or gripper motion.

### Check 1: recorded real pose through the complete shadow stream

Build the package after changing the launch/configuration files.

Alienware terminal:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

colcon \
  --log-base /tmp/bookshelf_candidate_log \
  build \
  --base-paths ros2/bookshelf_shadow_ros \
  --build-base /tmp/bookshelf_candidate_build \
  --install-base /tmp/bookshelf_candidate_install \
  --packages-select bookshelf_shadow_ros
```

Alienware terminal 1:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash
source /tmp/bookshelf_candidate_install/setup.bash

AUDIT_DIR=$PWD/logs/policy_tool_candidate_checks/recorded_$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$AUDIT_DIR"

ros2 launch bookshelf_shadow_ros policy_calibrated_static_shadow.launch.py \
  adapter_config:=$PWD/ros2/bookshelf_shadow_ros/config/policy_observation_adapter_policy_tool_candidate.yaml \
  policy_bundle:=$PWD/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz \
  audit_output_dir:="$AUDIT_DIR" \
  audit_samples:=300 \
  use_sim_time:=true
```

Alienware terminal 2:

```bash
cd /home/chris/RL/bookshelf
source /opt/ros/humble/setup.bash

BAG_DIR=$PWD/data/real_robot_audits/marker_book_grasp_01_2026-08-05/marker_book_grasp_01/rosbag

ros2 bag play "$BAG_DIR" \
  --clock \
  --topics /joint_states /tf /tf_static
```

The recorded marker view is not a pre-insertion pose, so clipped observations
or saturated actions are diagnostic rather than evidence that the candidate is
wrong. The completed `policy_stream_summary.json` must contain:

```text
hardware_commanded: false
policy_tool_transform_statuses:
  derived_unverified_sim_to_xarm_nominal_2026_08_06
```

After terminal 1 writes the completed summary, run the concise offline check:

Alienware terminal:

```bash
cd /home/chris/RL/bookshelf

SUMMARY=$(find logs/policy_tool_candidate_checks \
  -name policy_stream_summary.json \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

python3 scripts/check_policy_tool_candidate_recorded.py "$SUMMARY"
```

### Check 2: hypothetical calibrated pre-insertion pose

This check needs no ROS graph, rosbag, camera, or robot. It reconstructs the
calibrated pre-insertion target and evaluates VecNormalize, the deterministic
PPO actor, nominal controller, residual scaling, and final bounded delta.

Alienware terminal:

```bash
cd /home/chris/RL/bookshelf

OUTPUT=$PWD/logs/policy_tool_candidate_checks/hypothetical_$(date +%Y-%m-%d_%H-%M-%S)/hypothetical_preinsert_report.json

PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
python3 scripts/check_policy_tool_candidate_preinsert.py \
  --config $PWD/ros2/bookshelf_shadow_ros/config/policy_observation_adapter_policy_tool_candidate.yaml \
  --policy-bundle $PWD/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz \
  --output "$OUTPUT"
```

The concise terminal output reports transform round-trip error, expected and
unexpected observation clipping, saturated PPO actions, release state, and the
final pass/review result. `rear_to_mouth` and `front_to_back` are expected to
clip at the configured 30 mm standoff. Any other clipped observation or a
release request marks the offline candidate as requiring review.

Run the candidate-specific pure test with:

Alienware terminal:

```bash
cd /home/chris/RL/bookshelf

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
python3 -m pytest \
  ros2/bookshelf_shadow_ros/test/test_policy_tool_candidate_check.py \
  -q
```

### Simulator-equivalence check

This comparison does not insert an xArm model into Isaac Sim. Isaac retains the
native Franka policy tool used during training. The script samples that native
tool-to-book relationship, constructs the same semantic pre-insertion state,
and compares its 12D observation, VecNormalize output, actor mean, bounded PPO
action, and saturation pattern with the xArm virtual-tool candidate report.

It also records the simulator's ordinary nominal reset observations as a
separate diagnostic. Those reset observations are not used as the equivalence
pass criterion because reset and exact pre-insertion are different states.

Alienware terminal:

```bash
cd /home/chris/RL/bookshelf

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
python3 -m pytest \
  ros2/bookshelf_shadow_ros/test/test_candidate_sim_equivalence.py \
  -q

CANDIDATE_REPORT=$(find logs/policy_tool_candidate_checks \
  -name hypothetical_preinsert_report.json \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

OUTPUT_DIR=$PWD/logs/policy_tool_candidate_sim_equivalence/$(date +%Y-%m-%d_%H-%M-%S)

PYTHONPATH=$PWD/source/bookshelf:$PWD/ros2/bookshelf_shadow_ros \
~/isaacsim/python.sh scripts/compare_policy_tool_candidate_to_sim.py \
  --task Bookshelf-Residual-Direct-v0 \
  --num_envs 256 \
  --resets 4 \
  --seed 42 \
  --candidate_report "$CANDIDATE_REPORT" \
  --policy_bundle $PWD/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz \
  --output_dir "$OUTPUT_DIR" \
  --headless
```

The comparison passes only when the xArm candidate observation and normalized
observation lie inside the nominal simulator pre-insertion envelope, all six
bounded actions match, and the saturation pattern agrees. It remains a frame
and inference equivalence check, not physical calibration, IK, collision, or
execution validation.

## Export the trained actor

Run on the local training PC:

```bash
cd /home/chris/RL/bookshelf

PYTHONPATH=$PWD/source/bookshelf ~/isaacsim/python.sh \
  scripts/export_shadow_policy_bundle.py \
  --checkpoint \
    logs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/model.zip \
  --vecnormalize \
    logs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/model_vecnormalize.pkl \
  --output \
    data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz
```

The export fails if the NumPy actor differs from deterministic SB3 inference by
more than `1e-5`. The ROS node needs only Python, NumPy, and ROS 2; Riot does not
need Stable-Baselines3 or PyTorch.

## Build on Riot

The Riot machine previously rejected editable setuptools installs, so use a
standard colcon build without `--symlink-install`:

```bash
source /opt/ros/humble/setup.bash

mkdir -p /home/riot/Chris/bookshelf_shadow_ws

colcon \
  --log-base /home/riot/Chris/bookshelf_shadow_ws/log_standard \
  build \
  --base-paths /home/riot/Chris/Bookshelf-residual-RL/ros2/bookshelf_shadow_ros \
  --build-base /home/riot/Chris/bookshelf_shadow_ws/build_standard \
  --install-base /home/riot/Chris/bookshelf_shadow_ws/install_standard \
  --packages-select bookshelf_shadow_ros
```

## Run without motion

Keep the existing hardware/camera bringup running with RViz disabled:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash

ros2 launch bookshelf_policy_ros marker_vision_bringup.launch.py show_rviz:=false
```

The confirmed lab bringup happens to retain `marker_vision` in its name. The
shadow detector and adapter do not subscribe to its marker topics; this launch
is reused only for the robot state, D435 streams, and hand-eye TF.

In a second Riot terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_shadow_ws/install_standard/setup.bash

ros2 launch bookshelf_shadow_ros policy_hardware_shadow.launch.py \
  policy_bundle:=/home/riot/Chris/Bookshelf-residual-RL/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz
```

The full shadow launch now starts a subscriber-only audit by default. It
correlates the detector output, `/bookshelf_policy/slot_pose_base`, estimated
book pose, raw metrics, 12D observation, policy action, and
nominal/residual/final deltas. It never publishes a motion command.

Create a persistent report directory before launching:

```bash
AUDIT_DIR=/home/riot/Chris/Bookshelf-residual-RL/logs/policy_stream_audit/$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$AUDIT_DIR"

ros2 launch bookshelf_shadow_ros policy_hardware_shadow.launch.py \
  policy_bundle:=/home/riot/Chris/Bookshelf-residual-RL/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz \
  audit_output_dir:="$AUDIT_DIR" \
  audit_samples:=1200
```

At 20 Hz, 1200 samples take approximately one minute. The report contains:

- camera-derived slot confidence and width;
- slot and book pose repeatability in `link_base`;
- mean base-frame directions of slot `+X`, `+Y`, and `+Z`;
- every raw metric and clipped 12D observation component;
- PPO action and nominal, residual, and final diagnostic deltas;
- observation clipping fraction, invalid reasons, and book-pose source.

If a friend measures the physical opening, pass it in metres:

```bash
ros2 launch bookshelf_shadow_ros policy_hardware_shadow.launch.py \
  policy_bundle:=/home/riot/Chris/Bookshelf-residual-RL/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz \
  audit_output_dir:="$AUDIT_DIR" \
  audit_samples:=1200 \
  reference_slot_width_m:=0.038
```

Replace `0.038` with the actual measurement. The JSON report will then include
signed and absolute slot-width error. Without this value, the report measures
repeatability but cannot claim absolute physical accuracy.

If the full shadow pipeline is already running, attach only the auditor:

```bash
ros2 launch bookshelf_shadow_ros policy_stream_audit.launch.py \
  output_dir:="$AUDIT_DIR" \
  target_samples:=1200 \
  reference_slot_width_m:=0.0
```

The output files are:

```text
policy_stream_samples.csv
policy_stream_summary.json
```

The current markerless smoke configuration still labels its fixed
`link_eef -> book` estimate as `approximate_smoke_only`. Therefore a successful
stream audit proves software integration and temporal stability, but not the
physical accuracy of book-relative policy metrics.

## Acceptance checks

```bash
ros2 topic echo --once /slot_detector/confidence
ros2 topic echo --once /slot_detector/slot_width
ros2 topic echo --once /bookshelf_policy/slot_pose_base
ros2 topic echo --once /bookshelf_policy/book_pose_base
ros2 topic echo --once /bookshelf_policy/observation_valid
ros2 topic echo --once /bookshelf_policy/adapter_debug --field data
ros2 topic echo --once /bookshelf_shadow/inference_valid
ros2 topic echo --once /bookshelf_shadow/policy_debug --field data

ros2 node info /policy_shadow_inference
ros2 node info /policy_stream_audit
ros2 topic list | grep -E "^/bookshelf_policy/action$|trajectory|joint_trajectory"
```

Expected safety evidence:

- `/bookshelf_shadow/inference_valid` is `True` only with fresh, valid paired
  observation and raw-metric messages.
- The policy node publishes only shadow diagnostic topics.
- The policy node has no robot-control service clients or action clients.
- `/bookshelf_policy/action` is absent.

## Hardware-free validation

Run the offline suite before requesting any physical experiment:

```bash
cd /home/chris/RL/bookshelf

PYTHONPATH=$PWD/ros2/bookshelf_shadow_ros \
  python3 scripts/validate_shadow_pipeline_offline.py
```

It writes a timestamped report under `logs/offline_shadow_validation/` with:

- synthetic policy-response sweeps for lateral, vertical, yaw, pitch, and
  insertion-depth errors;
- Monte Carlo sensitivity to book-frame translation and rotation error;
- deterministic actor checks;
- stale, malformed, non-finite, low-confidence, and invalid-width rejection;
- exact portable-controller versus simulator-config constant parity;
- a static source audit for robot-control clients, action clients, command
  message imports, and command namespaces.

This report uses no camera or robot. Passing it does not identify the physical
`link_eef -> book` transform, shelf depth, slot target offset, RGB-D ground-truth
error, or contact dynamics.

### Audit the existing RGB-D rosbag

After rebuilding the package, replay the saved bag against a detector-only
launch. This starts no robot, adapter, or policy node.

Local PC terminal 1:

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/home/chris/RL/bookshelf/ros2/install/bookshelf_shadow_ros:$AMENT_PREFIX_PATH
export PYTHONPATH=/home/chris/RL/bookshelf/ros2/install/bookshelf_shadow_ros/lib/python3.10/site-packages:$PYTHONPATH

ros2 launch bookshelf_shadow_ros slot_detector_bag_audit.launch.py \
  output_dir:=/home/chris/RL/bookshelf/logs/slot_detector_bag_audit \
  target_samples:=1200
```

Local PC terminal 2:

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/home/chris/RL/bookshelf/ros2/install/bookshelf_shadow_ros:$AMENT_PREFIX_PATH
export PYTHONPATH=/home/chris/RL/bookshelf/ros2/install/bookshelf_shadow_ros/lib/python3.10/site-packages:$PYTHONPATH

ros2 bag play \
  /home/chris/RL/bookshelf/data/real_rgbd/slot_view_01_complete/slot_view_01
```

The audit writes raw samples and summary statistics for valid fraction, width,
3D position, orientation, and frame-to-frame changes. Because the bag has no
measured physical ground truth, these numbers establish repeatability only,
not absolute slot-pose accuracy.

The explicit shadow-only `AMENT_PREFIX_PATH` and `PYTHONPATH` avoid importing
the separate `bookshelf_policy_ros/launch` Python package as ROS 2's own
`launch` module in this merged local workspace.
