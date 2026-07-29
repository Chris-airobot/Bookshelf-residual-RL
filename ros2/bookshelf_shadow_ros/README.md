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
