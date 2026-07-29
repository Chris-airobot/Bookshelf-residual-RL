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

## Acceptance checks

```bash
ros2 topic echo --once /slot_detector/confidence
ros2 topic echo --once /slot_detector/slot_width
ros2 topic echo --once /bookshelf_policy/observation_valid
ros2 topic echo --once /bookshelf_policy/adapter_debug --field data
ros2 topic echo --once /bookshelf_shadow/inference_valid
ros2 topic echo --once /bookshelf_shadow/policy_debug --field data

ros2 node info /policy_shadow_inference
ros2 topic list | grep -E "^/bookshelf_policy/action$|trajectory|joint_trajectory"
```

Expected safety evidence:

- `/bookshelf_shadow/inference_valid` is `True` only with fresh, valid paired
  observation and raw-metric messages.
- The policy node publishes only shadow diagnostic topics.
- The policy node has no robot-control service clients or action clients.
- `/bookshelf_policy/action` is absent.
