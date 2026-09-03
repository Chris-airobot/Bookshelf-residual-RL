# Real xArm7 Bookshelf Session

## Purpose

Primary: July residual PPO with the frozen per-grasp EEF-to-book calibration and
the simulation-semantic held-gripper observation. Backup: F1-40M, selected only
deliberately after the first validation run. **Do not use `fixed_fallback` as the
main validation trial.**

## Terminal 1 — build and launch

```bash
cd ~/Chris/bookshelf-unified
source /opt/ros/humble/setup.bash
# Use local_setup for the required xArm underlay; do not source the stale setup.bash.
source ~/Chris/ros2_ws/install/local_setup.bash
source ~/Chris/ros2_ws/install_depth_fix/local_setup.bash
scripts/ros2/build_xarm_experiment.sh
source ~/Chris/bookshelf-unified/.ros2_ws/install/local_setup.bash

ros2 launch bookshelf_simple_experiment_ros \
  real_experiment_operator.launch.py \
  robot_ip:=192.168.1.209 \
  allow_execution:=true \
  shadow_full_sequence:=false \
  show_rviz:=true
```

This starts one hardware/MoveIt/Servo/camera stack, slot and preinsert nodes,
the July PPO controller, operator console, table scene, and exactly one RViz.

## Terminal 2 — full rosbag

```bash
cd ~/Chris/bookshelf-unified
source /opt/ros/humble/setup.bash
source ~/Chris/ros2_ws/install/local_setup.bash
source ~/Chris/ros2_ws/install_depth_fix/local_setup.bash
source ~/Chris/bookshelf-unified/.ros2_ws/install/local_setup.bash

mkdir -p ~/BookshelfFiles/experiment_logs/full_real_bags
BAG=~/BookshelfFiles/experiment_logs/full_real_bags/full_real_$(date +%Y%m%d_%H%M%S)
ros2 bag record -a \
  -x '.*(compressed|theora).*' \
  -o "$BAG"
```

Start recording before the experiment. Stop it with `Ctrl+C` after the trial.

## Local Alienware dry run

**FAKE HARDWARE ONLY — NEVER USE THIS LAUNCH FOR THE REAL ROBOT.**

Terminal 1:

```bash
cd ~/Chris/bookshelf-unified
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
scripts/ros2/build_xarm_experiment.sh
source ~/Chris/bookshelf-unified/.ros2_ws/install/local_setup.bash
ros2 launch bookshelf_simple_experiment_ros \
  offline_full_sequence_rehearsal.launch.py
```

Terminal 2 (optional log capture):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/Chris/bookshelf-unified/.ros2_ws/install/local_setup.bash
mkdir -p ~/BookshelfFiles/experiment_logs/local_dry_runs
ros2 bag record -a -x '.*(compressed|theora).*' \
  -o ~/BookshelfFiles/experiment_logs/local_dry_runs/dry_run_$(date +%Y%m%d_%H%M%S)
```

Practice this sequence:

```text
G → E → S → L → E → O → C
CHECK: PER-GRASP EEF->BOOK FROZEN
P → E → I → H → E
```

## Operator sequence

Use this exact sequence:

```text
G  E  S  L  E  O  C
```

- After `G` and `L`, inspect the planned trajectory in RViz before `E`.
- After `O`, physically load the book; then press `C`.

### Critical check after C

Wait approximately three seconds and require:

```text
PER-GRASP EEF->BOOK FROZEN
```

If the log instead reports `fixed_fallback`, **do not use that run as the main
validation trial**. Check marker visibility, valid sample count, and the
calibration warning, then retry the grasp/calibration. Do not bypass a physical
stop or automatically retry motion.

After successful calibration, continue:

```text
P  E  I  H  E
```

Inspect RViz after `P` and `H` before pressing `E`. `I` starts the PPO sequence;
it does not run automatically after preinsert execution.

## First corrected run

Question: **Does July plus both observation fixes continue past the previous
premature-release region and eventually release correctly?** The failed trial
released near `rear_to_mouth = -59 mm`. Treat this first run as validation, not
as a statistical experiment.

Watch only that calibration freezes, INSERT continues without immediate
release, learned release opens the gripper, retreat completes, empty close and
PUSH complete, and the reviewed `H` return remains safe.

## Backup F1-40M policy

- July primary: `/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz`
- F1-40M backup: `/home/riot/BookshelfFiles/trained_models/bookshelf_fresh_F1_40M_backup_actor.npz`

Do not switch for the first validation run. To select F1-40M, stop the experiment
launch normally and relaunch Terminal 1 with this additional launch argument:

```bash
actor_path:=/home/riot/BookshelfFiles/trained_models/bookshelf_fresh_F1_40M_backup_actor.npz
```

To switch back, omit `actor_path` (the launch default is July), or explicitly use:

```bash
actor_path:=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz
```

Never change policy while an experiment launch is running.

## After each trial

- Stop the bag cleanly with `Ctrl+C` when appropriate.
- Record success/failure, per-grasp versus fallback, and the obvious failure stage.
- Retain the `simple_policy_*` JSON log.
- Check the latest bag with:

```bash
ros2 bag info "$(find ~/BookshelfFiles/experiment_logs/full_real_bags -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
```

## Emergency / abort

If motion appears unsafe, use the physical E-stop or established execution stop
as appropriate. Do not automatically retry after a physical stop.
