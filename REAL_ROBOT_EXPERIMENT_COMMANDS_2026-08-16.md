# Supervised Real-Robot Experiment Commands - 2026-08-16

This is the current operator sequence for the xArm7 bookshelf experiment.
People must remain beside the robot with immediate stop access throughout every
movement. "One-shot" means one trajectory submission per executor process; it
does not mean the experiment is unattended.

The committed executor configuration is deliberately unable to move hardware.
Only create the external physical configuration after the book transform,
collision scene, and complete RViz path have been checked against the real setup.

## Stop Conditions

- Stop if the physical book is not rigidly held in the reviewed grasp.
- Stop if live and frozen slot poses disagree.
- Stop if the robot, book, shelf, table, target, or path differs from RViz.
- Stop if any person cannot reach the robot stop control immediately.
- Never loosen a failed trajectory, IK, scene, provenance, or joint-drift gate.
- Never launch the local policy executor during the global MoveIt approach.

## Fixed Paths

```bash
cat > /tmp/bookshelf_trial_env.sh <<'EOF'
export REPO=/home/riot/Chris/bookshelf-unified
export INSTALL=/home/riot/Chris/bookshelf_unified_ws/install
export TRIAL_NAME=physical_trial_20260816_01
export TRIAL_SLOT_CONFIG=/home/riot/BookshelfFiles/experiment_logs/environment_checks/physical_trial_20260813_01/trial_static_slot.yaml
export SCENE_CONFIG=/home/riot/BookshelfFiles/experiment_configs/physical_trial_20260813_01_bookshelf_scene.yaml
export POLICY=/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz
export ENVELOPE=/home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json
EOF

source /tmp/bookshelf_trial_env.sh
```

Use a new `TRIAL_NAME` for every physical attempt. Regenerate the pre-insertion
plan after the book is attached; never reuse an earlier trajectory. Source
`/tmp/bookshelf_trial_env.sh` at the start of every terminal below.

## 1. Riot PC Terminal 1 - Observation Bringup and Logging

This single launch starts the robot interface, camera, TF, MoveIt, automatic
logging, slot detector, and frozen-slot check. It starts no planner or executor.

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash
source /tmp/bookshelf_trial_env.sh

ros2 launch bookshelf_guarded_control_ros \
  physical_experiment_observation_bringup.launch.py \
  trial_name:="$TRIAL_NAME" \
  trial_slot_config:="$TRIAL_SLOT_CONFIG" \
  show_rviz:=false
```

Use `show_rviz:=true` only from the Riot desktop. Keep it false over SSH.

## 2. Riot PC Terminal 2 - Read-Only Readiness

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash
source /tmp/bookshelf_trial_env.sh

ros2 service list | grep -E '^/compute_ik$|^/plan_kinematic_path$|^/apply_planning_scene$'
ros2 topic echo --once /bookshelf_environment/static_slot_check_passed

ros2 node list | grep -E \
  'guarded_preinsert_executor|guarded_policy_tool_executor|policy_to_robot|cartesian_action_executor|action_executor' \
  && { echo 'STOP: execution node found'; exit 1; } \
  || echo 'PASS: no execution node'
```

All three services must exist and the frozen-slot result must be `true`.

## 3. Riot PC Terminal 2 - Fresh Global Plan Only

```bash
RUN=/home/riot/BookshelfFiles/experiment_logs/plan_only/${TRIAL_NAME}_$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "$RUN"
echo "$RUN" > /tmp/bookshelf_preinsert_run_path.txt

ros2 launch bookshelf_guarded_control_ros \
  calibrated_preinsert_spine_mount_candidate_plan_only.launch.py \
  target_config:="$TRIAL_SLOT_CONFIG" \
  scene_config:="$SCENE_CONFIG" \
  output_dir:="$RUN"
```

Leave this running while reviewing and executing. It performs collision-aware
seeded IK and planning only; it cannot command the robot.

In another terminal, require all values below to pass:

```bash
RUN=$(cat /tmp/bookshelf_preinsert_run_path.txt)

python3 - "$RUN/calibrated_preinsert_plan_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ik = r.get("ik_joint_branch") or {}
trajectory = r.get("trajectory_sanity") or {}
print("Valid:", r.get("valid"))
print("Reason:", r.get("reason"))
print("IK code:", r.get("ik_error_code"))
print("IK passed:", ik.get("passed"))
print("IK largest delta:", ik.get("largest_delta_joint"), ik.get("maximum_delta_rad"))
print("Path planned:", r.get("path_planned"))
print("Trajectory passed:", trajectory.get("passed"))
print("Duration:", trajectory.get("duration_s"))
print("Trajectory hash:", r.get("trajectory_sha256"))
print("Hardware commanded:", r.get("hardware_commanded"))
PY
```

Required: `Valid=True`, IK code `1`, both checks `True`, a nonempty trajectory
hash, and `Hardware commanded=False`.

## 4. Human RViz and Physical Review

On the Riot desktop, inspect `/display_planned_path` against the real robot.
Review the complete animated path, not only its endpoint. Confirm:

- the held-book collision box matches the attached book and grasp;
- the shelf and table boxes conservatively cover the physical obstacles;
- the target is the intended pre-insertion pose and orientation;
- every link and the held book maintain visible clearance;
- the operator and stop control are ready.

Do not continue if the candidate transform has not been physically validated.

## 5. Create the External Physical Executor Configuration

Run this only after Section 4 passes. The token and enabled configuration stay
outside Git.

```bash
source /tmp/bookshelf_trial_env.sh

DEFAULT_CONFIG=$REPO/ros2/bookshelf_guarded_control_ros/config/guarded_preinsert_executor.yaml
PHYSICAL_CONFIG=/home/riot/BookshelfFiles/experiment_configs/guarded_preinsert_executor_physical.yaml
SCENE_SHA=$(sha256sum "$SCENE_CONFIG" | awk '{print $1}')
TOKEN=$(openssl rand -hex 16)
TARGET_STATUS=derived_unverified_sim_to_xarm_spine_mount_candidate_2026_08_14

python3 - "$DEFAULT_CONFIG" "$PHYSICAL_CONFIG" "$SCENE_SHA" "$TOKEN" "$TARGET_STATUS" <<'PY'
import pathlib, sys, yaml
source, output, scene_sha, token, target_status = sys.argv[1:]
data = yaml.safe_load(open(source))
p = data["guarded_preinsert_executor"]["ros__parameters"]
p.update({
    "dry_run": False,
    "allow_execution": True,
    "planning_scene_complete": True,
    "human_trajectory_review_complete": True,
    "target_transform_physically_validated": True,
    "approval_token": token,
    "expected_scene_config_sha256": scene_sha,
    "expected_target_transform_status": target_status,
})
pathlib.Path(output).write_text(yaml.safe_dump(data, sort_keys=False))
print("Wrote reviewed external configuration:", output)
PY

chmod 600 "$PHYSICAL_CONFIG"
```

The status name contains `unverified` because it is the candidate's immutable
provenance label. Setting `target_transform_physically_validated: true` is the
separate human statement that it was checked against the real attached book.

## 6. Riot PC Terminal 3 - One Global Execution

Start the executor only after the plan review is complete:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash
source /tmp/bookshelf_trial_env.sh

PHYSICAL_CONFIG=/home/riot/BookshelfFiles/experiment_configs/guarded_preinsert_executor_physical.yaml

ros2 launch bookshelf_guarded_control_ros \
  guarded_preinsert_execute_once.launch.py \
  executor_config:="$PHYSICAL_CONFIG"
```

Confirm its startup log says the action client is active. With the operator at
the stop control, publish exactly one approval from a separate terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash
source /tmp/bookshelf_trial_env.sh

PHYSICAL_CONFIG=/home/riot/BookshelfFiles/experiment_configs/guarded_preinsert_executor_physical.yaml

TOKEN=$(python3 - "$PHYSICAL_CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["guarded_preinsert_executor"]["ros__parameters"]["approval_token"])
PY
)

ros2 topic pub --once \
  /bookshelf_preinsert/approve_once \
  std_msgs/msg/String "{data: '$TOKEN'}"

ros2 topic echo --once /bookshelf_preinsert/execution_report --field data
```

The process can submit only one trajectory even if the token is published
again. Stop after any rejection or unexpected motion; do not relaunch merely to
bypass a failed gate.

## 7. Post-Movement Checks and Supervised Local RL

After successful global movement:

1. compare live TCP and the physical book with the intended pre-insertion pose;
2. rerun frozen-slot and book-pose checks;
3. run the policy in shadow mode until activation is stable for 10 samples;
4. generate and review one local residual plan;
5. approve at most one guarded residual step;
6. inspect the robot and logs before starting a fresh process for another step.

This is supervised multi-step insertion. No software loop automatically chains
residual actions. Every additional step requires a fresh plan, human review,
and a fresh one-shot executor process.

## 8. Stop and Preserve Evidence

Stop the executor, planner, and observation launch with `Ctrl+C`. Verify the
automatic experiment directory and bag:

```bash
LATEST=$(find /home/riot/BookshelfFiles/experiment_logs -mindepth 1 -maxdepth 1 \
  -type d -name "*${TRIAL_NAME}*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "Run directory: $LATEST"
ros2 bag info "$LATEST/rosbag"
find "$LATEST" -maxdepth 2 -type f -printf '%s %p\n' | sort -n
```
