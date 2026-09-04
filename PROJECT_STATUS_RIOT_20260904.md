# Bookshelf Residual-RL: Riot Real-World Project Snapshot

Snapshot date: **2026-09-04 (Australia/Melbourne)**  
Host: **Riot real-robot PC**  
Repository: `/home/riot/Chris/bookshelf-unified`  
Scope: real xArm7 deployment, real data, deployment diagnostics, and the local evidence needed to merge this record with the Alienware simulation snapshot.

This is an audit of the state found on Riot. It does not equate software completion with physical task success. In particular, `episode_complete` means that the controller reached the end of its programmed release/retreat/close/PUSH sequence; it does **not** prove that the book was physically inserted successfully.

## 1. Git and repository state

### Recorded revision

- Branch: `simple-real-experiment`
- HEAD: `05fba2eba08a5dc3eb62fa97ce7d3504528f9644`
- HEAD subject: `Prepare validated xArm7 real experiment session`
- Upstream `origin/simple-real-experiment` was at `b5de154` when inspected; the local branch contains later local commits.

Important deployment commits in the local history:

| Commit | Purpose |
|---|---|
| `8176c0d` | Initial lightweight real experiment and dual-ArUco support. |
| `03bf245` | Real xArm `GripperCommand` handling and offline IK diagnostics. |
| `fdf0334` | Singularity-aware multi-branch preinsert IK selection. |
| `e69c5e0` | One-command real experiment operator. |
| `3135a64` | Renamed the operator's conflicting `clients` member. |
| `a082d5c` | Prevented stale/out-of-order status from faking PLAN/EXECUTE transitions. |
| `3cdcabc` | Release-pose-based PUSH estimate, with immutable release pose and separate mutable PUSH estimate. |
| `8e4e57d` | Multi-cycle operator sequence, reviewed plan/execute gates, scan/loading/return workflow. |
| `1055696` | Supporting execution gate, joint poses, operator actions, and offline rehearsal. |
| `05fba2e` | Per-grasp transform and semantic-gripper deployment baseline, table-only planning scene, session README. |

### Uncommitted state at this snapshot

The working tree is deliberately **not clean**. Two safety-relevant fixes exist in source and have been built/tested, but are newer than HEAD:

1. The policy loads the exact `S`-accepted frozen slot at `I` and logs `POLICY SLOT SNAPSHOT`.
2. Per-grasp calibration counts only fresh, uniquely timestamped marker observations and blocks `P` after a failed post-`C` capture.

Modified deployment/test files:

```text
ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/per_grasp_calibration.py
ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/preinsert_node.py
ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/simple_policy_control_node.py
ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/slot_detector_node.py
ros2/bookshelf_simple_experiment_ros/config/simple_policy_control.yaml
ros2/bookshelf_simple_experiment_ros/config/simple_preinsert.yaml
ros2/bookshelf_simple_experiment_ros/launch/real_experiment_operator.launch.py
ros2/bookshelf_simple_experiment_ros/launch/simple_policy_one_step.launch.py
ros2/bookshelf_simple_experiment_ros/test/test_per_grasp_calibration.py
ros2/bookshelf_simple_experiment_ros/test/test_repeatable_trials.py
```

The `slot_detector_node.py` change only demotes a high-rate status line to debug logging; it is unrelated to the two geometry fixes.

Modified training/evaluation files, not part of the real runtime fixes:

```text
scripts/export_shadow_policy_bundle.py
scripts/sb3/play.py
scripts/sb3/train.py
source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env.py
source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env_cfg.py
```

Untracked diagnostic/training infrastructure includes `scripts/hpc/`, the `scripts/sb3/*audit*`, counterfactual, robustness, summary and VecNormalize helpers, `test_real_policy_evidence_audit.py`, and `test_targeted_dr_profiles.py`. Untracked `install/` and `log/` directories are generated artifacts and must not be committed as source. Writing this snapshot also creates the untracked documentation file `PROJECT_STATUS_RIOT_20260904.md`.

At the last deployment-fix validation, `git diff --check` passed. No commit or push was made after the frozen-slot and marker-freshness fixes.

## 2. Real hardware setup

### Robot and control

- Robot: UFactory **xArm7**, seven arm joints plus xArm gripper.
- Controller address: `192.168.1.209`.
- Robot base/planning frame: `link_base`; world is fixed to `link_base`.
- End-effector frames: `link_eef`, `link_tcp`; MoveIt group: `xarm7`; robot model ID: `UF_ROBOT`.
- ROS: ROS 2 Humble.
- MoveIt: official xArm ROS 2 MoveIt stack from `/home/riot/Chris/ros2_ws/install`.
- Driver seen during physical bringup: xArm SDK `1.15.2`, controller firmware `v2.2.0`.
- Arm trajectory controller: `/xarm7_traj_controller`.
- Gripper action: `/xarm_gripper/gripper_action`, type `control_msgs/action/GripperCommand`.
- Gripper convention: open position `0.0`, closed position `0.85`, configured `max_effort=0.0`.
- MoveIt Servo input/status: `/servo_server/delta_twist_cmds`, `/servo_server/status`; output is `/xarm7_traj_controller/joint_trajectory`.
- Servo publishes every `0.034 s`, accepts speed units, checks collision at `10 Hz`, starts singularity deceleration at condition number `17`, hard-stops at `30`, and uses `0.1 rad` joint-limit margin. Self/scene proximity thresholds are `0.01/0.02 m`.
- Real controller rate: Servo command loop `30 Hz`; PPO update loop `20 Hz`; bounded-error command horizon `0.20 s`; maximum linear/angular speeds `0.025 m/s` and `0.10 rad/s`.

### Camera and perception

- Camera observed in the September 4 logs: Intel RealSense **D435**, serial `242322078188`, firmware `5.17.3.10`.
- RealSense ROS `4.55.1`, librealsense `2.56.4`.
- Color and depth: `640x480 @ 30 Hz`; depth aligned to color; sync and point cloud enabled.
- Slot detector inputs: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`.
- Hand-eye static transform `T_link_eef_camera_link`:
  - translation `[0.064694, -0.017286, 0.018312] m`
  - quaternion XYZW `[0.711047, -0.001225, 0.703143, -0.000110]`
- Held-book perception: ArUco `DICT_ARUCO_ORIGINAL`, ID `0`, black-square size `0.039 m`; output frame `target_book_center`.
- Marker centre in book frame: `[-0.0800, +0.0375, +0.0665] m`.
- Marker axes in the book frame: marker `+X = book -Y`, marker `+Y = book +Z`, marker `+Z = book -X`.

### Physical geometry and frames

- Book dimensions in semantic book axes `[depth, thickness, height]`: **`[0.156, 0.034, 0.236] m`**.
- Slot frame: origin at the detected slot mouth; `+X` enters the shelf, `+Z` is up, and `+Y` completes the right-handed frame.
- Book frame: local `+X` is book depth, `+Y` thickness/lateral, `+Z` height/up.
- Configured shelf/slot depth: **`0.200 m`**. The `0.250 m` visual slot height is a visualization value, not a separately established measured slot height.
- Slot width is detected per trial, accepted range `0.020–0.090 m`. The old approved stationary slot width is `0.0378443301 m`; a later moved-shelf accepted width was `0.0433247350 m`.
- Preinsert book-centre target in slot coordinates is nominally `[-0.108, 0, +0.006] m`: half the `0.156 m` book depth plus `0.030 m` standoff, with `+0.006 m` vertical offset and slot-aligned orientation.
- Validated table collision box: size `[1.50, 0.60, 0.05] m`, base-frame centre `[0.75, 0, -0.025] m`, identity orientation. The combined real launch loads this table-only scene. The older coarse shelf keep-out was `[0.30, 0.95, 0.40] m` centred `[0.15, 0, 0]` from the slot, but it is deliberately not loaded during local insertion.

### Policy artifacts

- Primary/deployed actor: `/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz`
- July actor SHA-256: `75773dde0edabebcb525469c2e2b1cf868d7724f45a9f661f994cd8847a0ab19`
- Contract: 12-D observation, 6-D action, deterministic ReLU actor with embedded VecNormalize statistics.
- Backup only: `/home/riot/BookshelfFiles/trained_models/bookshelf_fresh_F1_40M_backup_actor.npz`, SHA-256 `57741afb8434f3fa609d7f0cf15241fb0b1cdc302ddc94a7c79e78dee0d4ec05`.
- The launch default remains July; F1-40 is not selected automatically.

## 3. Operator workflow

### Build and launch

Terminal 1 owns the only hardware, MoveIt, Servo, camera, perception, table scene, preinsert, policy, operator console, and RViz processes:

```bash
cd ~/Chris/bookshelf-unified
source /opt/ros/humble/setup.bash
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

The top-level launch includes `bookshelf_policy_ros/physical_hardware_bringup.launch.py`; hardware must not be launched a second time. Physical bringup owns the single RViz instance. The preinsert include receives `show_rviz:=false`.

Terminal 2 records the complete ROS graph:

```bash
cd ~/Chris/bookshelf-unified
source /opt/ros/humble/setup.bash
source ~/Chris/ros2_ws/install/local_setup.bash
source ~/Chris/ros2_ws/install_depth_fix/local_setup.bash
source ~/Chris/bookshelf-unified/.ros2_ws/install/local_setup.bash
mkdir -p ~/BookshelfFiles/experiment_logs/full_real_bags
BAG=~/BookshelfFiles/experiment_logs/full_real_bags/full_real_$(date +%Y%m%d_%H%M%S)
ros2 bag record -a -x '.*(compressed|theora).*' -o "$BAG"
```

Start the bag before the trial and stop it cleanly with Ctrl+C afterward.

### State-gated key sequence

The current sequence is:

```text
G  E  S  L  E  O  C  [verify calibration]  P  E  I  H  E
```

| Key | Effect |
|---|---|
| `G` | PLAN the saved scan joint pose. No motion. |
| `E` | Execute the reviewed scan plan. |
| `S` | Accept/freeze the current RGB-D slot and save `/tmp/bookshelf_simple_frozen_slot.yaml`. |
| `L` | PLAN the saved loading joint pose. No motion. |
| `E` | Execute the reviewed loading plan. |
| `O` | Open the gripper for manual book loading. |
| `C` | Close around the book; on successful close, begin the per-grasp marker capture. |
| `P` | PLAN preinsert using the accepted slot and frozen per-grasp transform. No execution. |
| `E` | Execute only the already-reviewed preinsert plan. |
| `I` | Start the July PPO INSERT and automatic post-release sequence. |
| `H` | PLAN the return-to-loading trajectory after controller episode completion. |
| `E` | Execute the reviewed return; the return finish action opens the gripper. |

After every `G`, `L`, `P`, and `H`, inspect the RViz trajectory before pressing `E`. `E` is state-locked until a current plan is ready. A new plan invalidates prior execute authorization. `P` cannot dispatch execution.

After `C`, require both a calibration `PASS` and the visible line:

```text
PER-GRASP EEF->BOOK FROZEN
```

If calibration fails, `P` now fails with a clear reason and asks the operator to press `C` to retry. Do not treat `fixed_fallback` as a main validation trial.

At `I`, require:

```text
POLICY SLOT SNAPSHOT source=frozen_accepted:/tmp/bookshelf_simple_frozen_slot.yaml ...
```

## 4. Real deployment architecture

1. **Slot detection and freeze.** `rgbd_slot_detector` detects the free gap from aligned RGB-D. `S` copies the current valid candidate into preinsert state and writes its base-frame pose, width and confidence to `/tmp/bookshelf_simple_frozen_slot.yaml`. Live detection can continue for visualization, but the accepted pose is immutable for that preparation cycle.
2. **Scan/loading/return poses.** `simple_preinsert` loads reviewed seven-joint scan and loading YAML files. Each `G/L/H` request creates a collision-checked plan; a separate `E` is required to execute it.
3. **Per-grasp calibration.** Successful `C` starts 30 capture reads, nominally at 10 Hz. For each fresh marker observation, the node computes `T_eef_book_i = inverse(T_base_eef_i) @ T_base_book_marker_i`. Translation is robustly centred/averaged and rotation is averaged by SVD; samples outside `5 mm` or `5 deg` are rejected. At least 20 retained, fresh, unique samples are required.
4. **Preinsert planning.** The same active `T_eef_book` constructs the desired Cartesian book pose and attached-book collision geometry. A 24-seed redundant IK search deduplicates within `0.01 rad`, checks joint limits/collision/plannability, and samples 11 points over a predicted `0.10 m` insertion. Candidates require at least `0.05 rad` joint margin and predicted maximum Jacobian condition no greater than `27`; maximum condition ranks first, with transition cost used within a `1.0` condition band. The selected joint target goes through the existing plan/review/execute gate.
5. **Policy slot handoff.** At `/bookshelf_simple/start_policy`, the controller loads the exact frozen YAML, rejects a missing/invalid/wrong-frame file, snapshots that pose into `geometry.transform_base_slot`, recomputes retreat direction, and logs the source and transform. INSERT observation, nominal control, release geometry, retreat direction, and PUSH all use that same snapshot.
6. **Book pose during INSERT.** `T_base_book = T_base_eef @ T_eef_book_per_grasp`. The marker is not needed once the transform is frozen. The fixed reviewed transform is retained as an explicit fallback only for workflows in which no post-close capture was requested; a failed operator `C` capture blocks `P` rather than silently falling back.
7. **Observation.** Raw 12-D order is:

   ```text
   [mode, rear_to_mouth, front_to_back, lateral_error, z_error, yaw_error,
    tool_to_book_x, tool_to_book_y, tool_to_book_z, gripper_open, tilt_x, tilt_y]
   ```

   `rear_to_mouth` is the minimum book-corner slot X; `front_to_back = 0.200 - maximum book-corner X`; lateral is `-book_center_y`; tool deltas are policy-tool minus book centre in the slot frame; tilts are book-up-axis X/Y components. Scaling divisors are `[1, .08, .08, .05, .05, 30 deg, .25, .25, .25, 1, 1, 1]`, followed by clipping to `[-1,1]` and then the actor's embedded VecNormalize.
8. **Semantic gripper.** The physical `drive_joint` aperture remains recorded as `measured_gripper_open`. During held-book INSERT only, the actor receives `policy_gripper_open = 0.009838026859259968`, matching July simulation semantics. After actual opening, open semantics are `1.0`; physical commands and driver feedback are unchanged.
9. **Nominal plus residual INSERT.** Nominal insertion is `+0.0010 m` per policy decision (`+0.0007 m` near the mouth), plus alignment corrections. Five residual scales are `[0.0020 m, 0.0010 m, 0.0015 m, 0.35 deg, 0.30 deg]`; final limits are `[0.0080 m, 0.0030 m, 0.0070 m, 0.8 deg, 0.6 deg]`. `command_scale` is applied to the combined target; the real operator launch uses the launch/config path already validated for the experiment. Slot `+X` is the nominal insertion direction.
10. **Release.** Action order is residual `[X,Y,Z,yaw,pitch]` plus release. During mode 0/INSERT, release is requested when the deterministic action's sixth component is strictly greater than `0.5`. There is no deployment geometry release guard or consecutive-step requirement.
11. **Gripper and retreat.** A successful release sends open position `0.0`; action-level success is `GoalStatus.STATUS_SUCCEEDED`, not xArm's unreliable `reached_goal` flag. Retreat then moves `0.09 m` along slot `-X`, retaining the original geometric completion condition and allowing `15 s` for Servo singularity deceleration. The empty gripper then closes to `0.85`.
12. **PUSH.** The immutable `released_book_transform` is copied to mutable `push_book_transform`. The controller approaches from outside along slot `+X`, computes a geometric contact estimate from the release pose, and commands the existing **30 mm** nominal book push plus residual PPO. `push_x_uncertainty_m=0.005` documents physical X uncertainty only; it does not add 5 mm of commanded travel. There is no contact sensor, so logged “contact” is explicitly `release_geometry_no_contact_sensor`. PUSH timeout is `90 s`.
13. **Return.** Controller `episode_complete` leaves the operator at the return gate. `H` plans back to loading; `E` executes the reviewed trajectory and finishes the cycle. No next policy run starts automatically.

## 5. Historical real experiments

The evidence audit found **25** `policy_step.jsonl` files. The table below reports controller evidence only unless the note explicitly says “operator-reported.” `Δrear` is final minus initial `rear_to_mouth`, not independently measured physical book travel.

| Policy log suffix | Class | INSERT decisions | Δrear mm | Release | PUSH | `episode_complete` | Known note |
|---|---:|---:|---:|---:|---:|---:|---|
| `20260827_221044` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221119` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221138` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221347` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221356` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221402` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221413` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221549` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_221624` | partial after release | 774 | 105.5 | yes | no | no | Learned release at step 773, action `0.514665`; failed: gripper open action unavailable. |
| `20260827_230910` | one-step debug | 1 | 0.0 | no | no | no | Diagnostic only. |
| `20260827_230956` | partial after release | 1,130 | 103.9 | yes | no | no | Operator saw physical opening/release; failed because `reached_goal=false` was incorrectly treated as action failure. |
| `20260827_232929` | partial INSERT | 3,001 | 117.5 | no | no | no | Operator saw physical stall; Servo status 1 was approaching-singularity deceleration. |
| `20260901_231415` | partial INSERT | 3,001 | 0.0 | no | no | no | No encoded physical outcome. |
| `20260901_233343` | partial INSERT | 2,233 | 0.0 | no | no | no | Log ends without a terminal event. |
| `20260901_234542` | partial after release | 963 | 114.7 | yes | no | no | Retreat began physically, then old timeout failed: `scripted retreat timed out`. |
| `20260901_235523` | complete controller | 898 | 114.8 | yes | yes | yes | Physical placement outcome not encoded. |
| `20260902_001927` | complete controller | 914 | 114.8 | yes | yes | yes | Operator-reported early/incomplete insertion; release at rear `-58.967 mm`, action `0.501103`; paired bag. |
| `20260903_233842` | complete controller | 569 | 111.7 | yes | yes | yes | Latest failed-July observation later replayed; physical outcome not encoded in JSON. |
| `20260904_003219` | complete controller | 585 | 111.7 | yes | yes | yes | **Run A; operator believed physical success.** Slot-stale contaminated. |
| `20260904_004258` | complete controller | 566 | 112.9 | yes | yes | yes | **Run B; operator believed failure.** Slot-stale and marker-stale contaminated. |
| `20260904_005654` | complete controller | 606 | 96.8 | yes | yes | yes | **Run C; operator believed failure.** Slot-stale contaminated. |
| `20260904_010344` | partial INSERT | 405 | 45.3 | no | no | no | Log ends without a terminal event. |
| `20260904_011002` | partial INSERT | 217 | 41.2 | no | no | no | Log ends without a terminal event. |
| `20260904_011838` | partial after release/PUSH | 840 | 126.4 | yes | entered | no | 76 PUSH policy records; no `push_complete`/terminal event. |
| `20260904_012635` | partial INSERT | 265 | -16.4 | no | no | no | Moved-shelf stale-slot failure used for the definitive frozen-slot replay. |

Summary: 16 logs contain multi-step INSERT, six reached software `episode_complete`, and 19 are partial or debug-only. The only operator-labelled physical success in the audited A/B/C set is Run A, but it is still contaminated by the stale-slot bug. Therefore the archive does not support a clean real-world success rate.

### Full real bags

Eight bag directories exist; five were closed with metadata and three contain a DB3 but no metadata (incomplete/unclosed). Only four closed bags overlap a usable policy log closely enough for the automated accepted-slot audit.

| Bag directory | Size | Duration / messages | Audit pairing |
|---|---:|---:|---|
| `full_real_20260902_001428` | 7.1 GB | no metadata | none |
| `full_real_20260902_001642` | 5.6 GB | 36.287 s / 28,344 | none |
| `full_real_20260902_001937` | 35 GB | 303.677 s / 222,851 | `simple_policy_20260902_001927` |
| `full_real_20260903_234640` | 3.8 GB | no metadata | none |
| `full_real_20260904_003338` | 22 GB | 177.788 s / 135,227 | Run A |
| `full_real_20260904_004324` | 74 GB | 540.668 s / 364,989 | Run B |
| `full_real_20260904_005305` | 25 GB | no metadata | none |
| `full_real_20260904_005719` | 34 GB | 263.450 s / 192,863 | Run C |

## 6. Critical deployment bugs

### Bug 1: policy used approved-config slot instead of the `S`-frozen slot

**Evidence.** Preinsert `P/E` used the `S`-accepted pose, but `SimplePolicyControlNode` initialized `T_base_slot` only from the old approved configuration:

```text
translation [0.8554391825, 0.0841262575, 0.1709225333] m
quaternion  [0.0010688608, -0.0206673908, 0.0391970226, 0.9990171720]
```

Run A/B/C all logged this same policy slot despite accepting different physical slots. Their accepted-to-policy errors were:

| Run | Translation error | Rotation error |
|---|---:|---:|
| A | 18.360 mm | 3.259 deg |
| B | 18.398 mm | 3.266 deg |
| C | 33.216 mm | 3.528 deg |

The later moved-shelf run `20260904_012635` accepted `[0.8677716788, 0.0694443763, 0.1786139044] m`, quaternion `[-0.0081632248, -0.0332436170, +0.1204757711, +0.9921259055]`, width `0.0433247350 m`; its error from the stale policy slot was **20.659 mm / 9.519 deg**. The policy takeover error changed from approximately lateral `+33.83 mm`, yaw `+9.34 deg` to corrected lateral `+0.76 mm`, yaw `+0.13 deg`. With the correct quaternion, corrected initial book pose was approximately `[-108.1, -0.7, +5.7] mm` with about `0.06 deg` orientation error. A previous replay using incorrect signs for quaternion `z,w` created a fake `~7.85 deg` error and is invalid evidence.

**Root cause.** `S` froze the slot only inside `simple_preinsert` and in `/tmp/bookshelf_simple_frozen_slot.yaml`. The policy launch had no frozen-slot argument and the policy never loaded that file, so its immutable geometry remained `approved_config`.

**Current fix (uncommitted).** The top-level launch passes `frozen_slot_output` into the policy. At `I`, `_activate_frozen_policy_slot()` loads and validates the saved slot, replaces policy geometry, updates retreat direction, and logs `POLICY SLOT SNAPSHOT`. Missing or invalid configured frozen data rejects policy start; the running cycle does not fall back to live/stale detection. The same `geometry.transform_base_slot` feeds raw observation, `T_slot_book`, nominal control, release bookkeeping, retreat and PUSH.

Files: `simple_policy_control_node.py`, `simple_policy_control.yaml`, `real_experiment_operator.launch.py`, `simple_policy_one_step.launch.py`, and `test_repeatable_trials.py`. The regression constructs a different live/approved slot and verifies policy-side `T_base_slot` equals the accepted YAML after activation.

### Bug 2: cached marker TF counted as 30 independent per-grasp samples

**Run B evidence.** Capture ran from epoch `1788447025.262` to `1788447028.246`. The newest marker update was already **1.876545 s** old at capture start; there were **zero** new `target_book_center` updates during capture. The old loop read the same cached TF 30 times, reported `accepted_count=30`, translation RMS effectively `0` (`1.39e-16 m`) and orientation RMS `0`, and froze it as if it were stable evidence.

**Root cause.** `_lookup()` returned only the transform matrix and discarded `TransformStamped.header.stamp`. The capture loop counted reads rather than unique sensor observations and imposed no maximum source age.

**Current freshness rule (uncommitted).** `_lookup_message()` preserves the marker timestamp. `FreshMarkerSampleGate` accepts only a nonzero, uniquely timestamped marker sample whose age is in `[0, 0.25] s`. It separately counts lookup, stale and duplicate rejections. Capture still makes 30 target reads and retains the existing robust/SVD averaging, `5 mm / 5 deg` outlier limits, and **20 fresh unique retained samples** minimum. End diagnostics report reads, fresh unique count, duplicate/stale/lookup rejections, accepted age range, and PASS/FAIL.

After a successful operator `C`, insufficient freshness is a hard calibration failure and `P` is rejected with `press C to retry`; it does not silently select `fixed_fallback`. Run B would now produce 30 stale rejections, `0/20` fresh unique samples, and FAIL.

Files: `per_grasp_calibration.py`, `preinsert_node.py`, `simple_preinsert.yaml`, and `test_per_grasp_calibration.py`. Tests cover one stale cached transform, duplicate timestamps, sufficient unique fresh data, mixed fresh/stale data, and insufficient-data failure.

Validation recorded immediately after this fix: eight focused calibration tests passed; 26 related workflow tests passed; `scripts/ros2/build_xarm_experiment.sh` built all five selected ROS packages successfully; `git diff --check` passed. No hardware was started.

## 7. Real A/B/C comparison

### Accepted slots

```text
Run A
  t = [0.8719010938, 0.0848576800, 0.1790185251] m
  q = [-0.0055661886, -0.0366523706, 0.0166313168, 0.9991741693]

Run B
  t = [0.8719583277, 0.0846242885, 0.1790071684] m
  q = [-0.0058705481, -0.0365476866, 0.0165714849, 0.9991772561]

Run C
  t = [0.8877452259, 0.0841267880, 0.1786461368] m
  q = [-0.0091875265, -0.0336981976, 0.0650767628, 0.9972687881]
```

Historical accepted widths were not retained in the compact audit output and must not be guessed from the current `/tmp` file.

### Per-grasp transforms used

```text
Run A (fresh marker)
  t = [0.0058744904, 0.0016427737, 0.1825885286] m
  q = [0.7215095291, -0.0096730379, 0.6920251757, 0.0207746949]
  capture: 64 unique marker updates; newest age 0.0313 s
  fit RMS: 0.2475 mm, 0.2113 deg

Run B (STALE marker)
  t = [0.0080628393, -0.0113714122, 0.1819561449] m
  q = [0.7262115909, -0.0795812939, 0.6825563279, -0.0200100523]
  capture: 0 unique marker updates; newest pre-capture age 1.8765 s
  apparent fit RMS: ~0 mm, 0 deg because the cached transform was duplicated

Run C (fresh marker)
  t = [0.0065900202, -0.0045888955, 0.1823683860] m
  q = [0.7215781691, -0.0345243538, 0.6913595260, -0.0124507277]
  capture: 84 unique marker updates; newest age 0.00415 s
  fit RMS: 0.2476 mm, 0.2071 deg
```

Pairwise transform differences:

| Pair | Translation | Rotation |
|---|---:|---:|
| A–B | 13.212 mm | 9.356 deg |
| A–C | 6.276 mm | 4.756 deg |
| B–C | 6.953 mm | 5.358 deg |

### Recorded release and corrected counterfactual

| Run | Operator outcome | Recorded rear at release | INSERT Δrear | Lateral / Z / yaw at release | Recorded release output | Slot error | Correct-slot replay |
|---|---|---:|---:|---:|---:|---:|---|
| A | believed success | -61.802 mm | 111.652 mm | -0.849 / +4.226 mm / -1.391 deg | 0.591276 | 18.360 mm, 3.259 deg | no crossing; maximum raw release `-6.221854` |
| B | believed failure | -60.532 mm | 112.921 mm | -1.083 / +4.557 mm / -1.494 deg | 0.559683 | 18.398 mm, 3.266 deg | no crossing; maximum raw release `-6.201921`; marker calibration remains untrustworthy |
| C | believed failure | -60.548 mm | 96.753 mm | -0.915 / +4.236 mm / -1.325 deg | 0.621900 | 33.216 mm, 3.528 deg | no crossing; maximum raw release `-6.419555` |

All three controllers completed retreat, empty close and PUSH in software. Only A was reported physically successful. Counterfactual replay holds the recorded trajectory fixed and therefore answers whether the actor would release on those corrected observations; it does not predict the corrected closed-loop real trajectory.

Run B is contaminated by both bugs and cannot support a conclusion about the July actor or the true grasp. A and C are useful corrected geometry states but were still executed with the stale slot at the time.

## 8. Evidence audit and policy evidence

The machine-readable audit at `/tmp/bookshelf_real_evidence_audit/audit_summary.json` found:

- 25 policy logs.
- 16 multi-step/usable INSERT runs.
- 6 software-complete controller episodes.
- 19 partial/debug runs.
- 4 bag-paired runs with recoverable historical accepted slots: `20260902_001927` and A/B/C.
- 4/4 had material stale-slot evidence.
- 1 had demonstrated stale-marker evidence (Run B).
- **0 fully clean historical real runs.**
- Correct-slot counterfactual release crossings: **0/4**.
- Two trustworthy corrected takeover states for later physics testing: A and C.

Corrected A/C takeover observations are stored in `/tmp/bookshelf_real_evidence_audit/corrected_takeover_states.json`. They were very well aligned:

| State | rear_to_mouth | lateral | Z | yaw |
|---|---:|---:|---:|---:|
| A corrected | -185.847 mm | +0.771 mm | +6.784 mm | -0.0864 deg |
| C corrected | -186.480 mm | +0.734 mm | +5.293 mm | -0.0382 deg |

The initial audit recommendation was `INSUFFICIENT_EVIDENCE` because only two trustworthy corrected real-derived states and no clean real run existed. Later local Alienware physics and sensitivity tests add evidence but do not replace a clean real trial:

- Closed-loop corrected A/C Isaac tests: both entered and released; A reached final success, C failed later in PUSH/final completion.
- Earlier exact-state plus ten small perturbations: 11/11 entered and released; 6/10 perturbed cases reached final success; the four failures were PUSH/final timeouts.
- Front-to-back ±22 mm test: exactly zero actor-output effect at these states because that scaled observation was already clipped at `+1`.
- One-at-a-time realistic perturbations: no release crossing for ±2 mm lateral/Z, ±1–2 deg yaw/tilt, or gripper `0.39`; rear-to-mouth was the dominant release cue.
- Joint offline sensitivity: 4,000 normal plus 4,000 stress samples, **0/8,000** release crossings. Normal raw release range `[-9.816, -4.640]`; stress `[-10.903, -3.228]`.
- Joint closed-loop Isaac: normal 47/50 entered and correctly released, zero premature releases, three never-releases, 36/50 final successes; stress 30/30 entered and correctly released, zero premature/never-release, 24/30 final successes. Failures after correct release were primarily PUSH/final.

The cumulative diagnostic recommendation is therefore **KEEP_JULY, medium confidence**, conditional on validating both uncommitted deployment fixes in a clean hardware run. The three normal Isaac non-entry/never-release tails and the absence of a clean real episode prevent high confidence.

## 9. Ruled out versus still uncertain

### Ruled out or strongly unsupported

- **Shelf movement does not prove July is bad.** The policy was observing the old approved slot. Correct-slot replay removed all four historical release crossings.
- **No gross slot-X sign or insertion-direction inversion.** Nominal insertion is slot `+X`, and corrected geometry has the expected preinsert sign.
- **No simulation/deployment release-logic mismatch.** Both use the sixth action with threshold `>0.5` in INSERT; there was no hidden simulated depth guard responsible for the real discrepancy.
- **No P/E-to-I held-book transform mismatch after using the correct slot quaternion.** Preinsert and policy consume the same latched per-grasp `T_eef_book`. The reported `~14 mm Z / ~7 deg` handoff discrepancy came from an incorrectly sign-flipped slot quaternion in an offline replay, not runtime code.
- **The 178 mm Isaac versus 200 mm real shelf-depth difference does not affect actor output in the tested states.** `front_to_back` is clipped at `+1`; ±22 mm and the wider ±30 mm sweep produced zero change.
- **Small one-variable realistic errors did not produce premature release** in corrected A/C mid/near-release states.
- **Realistic simultaneous perturbations did not produce premature release** in 8,000 offline samples or 80 joint-perturbation Isaac episodes.
- **Raw action saturation alone is not evidence of unsafe closed-loop behavior.** Physics tests, not saturation counts, were used for the policy recommendation.
- **The old `reached_goal` flag and unavailable trajectory gripper interface are not current release blockers.** The real action/type and status-based success criterion are fixed.
- **The earlier retreat timeout is not current.** It is now 15 s with the same 90 mm completion requirement.

### Still unknown or unresolved

- There is no clean post-fix real closed-loop July trial using both the exact frozen slot and timestamp-enforced per-grasp calibration.
- Physical PUSH reliability is not established. It uses release geometry, not measured contact; actual book travel can differ by about the documented ±5 mm X uncertainty, and incomplete INSERT cannot necessarily be repaired by a 30 mm PUSH.
- `episode_complete` does not establish the book's final physical pose, so no defensible real success rate exists yet.
- The three normal joint-perturbation Isaac non-entry/never-release cases are a real tail concern, although no stress case reproduced them and none released prematurely.
- Servo can still decelerate near singularities. Runtime preinsert selection reduces this risk but does not abolish all hardware-specific kinematic behavior.
- Historical frozen-slot widths for A/B/C are not present in the compact audit output.
- Three large DB3 bags lack metadata; their completeness and pairing were not assumed.
- A fully independent physical ground-truth final-book pose is not logged for every trial.

## 10. Current deployment status and next-trial checks

Current source contains all of the following:

- real xArm `GripperCommand` interface and status-based result handling;
- singularity-aware preinsert IK branch selection;
- table collision scene in the combined launch;
- one hardware stack and one RViz;
- state-aware, asynchronous PLAN/REVIEW/EXECUTE operator workflow;
- immutable release pose and separate mutable PUSH estimate;
- 15 s retreat timeout;
- per-grasp EEF-to-book transform used by P/E and I;
- semantic held-gripper observation;
- exact frozen-slot snapshot for policy/PUSH;
- timestamp freshness and uniqueness enforcement for per-grasp marker samples.

Before the next physical validation, rebuild/source the canonical `.ros2_ws` overlay and confirm all three lines/conditions:

```text
POLICY SLOT SNAPSHOT source=frozen_accepted:/tmp/bookshelf_simple_frozen_slot.yaml ...
PER-GRASP EEF->BOOK CALIBRATION ... result=PASS (fresh_unique >= 20, age <= 0.25 s)
PER-GRASP EEF->BOOK FROZEN
```

Also compare the logged snapshot numerically with `/tmp/bookshelf_simple_frozen_slot.yaml`, ensure the marker is visibly updating during the post-`C` capture, review every trajectory in RViz, and treat any `fixed_fallback` trial as diagnostic rather than the main validation.

## 11. Paper-ready real-world information

### A. Defensible now

- Exact xArm7/RealSense/MoveIt/Servo architecture, rates, frames, book dimensions, configured shelf depth, table geometry, observation/action definitions and controller sequence.
- The July actor artifact/hash and exact embedded-normalization deployment contract.
- Rosbag-backed proof that the historical policy used the wrong slot: four recovered accepted slots, four stale policy slots, quantified A/B/C errors.
- Rosbag-backed proof of Run B's cached-marker problem: 1.8765 s old at start, zero fresh updates, 30 duplicate reads falsely accepted.
- Exact production fixes and hardware-free tests/build status.
- Correct-slot counterfactual result: 0/4 historical crossings on fixed recorded trajectories.
- Corrected real-derived sensitivity and Isaac outcomes, clearly labelled offline/simulation evidence.
- The conservative policy conclusion: keep July pending clean hardware validation; downstream PUSH is a separate issue.

### B. Development/debugging details probably not for the main paper

- Obsolete gripper action endpoint and `reached_goal` result bug.
- Operator console property-name and stale-status bugs.
- Candidate-5-specific offline utilities and intermediate singularity debugging.
- The invalid replay with sign-flipped slot quaternion.
- Individual aborted one-step/debug runs unless used in an appendix or engineering audit.
- HPC checkpoint-loader/version compatibility and training orchestration details.

### C. Claims that must not be made yet

- Do not claim a post-fix real insertion success rate or repeatable real robustness.
- Do not claim that July plus both fixes has completed a clean physical episode; no such logged historical episode exists.
- Do not claim `episode_complete` means physical insertion success.
- Do not claim PUSH measures contact or guarantees 30 mm physical book travel.
- Do not claim all sim-to-real error is solved, or that retraining is conclusively unnecessary.
- Do not claim Run A is clean; it was operator-reported successful but used the stale policy slot.

### D. Final hardware evidence still needed

1. One deliberately cautious validation using July, exact frozen-slot snapshot, fresh per-grasp PASS, full bag, and explicit physical stage/outcome notes.
2. Several repeated clean trials with shelf/grasp variation, reporting INSERT, release, PUSH and final placement separately.
3. Independent release/final book pose measurement where feasible, plus observed physical PUSH travel/contact outcome; do not substitute the controller's geometric contact estimate.

Evidence that would reverse the current `KEEP_JULY` recommendation: repeated clean post-fix real INSERT divergence, premature release, or never-release under small plausible grasp/slot variations. Repeated correct INSERT/release with PUSH failures would instead motivate PUSH/deployment work, not PPO retraining.

## 12. Exact file and path index

### Runtime source and configuration

- Main policy controller: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/simple_policy_control_node.py`
- Preinsert/operator planning node: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/preinsert_node.py`
- Per-grasp calibration math/gate: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/per_grasp_calibration.py`
- Observation math: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/policy_observation_math.py`
- Nominal/residual math and NPZ actor: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/residual_policy_math.py`
- PUSH geometry: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/post_insert_math.py`
- Operator console: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/operator_console_node.py`
- Operator action bridge: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/operator_action_node.py`
- Policy config: `ros2/bookshelf_simple_experiment_ros/config/simple_policy_control.yaml`
- Preinsert/slot config: `ros2/bookshelf_simple_experiment_ros/config/simple_preinsert.yaml`
- Top-level launch: `ros2/bookshelf_simple_experiment_ros/launch/real_experiment_operator.launch.py`
- Preinsert launch: `ros2/bookshelf_simple_experiment_ros/launch/real_preinsert_workflow.launch.py`
- Policy launch: `ros2/bookshelf_simple_experiment_ros/launch/simple_policy_one_step.launch.py`
- Physical bringup: `ros2/bookshelf_policy_ros/launch/physical_hardware_bringup.launch.py`
- xArm/MoveIt/Servo bringup: `ros2/bookshelf_policy_ros/launch/robot_setup.launch.py`, `ros2/bookshelf_policy_ros/launch/xarm7_moveit_servo_server.launch.py`
- RealSense and hand-eye: `ros2/bookshelf_policy_ros/launch/camera_setup.launch.py`, `ros2/bookshelf_policy_ros/launch/publish_handeye_camera_link.launch.py`
- Servo source config: `/home/riot/Chris/ros2_ws/src/xarm_ros2/xarm_moveit_servo/config/xarm_moveit_servo_config.yaml`
- Build script: `scripts/ros2/build_xarm_experiment.sh`
- Existing operator checklist: `REAL_XARM7_SESSION.md`

### Actors and calibration

- July actor: `/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz`
- F1-40 backup: `/home/riot/BookshelfFiles/trained_models/bookshelf_fresh_F1_40M_backup_actor.npz`
- Approved geometry/config: `/home/riot/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml`
- Provenance: `/home/riot/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.provenance.json`
- Marker/book mount: `ros2/bookshelf_shadow_ros/config/real_book_aruco0_mount.yaml`
- Standalone marker mount copy: `ros2/bookshelf_simple_experiment_ros/config/reference_marker0_book_mount.yaml`
- Scan pose: `/home/riot/BookshelfFiles/experiment_configs/operator_joint_poses/scan_joint_state.yaml`
- Loading/return pose: `/home/riot/BookshelfFiles/experiment_configs/operator_joint_poses/loading_joint_state.yaml`
- Current-cycle frozen slot: `/tmp/bookshelf_simple_frozen_slot.yaml` (ephemeral; never use it as an assumed historical slot).

### Primary policy logs and bags

- All policy logs: `/home/riot/BookshelfFiles/experiment_logs/simple_policy_*/policy_step.jsonl`
- All full bags: `/home/riot/BookshelfFiles/experiment_logs/full_real_bags/`
- July early-release bag/log: `full_real_20260902_001937/`, `simple_policy_20260902_001927/policy_step.jsonl`
- Run A: `full_real_20260904_003338/`, `simple_policy_20260904_003219/policy_step.jsonl`
- Run B: `full_real_20260904_004324/`, `simple_policy_20260904_004258/policy_step.jsonl`
- Run C: `full_real_20260904_005719/`, `simple_policy_20260904_005654/policy_step.jsonl`
- Definitive moved-shelf slot replay log: `simple_policy_20260904_012635/policy_step.jsonl`

### Diagnostic evidence

- Real evidence audit: `/tmp/bookshelf_real_evidence_audit/audit_summary.json`
- Corrected takeover states: `/tmp/bookshelf_real_evidence_audit/corrected_takeover_states.json`
- Corrected A/C closed-loop results: `/tmp/bookshelf_real_evidence_audit/closed_loop_results.json`
- Front/back sensitivity: `/tmp/bookshelf_real_evidence_audit/front_to_back_sensitivity.csv`
- Release sensitivity: `/tmp/bookshelf_real_evidence_audit/release_sensitivity.csv`, `/tmp/bookshelf_real_evidence_audit/release_sensitivity_summary.json`
- Final retrain-decision directory: `/home/riot/BookshelfFiles/evaluation/july_final_retrain_decision/`
- Final summary: `/home/riot/BookshelfFiles/evaluation/july_final_retrain_decision/final_decision_summary.json`
- Joint offline samples/summary: `joint_offline_samples.csv`, `joint_offline_summary.json` in that directory.
- Isaac cases/results: `isaac_joint_cases.json`, `isaac_joint_results.json` in that directory.
- Audit helper: `scripts/sb3/real_policy_evidence_audit.py`
- Counterfactual helper: `scripts/sb3/real_release_counterfactual.py`
- Corrected takeover physics helper: `scripts/sb3/july_corrected_takeover_closed_loop.py`
- Joint robustness helper: `scripts/sb3/july_joint_robustness.py`
- Robustness summarizer: `scripts/sb3/summarize_july_final_robustness.py`
- Offline IK branches: `scripts/xarm7_preinsert_ik_branches.py`
- Candidate transition plan: `scripts/xarm7_candidate5_transition_plan.py`
- Fake Servo diagnostic: `scripts/xarm7_candidate5_servo_diagnostic.py`

## Bottom line

The strongest current explanation for the historical premature/poor real runs is deployment geometry contamination, not demonstrated inherent failure of the July policy: policy INSERT used the wrong slot in every bag-paired corrected run, and Run B also accepted a stale cached marker transform. Corrected replay produced zero release crossings in four runs, and realistic offline/Isaac perturbations produced no premature releases. The appropriate current decision is **KEEP_JULY with medium confidence**, deploy only after rebuilding the uncommitted frozen-slot and marker-freshness fixes, and obtain a clean fully bagged real trial before making paper claims about real-world success or deciding that retraining is necessary.
