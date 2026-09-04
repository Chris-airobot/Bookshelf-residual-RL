# Bookshelf Residual-RL: Master Project Snapshot

**Snapshot date:** 2026-09-04 (Australia/Melbourne)  
**Purpose:** definitive paper and experiment source of truth before clean post-fix hardware trials  
**Inputs reconciled:** `PROJECT_STATUS_RIOT_20260904.md`, `PROJECT_STATUS_ALIENWARE_20260904.md`, current Riot source/data, durable Riot robustness results, and read-only completed Tooarrana evaluation results.

Evidence labels used throughout:

- **REAL MEASURED:** physical robot, policy log, rosbag, or operator observation.
- **OFFLINE COUNTERFACTUAL:** deterministic inference on recorded states; no action is fed back.
- **ISAAC CLOSED-LOOP:** simulated physics with policy actions fed back.
- **INFERENCE/INTERPRETATION:** conclusion supported by the preceding evidence, not a direct measurement.

`episode_complete` means the real controller completed its programmed sequence. It does **not** prove physical insertion success.

# 1. Research Goal

The task is to insert a grasped, rigid book into a narrow, cluttered bookshelf slot, release it, and finish seating it. Tight lateral clearance, contact, pose-estimation error, redundant robot kinematics, release timing, and the gap between simulation and hardware make this harder than free-space reaching.

The system combines a transparent Cartesian nominal controller with a learned residual PPO policy. The nominal term supplies forward insertion and alignment; PPO supplies bounded corrections and a learned release decision. The intended scientific contribution is evidence that structured nominal control plus learned residuals improves contact-rich insertion robustness over nominal-only and PPO-only alternatives, while retaining an interpretable controller and a practical sim-to-real deployment path. Real-world claims remain conditional on clean post-fix trials.

# 2. Complete System Overview

1. Isaac Lab simulates a Franka Panda, book, shelf row, contact, release, retreat, and PUSH at 60 Hz.
2. A six-action Stable-Baselines3 PPO policy observes a geometric 12-D state and learns bounded residual motion plus release.
3. Frozen scenario banks compare nominal-only, PPO-only, and nominal-plus-residual policies on paired cases.
4. The trained July policy and VecNormalize state are exported to a deterministic NPZ actor.
5. On the real xArm7, RealSense/ArUco perception estimates a slot and the held book. The operator freezes the slot and a post-grasp calibration freezes the trial-specific EEF-to-book transform.
6. MoveIt plans a collision-checked preinsert trajectory using a singularity-aware xArm7 IK branch selector.
7. The real controller reconstructs the same 12-D geometric observation, applies the embedded normalization, combines nominal and residual commands, and streams Cartesian twists through MoveIt Servo.
8. A learned release opens the gripper; scripted retreat, empty close, geometric PUSH, and reviewed return follow.

The simulation robot and real robot differ, but the policy acts in task-space geometry rather than joint space. Ground truth used by Isaac reward, success, and evaluation never enters the real controller.

# 3. Isaac Simulation Environment

## Software and task

- Isaac Sim: `5.1.0-rc.19+release.26219.9c81211b.gl` at `/home/chris/isaacsim`.
- Isaac Lab commit: `3e73d6dd79080fd7632488c061052a6edd52e230`; core extension `0.54.3`.
- Task: `Bookshelf-Residual-Direct-v0`.
- Robot: Franka Panda with joint-position targets.
- Physics: `dt=1/120 s`, decimation 2, hence 60 Hz policy/control.
- Episode: 10 s, about 600 policy steps.
- Default replication: 4,096 environments at 2 m spacing; July training explicitly used 256.

## Geometry and dynamics

- Book semantic dimensions: depth 156 mm, lateral thickness 34 mm, height 236 mm (`_BOOK_LWH=(0.156,0.236,0.034)` in historical code ordering).
- Book mass 0.45 kg; static/dynamic friction 1.8/1.5; linear/angular damping 0.2/4.0.
- Slot mouth x 0.63 m; back x 0.83 m; depth 200 mm; center y 0; shelf top z 0.05 m; deck thickness 20 mm.
- July residual task uses 3 mm added lateral clearance: 37 mm opening for a 34 mm book, 1.5 mm per side when centered.
- The row has ten logical book positions and one sampled missing position. Side books are kinematic. Adjacent single books merge into double-width books with probability 0.35. Single-book heights are `[229,205,185,218,195,229,202,175,214]` mm and double-book heights `[215,190,229,200]` mm.

The old v5 docstring mentioning a 23–26 mm physical gap conflicts with executable geometry and is not authoritative.

## Reset randomization and July curriculum

| Stage | Progress | arm joints | grasp X | grasp Y | grasp Z | grasp yaw |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0–20% | ±1.5° | ±3 mm | ±3 mm | ±1.5 mm | ±3° |
| 2 | 20–50% | ±2° | ±5 mm | ±4 mm | ±2 mm | ±5° |
| 3 | 50–100% | ±3° | ±8 mm | ±6 mm | ±3 mm | ±8° |

The curriculum counter uses vector/common steps and totals 260,416. July ran 4,096 iterations × 32 rollout steps = 131,072 vector steps, reaching the 50% boundary at the end, although SB3 counted 33,554,432 aggregate transitions. Clearance and residual-scale curricula were off.

## Stages and termination

Simulation modes are INSERT `0`, SCRIPTED `0.5`, and PUSH `1`. A release opens for 3 steps, retreats for 6 steps using a `-15 mm` target increment per step, closes for 5 steps, then enters PUSH. These are targets, not guaranteed achieved distances.

Success is checked only in PUSH, after at least five PUSH steps and for four consecutive steps: rear-to-mouth ≥ -12 mm; front-to-back ≤ 55.2 mm; lateral extent within the opening plus 1.5 mm; |z error| < 15 mm; |yaw| < 8°; upright dot ≥ 0.85. Velocity thresholds are disabled. Episodes otherwise end on drop or timeout. Failure categories include `not_push`, `depth`, `lateral`, `z`, `yaw`, `upright`, `unstable`, `drop`, and `timeout`.

# 4. Observation and Action Space

## Exact 12-D raw observation

| i | semantic | pre-VecNormalize scale |
|---:|---|---:|
| 0 | mode: INSERT 0, SCRIPTED 0.5, PUSH 1 | direct |
| 1 | rear edge to slot mouth, `book_rear_x-mouth_x` | ÷0.08 m |
| 2 | front to slot back, `back_x-book_front_x` | ÷0.08 m |
| 3 | lateral error, `slot_center_y-book_center_y` | ÷0.05 m |
| 4 | vertical error from supported target center | ÷0.05 m |
| 5 | wrapped yaw error | ÷30° |
| 6–8 | policy-tool minus book-center XYZ | ÷0.25 m |
| 9 | gripper-open semantic | direct |
| 10–11 | world X/Y components of rotated local upright/spine axis `(0,1,0)` | direct |

Scaled components are clipped to `[-1,1]`, then the saved VecNormalize mean/variance is applied and normalized observations are clipped to `[-10,10]`. The corrected real takeover has rear `-0.186 m` and front-to-back about `+0.230 m`; indices 1 and 2 are already clipped to `-1/+1`. Consequently the tested 178-versus-200 mm shelf-depth difference has zero policy effect in that region.

Simulation computes gripper opening from mean finger position, mapped from 0.015 m closed to 0.060 m open. The recovered July held-book value used on hardware is `0.009838026859259968`; fully open is 1.0. The physical aperture remains separately logged.

## Exact 6-D policy action

| i | semantic | residual scale per step |
|---:|---|---:|
| 0 | X translation | 2.0 mm |
| 1 | Y translation | 1.0 mm |
| 2 | Z translation | 1.5 mm |
| 3 | yaw | 0.35° |
| 4 | pitch | 0.30° |
| 5 | release request | threshold `>0.5` in INSERT |

Actions are clipped to `[-1,1]`. Nominal and residual commands are added, then bounded to X ±8 mm, Y ±3 mm, Z ±7 mm, yaw ±0.8°, pitch ±0.6° per policy step. The obsolete July `experiment_spec.json` describing 16 observations is inconsistent with executable code and copied `env.yaml`; it must not define the contract.

# 5. Nominal Controller

In INSERT, nominal X is enabled only when |lateral|<6 mm, |z|<10 mm, |yaw|<6°, and tilt-X<0.1 rad. It advances +1.0 mm/step, reducing to +0.7 mm after rear-to-mouth exceeds -35 mm. Alignment terms are:

- Y: `0.25*lateral_error`, limited to 1.5 mm/step.
- Z: `-0.18*(z_error-6 mm)`, limited to 1.8 mm/step.
- yaw: `-0.14*yaw_error`, limited to 0.35°/step.
- pitch: `-0.02*tilt_x`, limited to 0.25°/step.

Release is the sixth PPO output `>0.5`; there is no consecutive-step or deployment geometry guard. The optional observable release mask exists but default guard mode is `none`.

Simulation PUSH nominal X is +0.8 mm/step, with Y/Z/yaw/pitch gains 0.35/0.30/0.20/0.08 and Y/Z limits 0.5/1.0 mm. Real PUSH preserves nominal-plus-residual PPO behavior but plans 30 mm book travel from the release-time geometric face estimate. The ±5 mm X uncertainty is documentation/physical uncertainty only and does not add controller travel.

# 6. Residual RL / PPO

## July training

- Algorithm/policy: SB3 PPO `MlpPolicy`.
- Actor and critic: two 256-unit ReLU layers each.
- Seed 42; CUDA.
- `n_steps=32`, batch 8,192, 10 epochs/update.
- learning rate `1e-4`; gamma 0.99; GAE 0.95; clip 0.2.
- entropy coefficient 0.003; value coefficient 1.0; max gradient norm 1.0.
- observation normalization on (`clip_obs=10`); reward normalization off.
- command: `train.py --task Bookshelf-Residual-Direct-v0 --num_envs 256 --max_iterations 4096 --headless`.
- total: 33,554,432 aggregate transitions.

Final rolling 1,000 training episodes: success 0.915, timeout 0, drop 0, mean return 87.870546, mean length 150.738, lateral 2.7639 mm, z 4.3131 mm, yaw 0.6533°. Cumulative rows are 194,157 success, 33,756 `not_push`, 3,153 `depth`, and 11 timeout; cumulative and rolling figures are not interchangeable.

## Immutable artifacts

| artifact | path | SHA-256 |
|---|---|---|
| July SB3 model | `/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/model.zip` | `80f7aa2d6675a99f3965b2479bc0b62f5f3320e724a6f3399efacc1640b3b4ed` |
| July VecNormalize | same directory, `model_vecnormalize.pkl` | `88670e59194fa5d70743872ea02232acdbea93f827fbce116d1fcb4a0745a635` |
| deployed July NPZ | `/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz` | `75773dde0edabebcb525469c2e2b1cf868d7724f45a9f661f994cd8847a0ab19` |
| F1-40 backup NPZ | `/home/riot/BookshelfFiles/trained_models/bookshelf_fresh_F1_40M_backup_actor.npz` | `57741afb8434f3fa609d7f0cf15241fb0b1cdc302ddc94a7c79e78dee0d4ec05` |

The NPZ contains the deterministic actor and normalization needed by the real controller. July checkpoint and exported actor were verified tensor/output equivalent. F1-40 is compatible (12-D in, 6-D out) and is backup only.

# 7. Training History and Model Selection

The reliable lineage is manual demonstrations/BC, PPO, then nominal-plus-residual PPO. Older v4 checkpoints use a 10-D/5-D contract and are not current-policy candidates. The July residual model remains primary because it has the strongest frozen nominal evaluation and corrected deployment evidence.

## Targeted fresh sweep F0–F4

All fresh runs used seed 42, 256 environments, about 40,001,536 transitions, and checkpoints at 10M/20M/30M/40M/final. Definitions:

- **F0:** moderate observation/gripper DR, July-scale grasp DR, original release objective.
- **F1:** F0-like DR plus smooth release-readiness objective.
- **F2:** F1 plus corruption curriculum from 25% to 100% by 30M.
- **F3:** observation-bias plus gripper DR only, no grasp DR, smooth objective.
- **F4:** observation/gripper DR plus moderate grasp DR (X±5, Y±8, Z±3 mm, yaw±8°), smooth objective.

Moderate observation bias was episode-fixed and affected only the policy-observed book pose: X±5, Y±10, Z±4 mm; yaw±8°, roll/pitch±3°. Held gripper nuisance covered `[0,0.45]`. Strong-tail observation ranges were approximately X±7, Y±12, Z±6 mm; yaw±12°, roll/pitch±5°. The smooth objective retained release mechanics but added a continuous geometric readiness signal: ready release bonus 2, premature release penalty 2, withheld-release penalty `0.02*readiness`, and never-release timeout penalty 20.

Completed Tooarrana evaluation used fixed paired banks, 512 episodes/profile. Concrete storage contains **179/182** expected results: F2-20 nominal, physical-like, and gripper-only are absent. They were not rerun. The `premature_release` field means “release while `_nominal_release_mask` was false”; because that mask is stricter/different from final success, it labels many successful releases and must not be read as a physical failure rate.

Compact success table (%); `—` means missing:

| Policy | Nominal | Physical-like | Combined | Strong-tail | Actuation |
|---|---:|---:|---:|---:|---:|
| July | 86.7 | 0.0 | 3.5 | 1.6 | 2.7 |
| F0-10 / 20 / 30 | 0.0 / 10.5 / 33.6 | 0.0 / 7.8 / 2.7 | 0.0 / 19.3 / 14.8 | 0.0 / 11.7 / 10.9 | 0.0 / 16.2 / 11.5 |
| F0-40 / final | 34.8 / 27.9 | 27.9 / 31.6 | 16.8 / 18.0 | 13.9 / 10.7 | 16.0 / 14.3 |
| F1-10 / 20 / 30 | 2.7 / 35.2 / 31.8 | 2.3 / 29.9 / 46.5 | 6.4 / 25.2 / 23.8 | 1.8 / 15.0 / 13.1 | 4.1 / 20.1 / 17.8 |
| **F1-40 / final** | **57.2 / 62.5** | **53.3 / 49.8** | **33.6 / 30.9** | **16.2 / 16.6** | **20.9 / 21.1** |
| F2-10 / 20 / 30 | 0.4 / — / 50.8 | 0.4 / — / 49.0 | 5.3 / 15.4 / 24.8 | 2.3 / 2.7 / 9.6 | 4.9 / 15.8 / 20.3 |
| F2-40 / final | 9.2 / 9.0 | 41.4 / 44.9 | 25.0 / 24.0 | 13.9 / 14.8 | 14.6 / 17.8 |
| F3-10 / 20 / 30 | 0.0 / 3.5 / 12.9 | 0.0 / 10.7 / 18.8 | 0.0 / 9.2 / 12.3 | 0.0 / 6.2 / 8.4 | 0.0 / 10.2 / 6.6 |
| F3-40 / final | 10.5 / 10.4 | 0.6 / 2.1 | 13.1 / 14.5 | 10.0 / 9.8 | 10.7 / 10.4 |
| F4-10 / 20 / 30 | 0.4 / 1.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 1.0 / 0.2 | 0.0 / 0.8 / 0.0 | 0.0 / 0.4 / 0.2 |
| F4-40 / final | 0.2 / 0.0 | 1.2 / 1.6 | 0.0 / 0.6 | 0.0 / 0.0 | 0.2 / 0.0 |

Never-release remained the dominant failure under perturbation. For F1-40 it was 42.6% nominal, 46.7% physical-like, 65.4% combined, 83.6% strong, and 78.7% actuation. July was 5.7% nominal but 100% physical-like and 92.6% combined under these synthetic profiles. Thus fresh training improved the synthetic physical-like profile but did not solve robustness as a task: roughly half or more of difficult cases still never released. F1 improved substantially over F0; F2 curriculum was unstable and collapsed nominally after 30M; F3 was worse, and F4 collapsed. F1-40 is the best conservative backup. F1 final is only 1,536 steps newer than 40M but differs by +5.3 pp nominal, -3.5 pp physical-like, and -2.7 pp combined; it is not redundant in performance. July remains primary because of its 86.7% paired nominal success and the evidence that historical real failures were deployment-contaminated.

# 8. Main Simulation Results

## Frozen paired 3 mm comparison

Two thousand identical scenarios per method/run:

| Method | Seed | Success |
|---|---:|---:|
| nominal only | deterministic | 577/2000 = 28.85% |
| PPO only | 42 / 123 / 2026 | 0 / 0 / 0 = 0% |
| residual PPO | 42 | 1792/2000 = 89.60% |
| residual PPO | 123 | 1781/2000 = 89.05% |
| residual PPO | 2026 | 1883/2000 = 94.15% |

Residual mean 90.933%, sample SD 2.799 percentage points, t-based 95% CI 83.98–97.89% (only three policy seeds). Paired residual-only versus nominal-only wins were 1276, 1264, and 1332; McNemar p-values were `1.6985e-296`, `3.6477e-294`, and numerical underflow to zero. The bank hash is `71282dd1c471ebbcf8c4145b6ee01b47af37b11f86f0686284fbec0111981f1b`.

## Clearance robustness: success %

| clearance | nominal | residual seeds 42/123/2026 | mean ± SD |
|---:|---:|---:|---:|
| 1 mm | 9.70 | 4.80 / 9.45 / 13.70 | 9.32 ± 4.45 |
| 2 mm | 20.00 | 52.00 / 52.65 / 60.25 | 54.97 ± 4.59 |
| 3 mm | 28.85 | 89.60 / 89.05 / 94.15 | 90.93 ± 2.80 |
| 4 mm | 34.20 | 92.15 / 97.35 / 97.90 | 95.80 ± 3.17 |
| 5 mm | 42.30 | 94.85 / 97.40 / 97.65 | 96.63 ± 1.55 |

PPO-only was 0% at every clearance.

## Reset-offset robustness: success %

| offset scale | nominal | residual seeds | mean ± SD |
|---:|---:|---:|---:|
| 0.00× | 28.00 | 90.75 / 94.10 / 96.50 | 93.78 ± 2.89 |
| 0.50× | 26.65 | 90.60 / 89.65 / 94.45 | 91.57 ± 2.54 |
| 1.00× | 18.10 | 87.60 / 86.15 / 91.60 | 88.45 ± 2.82 |
| 1.25× | 15.70 | 85.40 / 83.65 / 87.90 | 85.65 ± 2.14 |
| 1.50× | 15.40 | 81.10 / 79.15 / 83.35 | 81.20 ± 2.10 |

The earlier July “current noise” 95.8% and “old noise” 87.95% table repeated identical counts for three supposed seeds and is not evidence of independent seed variation; prefer the frozen-bank results.

# 9. Real Robot Hardware and Deployment

- Robot: UFactory xArm7, firmware v2.2.0, SDK 1.15.2, IP `192.168.1.209`.
- ROS 2 Humble, official xArm MoveIt stack under `/home/riot/Chris/ros2_ws/install`.
- Arm controller `/xarm7_traj_controller`; gripper `/xarm_gripper/gripper_action` (`control_msgs/action/GripperCommand`), open 0.0, closed 0.85.
- RealSense D435 serial `242322078188`, firmware 5.17.3.10; RGB/depth 640×480 at 30 Hz; RealSense ROS 4.55.1/librealsense 2.56.4.
- Hand-eye `T_link_eef_camera_link`: translation `[0.064694,-0.017286,0.018312]` m, quaternion `[0.711047,-0.001225,0.703143,-0.000110]` XYZW.
- Real book: depth/thickness/height `[0.156,0.034,0.236]` m. Configured shelf depth 0.200 m. Observed accepted slot widths include 0.037844 m (approved configuration) and 0.043325 m (moved-shelf run).
- Frames: `link_base` base; `link_eef`; `link_tcp`; slot +X enters shelf, +Z is up, +Y completes a right-handed frame.
- Real policy 20 Hz; Servo command 30 Hz. Servo singularity deceleration/hard stop conditions are 17/30, joint margin 0.1 rad, collision checking 10 Hz, proximity 0.01/0.02 m.
- Planning table collision object: size `[1.5,0.6,0.05]` m, center `[0.75,0,-0.025]` m.
- Primary actor: July NPZ above.

Launch after canonical build/source:

```bash
ros2 launch bookshelf_simple_experiment_ros real_experiment_operator.launch.py \
  robot_ip:=192.168.1.209 allow_execution:=true \
  shadow_full_sequence:=false show_rviz:=true
```

Full bag:

```bash
mkdir -p ~/BookshelfFiles/experiment_logs/full_real_bags
BAG=~/BookshelfFiles/experiment_logs/full_real_bags/full_real_$(date +%Y%m%d_%H%M%S)
ros2 bag record -a -x '.*(compressed|theora).*' -o "$BAG"
```

Operator sequence: **G → E → S → L → E → O → C → P → E → I → H → E**. After G/L/P/H, inspect RViz before E. After C, require fresh per-grasp PASS and `PER-GRASP EEF->BOOK FROZEN`. Before/at I, require `POLICY SLOT SNAPSHOT source=frozen_accepted:/tmp/bookshelf_simple_frozen_slot.yaml`.

# 10. Real Deployment Architecture

`S` accepts the current RGB-D slot estimate, writes `/tmp/bookshelf_simple_frozen_slot.yaml`, and freezes visualization. `P` computes the same Cartesian preinsert book target (`[-0.108,0,+0.006]` m in slot), searches 24 diverse xArm7 IK seeds, deduplicates within 0.01 rad, checks joint limits/collision/plannability, samples 11 points over the next 0.10 m insertion, rejects predicted Jacobian condition above 27, and ranks primarily by maximum condition then transition cost within a score band. `E` alone executes a reviewed plan.

After `C`, the preinsert node attempts 30 stamped `/bookshelf_simple/target_book_pose` observations at 10 Hz (about 3 s) with corresponding robot EEF poses, computes `T_eef_book=inv(T_base_eef)@T_base_book`, robustly averages translation and rotation, and freezes it. P/E and I use this exact transform. The marker need not remain visible during insertion.

At I, the policy snapshots the exact frozen slot and never returns to a live/approved slot during the cycle. It reconstructs `T_base_book=T_base_eef@T_eef_book_frozen`, transforms book/tool into the slot, maps held gripper to the July semantic value, evaluates the NPZ, combines nominal/residual commands, and streams a bounded twist. The raw physical aperture remains logged.

Release opens via the real GripperCommand action; success uses action `GoalStatus.STATUS_SUCCEEDED`, not the driver's unreliable `reached_goal` bit. Scripted retreat is 90 mm along slot -X with a 15 s timeout, then empty close, 30 mm geometric PUSH along +X, and reviewed return. No contact sensor exists; geometric contact is not measured contact.

# 11. Historical Real Experiments

The audit discovered 25 `policy_step.jsonl` files: 16 multi-step INSERT runs, six controller-complete episodes, and 19 partial/debug runs. Controller-complete logs were `20260901_235523`, `20260902_001927`, `20260903_233842`, and A/B/C below.

Meaningful development runs:

- `20260827_221624`: learned release at step 773, output 0.514665; obsolete gripper trajectory action unavailable.
- `20260827_230956`: gripper physically opened, but `reached_goal=false` incorrectly blocked retreat.
- `20260827_232929`: forward action continued while Servo status 1 decelerated near a singularity.
- `20260901_234542`: retreat started but the old timeout expired.
- `20260902_001927`: **REAL MEASURED/operator:** book incompletely inserted; release at rear -58.967 mm, output 0.501103; controller later completed.
- Run A `20260904_003219`: operator believed physical success; controller complete, but stale-slot contaminated.
- Runs B `004258` and C `005654`: operator believed physical failures; controller complete; B also has stale marker calibration.
- `20260904_012635`: moved-shelf partial run exposed the stale-slot bug; no learned release.

No historical run is fully clean under the current slot and marker-freshness requirements. Operator reports are retained as physical observations, not converted into an automatic success label.

# 12. Critical Deployment Bugs Found

## 12.1 Stale slot bug

The `S` service wrote the accepted slot, and P/E used it, but `SimplePolicyControlNode` loaded `trial_static_slot.yaml` from approved configuration at I. Thus policy observation, `T_slot_book`, nominal geometry, release/depth logic, and PUSH referenced a different slot.

The common stale slot in A/B/C was translation `[0.8554391825,0.0841262575,0.1709225333]` m and quaternion `[0.0010688608,-0.0206673908,0.0391970226,0.9990171720]`. Errors versus accepted slots were A 18.360 mm/3.259°, B 18.398 mm/3.266°, C 33.216 mm/3.528°. The moved-shelf case was 20.659 mm/9.519° wrong and changed the apparent takeover from lateral +33.83 mm/yaw +9.34° to +0.7 mm/about 0° when corrected.

Current source snapshots `frozen_slot_config` at policy start, reports `POLICY SLOT SNAPSHOT`, and fails rather than silently reverting after a frozen slot is expected. All downstream policy/PUSH geometry uses that immutable snapshot. Focused regression tests and the selected-package build passed.

## 12.2 Stale book-marker calibration bug

The old capture loop repeatedly looked up cached `target_book_center` TF. Thirty reads of the same transform produced zero apparent spread and passed robust fitting even if no new marker observation arrived. In Run B, zero fresh updates occurred; the newest update was already 1.876545 s old when capture began.

Current validation uses the source PoseStamped timestamp: age must be in `[0,0.25]` s, timestamp nonzero, and never previously counted. Capture makes at most 30 reads and requires at least 20 unique fresh samples before and after the existing 5 mm/5° robust outlier rejection. Translation averaging and SVD rotation averaging are unchanged. Diagnostics report reads, accepted unique samples, stale/duplicate/lookup rejections, accepted age range, and PASS/FAIL. Insufficient data blocks P and requests another C; validation trials no longer silently use `fixed_fallback`. Historical Run B would reject all 30 as stale and fail 0/20.

Tests cover stale repeated TF, duplicate stamps, sufficient fresh unique samples, mixed data, and insufficient data. Eight calibration-focused and 26 related workflow tests passed; the five-package build and `git diff --check` passed.

# 13. A/B/C Real Trial Analysis

## Recovered accepted slots

| Run | translation XYZ m | quaternion XYZW |
|---|---|---|
| A | `[.871901094,.084857680,.179018525]` | `[-.005566189,-.036652371,.016631317,.999174169]` |
| B | `[.871958328,.084624289,.179007168]` | `[-.005870548,-.036547687,.016571485,.999177256]` |
| C | `[.887745226,.084126788,.178646137]` | `[-.009187527,-.033698198,.065076763,.997268788]` |

Historical widths are unavailable and are not inferred from today's `/tmp` file.

## Per-grasp transforms and outcomes

| Run | `T_eef_book` translation m | quaternion XYZW | freshness | recorded release rear / output | correct-slot counterfactual | physical interpretation |
|---|---|---|---|---|---|---|
| A | `[.00587449,.00164277,.18258853]` | `[.72150953,-.00967304,.69202518,.02077469]` | 64 updates; age .0313 s; fit .2475 mm/.2113° | -61.802 mm / .591276 | no crossing; max raw -6.221854 | operator believed success |
| B | `[.00806284,-.01137141,.18195614]` | `[.72621159,-.07958129,.68255633,-.02001005]` | **0 updates; 1.8765 s stale** | -60.532 mm / .559683 | no crossing; max raw -6.201921 | operator failure; both bugs |
| C | `[.00659002,-.00458890,.18236839]` | `[.72157817,-.03452435,.69135953,-.01245073]` | 84 updates; age .00415 s; fit .2476 mm/.2071° | -60.548 mm / .621900 | no crossing; max raw -6.419555 | operator failure |

Pairwise transform differences: A–B 13.212 mm/9.356°, A–C 6.276 mm/4.756°, B–C 6.953 mm/5.358°. A/B/C recorded release lateral/Z/yaw were respectively `-0.849/+4.226 mm/-1.391°`, `-1.083/+4.557/-1.494°`, and `-0.915/+4.236/-1.325°`. Recorded rear progression was 111.652, 112.921, and 96.753 mm.

**OFFLINE COUNTERFACTUAL:** changing only to each recovered accepted slot eliminates release crossing on all three recorded trajectories. This does not predict corrected closed-loop hardware motion. B remains unusable for grasp conclusions because its frozen transform was stale.

# 14. Historical Evidence Audit

Machine-readable source: `/tmp/bookshelf_real_evidence_audit/audit_summary.json`.

| item | count |
|---|---:|
| policy logs | 25 |
| usable multi-step INSERT | 16 |
| controller-complete episodes | 6 |
| partial/debug runs | 19 |
| bag-paired runs with recovered frozen slot | 4 |
| recovered runs contaminated by stale slot | 4/4 |
| demonstrated stale-marker runs | 1 (Run B) |
| fully clean historical real runs | 0 |
| corrected-slot replay release crossings | 0/4 |

Trustworthy corrected takeovers are A and C:

| state | rear | lateral | Z | yaw |
|---|---:|---:|---:|---:|
| A | -185.847 mm | +0.771 mm | +6.784 mm | -0.0864° |
| C | -186.480 mm | +0.734 mm | +5.293 mm | -0.0382° |

The audit initially returned `INSUFFICIENT_EVIDENCE`; subsequent durable sensitivity and closed-loop tests support `KEEP_JULY` at medium confidence, not a real-world success claim.

# 15. Corrected Real-State Isaac Closed-Loop Tests

All entries are **ISAAC CLOSED-LOOP**, not hardware results.

- Exact corrected state: success in 163 steps; entered, released at step 65, rear -64.55 mm/front 75.34 mm/insertion depth 102.66 mm, entered PUSH, final success. Initial motion included a 1.35 mm retreat.
- Ten small perturbations: 10/10 entered, released, and entered PUSH; 6/10 final success. Four failures were PUSH/final timeouts; zero premature and zero never-release.
- Corrected A/C: both entered and released; A final success, C later failed PUSH/final completion.

Final joint-perturbation results:

| profile | episodes | forward | entered | correct release | premature | never release | final success |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL | 50 | 50/50 | 47/50 | 47/50 | 0 | 3 | 36/50 (72%) |
| STRESS | 30 | 30/30 | 30/30 | 30/30 | 0 | 0 | 24/30 (80%) |

NORMAL release rear median -62.470 mm, range -71.682 to -59.460; insertion-depth median 101.847 mm. STRESS median -62.210 mm, range -65.003 to -59.324; depth median 101.336 mm. Maximum initial retreat was 9.54 mm NORMAL and 12.50 mm STRESS. The higher STRESS percentage is not evidence that stress helps: sets have different sizes and sampled cases. The three NORMAL tail failures progressed forward but never entered/released before timeout. Downstream failure after correct release occurred in 11 NORMAL and six STRESS cases.

# 16. Release Sensitivity Analysis

One-at-a-time raw-release influence, most to least, across corrected A/C midpoint and near-release states:

| rank | observation |
|---:|---|
| 1 | rear_to_mouth |
| 2 | tilt2 |
| 3 | tilt1 |
| 4 | yaw |
| 5 | lateral |
| 6 | gripper |
| 7 | tool Y |
| 8 | Z |
| 9 | tool Z |
| 10 | tool X |
| 11 | front_to_back |

Realistic ±2 mm lateral or Z, ±1–2° yaw/tilt, and replacing held semantic gripper with 0.39 caused no release crossing. Front-to-back had exactly zero effect over the tested ±30 mm because it was already clipped. Rear-to-mouth was the only swept variable to cross; it required substantially deeper states (for one A near-release state, roughly +35 mm from a -75 mm baseline toward -40 mm). Thus no small one-variable deployment-plausible perturbation reproduced premature release.

# 17. Joint Perturbation Robustness

**OFFLINE COUNTERFACTUAL**, exact July normalization and actor:

| profile | samples | release crossings | raw release min / median / p95 / max | X forward / reverse / clipped |
|---|---:|---:|---|---|
| NORMAL | 4000 | 0 | -9.816 / -7.268 / -5.668 / -4.640 | 55.83% / 44.18% / 88.93% |
| STRESS | 4000 | 0 | -10.903 / -6.378 / -4.519 / **-3.228** | 58.38% / 41.63% / 86.00% |

Motion-output clipping fractions `[X,Y,Z,yaw,pitch]` were `[.8893,.8323,.9943,.9158,.8488]` NORMAL and `[.8600,.8425,.9730,.8978,.8453]` STRESS. The stress perturbations jointly included rear ±3 mm, lateral/Z up to ±4 mm, yaw/tilts up to ±2°, historical tool-delta ranges, and gripper up to 0.39. Saturated raw policy actions are not themselves closed-loop failure; the 80 physics episodes above provide that test.

# 18. What We Have Ruled Out

Strongly unsupported by current evidence:

- Shelf movement itself does not demonstrate failure to generalize; the catastrophic moved-shelf observation used the old slot.
- The moved-shelf failure was caused by stale slot state, not a slot-X sign inversion.
- There is no P/E-to-I held-book transform handoff bug when the correct slot quaternion is used; both consume the same frozen `T_eef_book`.
- The apparent 14 mm Z/7° tilt handoff mismatch came from an offline replay with wrongly negated slot quaternion z/w, not runtime geometry.
- The 178 versus 200 mm front-to-back difference changes no July output in relevant saturated states.
- Realistic small one-variable errors did not cause premature release.
- Realistic joint perturbations caused 0/8,000 offline release crossings and 0/80 closed-loop premature releases.
- Corrected historical trajectories show 0/4 release crossings.
- Simulation and real release semantics both use action 5 `>0.5`; no hidden sim guard explains the old result.
- Current evidence does not support an intrinsic July premature-release defect.

# 19. What Remains Unknown

- Zero fully clean post-fix real closed-loop trials currently exist.
- Clean real July INSERT/release performance remains unmeasured.
- Physical PUSH reliability and actual contact/travel remain uncertain.
- Final physical book pose/success is not automatically recorded; software completion is insufficient.
- Three NORMAL Isaac tails failed to enter/release before timeout.
- Real Servo/contact/friction/calibration behavior can differ from Isaac; simulation cannot prove hardware safety.
- Historical A/B/C slot widths are not preserved in the compact audit.
- Some large DB3 bags lack metadata and were not assumed complete.
- Fresh sweep has three missing F2-20 profile results.

# 20. Current Policy Decision

**DECISION: KEEP_JULY**  
**CONFIDENCE: MEDIUM**

July has strong paired nominal simulation performance, every historically recoverable release failure is contaminated by the stale slot (and one by stale marker), corrected replay produces 0/4 crossings, realistic sensitivity produces 0/8,000 crossings, and corrected-state physics produces 0/80 premature releases with 77/80 correct releases. Fresh policies improve some synthetic corrupted profiles but sacrifice nominal performance and retain large never-release rates. Retraining is therefore not justified before testing the corrected deployment.

Retraining would become justified if repeated clean post-fix real trials show INSERT divergence, premature release, or never-release under small plausible slot/grasp variations, or if a broader, provenance-controlled corrected-state simulation bank exposes the same failure repeatedly. Correct INSERT/release followed by PUSH failure would justify PUSH work, not PPO retraining.

# 21. PUSH Status

PUSH is downstream and must be assessed separately. The real controller commands the existing 30 mm book push from release-time X, with nominal-plus-residual PPO. It does not add the documented ±5 mm uncertainty and has no contact sensor. Therefore actual travel may be roughly 25–35 mm even if commands execute as designed.

Corrected-state Isaac failures predominantly occur after correct release: 17/77 released episodes failed final success in the 80-case test, plus four of ten earlier perturbations. Historical real controller completion proves commands were sent, not that physical contact or seating succeeded. Next investigation should measure release pose, TCP approach/contact proxy, actual book displacement, and final book pose without conflating these with learned INSERT/release.

# 22. Current Software Readiness

Current Riot source includes:

- exact frozen policy-slot snapshot for observation, nominal, release/depth, and PUSH;
- fresh unique marker calibration: age≤0.25 s, unique stamps, ≥20/30 samples;
- robust per-grasp transform shared by P/E and I;
- semantic held-gripper observation with physical aperture separately logged;
- real GripperCommand and status-based completion;
- singularity-aware preinsert IK selection and validated table collision scene;
- asynchronous operator PLAN/REVIEW/EXECUTE gating;
- immutable release pose, separate PUSH estimate, 30 mm semantics;
- 15 s retreat timeout with unchanged 90 mm completion target.

Required pre-trial checks:

```text
POLICY SLOT SNAPSHOT source=frozen_accepted:/tmp/bookshelf_simple_frozen_slot.yaml ...
PER-GRASP EEF->BOOK CALIBRATION ... result=PASS ... fresh_unique>=20 ... age<=0.25 s
PER-GRASP EEF->BOOK FROZEN
```

If calibration fails, press C to retry; do not use `fixed_fallback` as the main validation. Last validation: eight freshness-focused tests and 26 related workflow tests passed; `scripts/ros2/build_xarm_experiment.sh` built all five selected packages; `git diff --check` passed. These latest two safety fixes remain uncommitted at Riot HEAD.

# 23. Final Real-World Experiments Still Needed

Minimum evidence plan:

1. One cautious, fully bagged July validation with exact frozen-slot log, fresh per-grasp PASS, stage-by-stage operator notes, and explicit physical outcome.
2. A small repeated set of clean trials spanning ordinary shelf/grasp variation, reporting INSERT entry/progression, release quality, PUSH entry/completion, and final placement separately.
3. Record or independently annotate book pose at release and after settling/PUSH so physical success is not inferred from `episode_complete`.

Do not switch to F1-40 for the first validation. Do not expand into a large hardware study until the single corrected pipeline run is verified.

# 24. Paper-Ready Quantitative Results

## A. Simulation/training

| result | value | evidence class |
|---|---:|---|
| nominal-only, frozen 3 mm bank | 28.85%, 577/2000 | ISAAC CLOSED-LOOP |
| PPO-only, three seeds | 0/2000 each | ISAAC CLOSED-LOOP |
| residual PPO, three seeds | 89.60%, 89.05%, 94.15% | ISAAC CLOSED-LOOP |
| residual mean ± SD | 90.93±2.80 pp | derived simulation |
| July final rolling training success | 91.5% | training metric, not evaluation |
| F1-40 nominal/physical-like/combined | 57.2% / 53.3% / 33.6% | ISAAC CLOSED-LOOP, paired 512/profile |
| July same profiles | 86.7% / 0.0% / 3.5% | ISAAC CLOSED-LOOP, paired 512/profile |

## B. Corrected-state robustness

| test | INSERT/release | final | evidence class |
|---|---:|---:|---|
| exact corrected takeover | 1/1 | 1/1 | ISAAC CLOSED-LOOP |
| ten perturbations | 10/10 release, 0 premature | 6/10 | ISAAC CLOSED-LOOP |
| NORMAL joint cases | 47/50 correct, 0 premature, 3 never | 36/50 | ISAAC CLOSED-LOOP |
| STRESS joint cases | 30/30 correct, 0 premature/never | 24/30 | ISAAC CLOSED-LOOP |
| offline NORMAL/STRESS | 0/4000 and 0/4000 release crossings | not applicable | OFFLINE COUNTERFACTUAL |

## C. Historical real/offline audit

| result | value | evidence class |
|---|---:|---|
| logs / usable INSERT / controller complete | 25 / 16 / 6 | REAL LOG AUDIT |
| clean historical trials | 0 | REAL LOG AUDIT |
| recovered stale-slot runs | 4/4 | REAL MEASURED + bag reconstruction |
| correct-slot release crossings | 0/4 | OFFLINE COUNTERFACTUAL |
| Run B marker freshness | 0 updates; age 1.8765 s | REAL MEASURED |
| A/B/C slot errors | 18.360 / 18.398 / 33.216 mm | REAL/OFFLINE geometry |

## D. Not real-world task-success evidence

Isaac outcomes, actor replay, `episode_complete`, commanded PUSH distance, and operator-believed outcomes without clean current logging cannot be reported as post-fix real task success or a real success rate.

# 25. Candidate Paper Figures/Tables

1. Method diagram: geometric observation → nominal controller + bounded PPO residual → release/scripted PUSH.
2. Frozen paired main comparison (nominal/PPO-only/residual) with paired-win annotation.
3. Clearance and reset-offset robustness curves.
4. Training curve for the selected July policy, clearly separated from evaluation.
5. Corrected-state robustness stage funnel: forward → entered → correct release → final, NORMAL/STRESS.
6. Release sensitivity plot emphasizing rear-to-mouth and showing safe realistic perturbation bands.
7. Real deployment frame/hand-off diagram showing frozen slot and per-grasp `T_eef_book` as immutable sources.
8. Final clean hardware table, once collected, separating INSERT, release, PUSH, and final placement.

The fresh F0–F4 sweep is best a compact ablation/supplement table unless the paper specifically studies failed robustness objectives.

# 26. Claims Currently Defensible

- On a paired frozen Isaac bank at 3 mm added clearance, residual PPO substantially outperformed nominal-only and PPO-only control.
- Residual control remained strong across 2–5 mm clearance and up to 1.5× reset offsets, except at the hardest 1 mm setting.
- A task-space 12-D/6-D policy can be transferred across robot embodiments through explicit frame transforms and bounded Cartesian commands.
- Historical real failures were materially contaminated by stale task geometry; four recovered runs used the wrong slot and one also used stale repeated marker data.
- Correcting recorded geometry removed all four historical release crossings in counterfactual inference.
- Under tested corrected real-derived states, small realistic individual and joint perturbations did not induce premature release; physics failures were mainly insertion timeouts or post-release completion.
- Current evidence supports keeping July for a clean validation rather than retraining immediately.

# 27. Claims NOT Yet Defensible

- Successful or robust post-fix sim-to-real deployment.
- Any post-fix real success rate.
- That `episode_complete` means physical success.
- That Run A was a clean validation; it used the stale policy slot.
- Reliable physical PUSH/contact or guaranteed 30 mm book travel.
- That all sim-to-real error is removed, July is universally robust, or retraining will never be needed.
- That fresh DR solved the never-release pathology.
- That Isaac proves real hardware safety.

# 28. Recommended Final Paper Story

The main scientific narrative should be: tight-clearance bookshelf insertion benefits from decomposing control into a geometry-based nominal policy and a bounded learned residual, including learned release. A paired Isaac study demonstrates the performance gain and robustness envelope. A robot-agnostic task-space interface enables xArm7 deployment, while explicit per-trial state freezing addresses grasp and scene-estimation uncertainty. Final clean real trials will establish the hardware result and separate learned INSERT/release from scripted post-release PUSH.

Implementation details belong in methods: observation/action definitions, normalization, nominal gains, release semantics, per-grasp calibration, IK selection, and safety-gated operator flow. Debugging evidence—obsolete gripper interfaces, status races, wrong replay quaternion, Candidate 5, stale slot/marker forensic detail—should remain internal or supplementary except where needed to define valid experimental protocol. Deployment bugs are not themselves scientific contributions.

# 29. Exact File / Result / Checkpoint Index

## Simulation/training source

- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env.py`
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env_cfg.py`
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_v4.py`
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_v5.py`
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_cfg_v4.py`
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_cfg_v5.py`
- `scripts/sb3/train.py`, `scripts/sb3/play.py`, `scripts/sb3/evaluation_scenarios.py`
- Fresh profiles: `scripts/sb3/targeted_dr_profiles.py`

## Simulation results/checkpoints

- July full checkpoint/VecNormalize: `/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/`
- Main aggregation: `/home/chris/BookshelfFiles/evaluation_results/paper_simulation_20260816/`
- Frozen multiseed: `/home/chris/BookshelfFiles/evaluation_results/frozen_multiseed_20260816/`
- Clearance/offset: `/home/chris/BookshelfFiles/evaluation_results/clearance_sweep_20260816/`, `offset_sweep_20260816/`
- Frozen banks: `/home/chris/BookshelfFiles/evaluation_results/frozen_banks/`
- Fresh F0–F4 checkpoints: `/fred/oz430/tliu/code/bookshelf-targeted-dr-20260902/logs/sb3/Bookshelf-Residual-Direct-v0/fresh_release_20260902_F{0..4}/`
- Fresh evaluation: `/fred/oz430/tliu/evaluation_results/bookshelf/fresh_model_selection_20260903/`
- F1-40 model: `fresh_release_20260902_F1/model_fresh_40000000_steps.zip`, SHA `c63e4683afc62b297c13c436b47c47647de488beb7d35ce2727dbdd93c1ea471`
- F1-40 VecNormalize: `model_vecnormalize_fresh_40000000_steps.pkl`, SHA `ab5f2b22a6fb66d7302fe3e5d4d7697ee558ae1953739899bdfc9195b1c4d14b`

## Real runtime

- Controller: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/simple_policy_control_node.py`
- Preinsert: `.../preinsert_node.py`; calibration: `.../per_grasp_calibration.py`
- Observation/controller/PUSH math: `.../policy_observation_math.py`, `.../residual_policy_math.py`, `.../post_insert_math.py`
- Operator: `.../operator_console_node.py`, `.../operator_action_node.py`
- Configs: `ros2/bookshelf_simple_experiment_ros/config/simple_policy_control.yaml`, `simple_preinsert.yaml`
- Launch: `ros2/bookshelf_simple_experiment_ros/launch/real_experiment_operator.launch.py`, `real_preinsert_workflow.launch.py`, `simple_policy_one_step.launch.py`
- Hardware/Servo: `ros2/bookshelf_policy_ros/launch/physical_hardware_bringup.launch.py`, `robot_setup.launch.py`, `xarm7_moveit_servo_server.launch.py`
- Hand-eye/camera: `camera_setup.launch.py`, `publish_handeye_camera_link.launch.py`
- Servo config: `/home/riot/Chris/ros2_ws/src/xarm_ros2/xarm_moveit_servo/config/xarm_moveit_servo_config.yaml`
- Session checklist: `REAL_XARM7_SESSION.md`; build: `scripts/ros2/build_xarm_experiment.sh`.

## Real actors/configuration

- July actor: `/home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz`
- F1 backup: `/home/riot/BookshelfFiles/trained_models/bookshelf_fresh_F1_40M_backup_actor.npz`
- Approved configuration: `/home/riot/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml`
- Marker mount: `ros2/bookshelf_shadow_ros/config/real_book_aruco0_mount.yaml`
- Scan/loading poses: `/home/riot/BookshelfFiles/experiment_configs/operator_joint_poses/{scan_joint_state,loading_joint_state}.yaml`
- Current-cycle slot: `/tmp/bookshelf_simple_frozen_slot.yaml` (ephemeral, never historical evidence).

## Real logs/bags

- Logs root: `/home/riot/BookshelfFiles/experiment_logs/simple_policy_*/policy_step.jsonl`
- Bags root: `/home/riot/BookshelfFiles/experiment_logs/full_real_bags/`
- Early release: `simple_policy_20260902_001927/`, bag `full_real_20260902_001937/`
- A: `simple_policy_20260904_003219/`, bag `full_real_20260904_003338/`
- B: `simple_policy_20260904_004258/`, bag `full_real_20260904_004324/`
- C: `simple_policy_20260904_005654/`, bag `full_real_20260904_005719/`
- Moved shelf: `simple_policy_20260904_012635/policy_step.jsonl`

## Diagnostic/audit outputs and scripts

- Audit: `/tmp/bookshelf_real_evidence_audit/audit_summary.json`
- Corrected states: `/tmp/bookshelf_real_evidence_audit/corrected_takeover_states.json`
- A/C closed-loop: `/tmp/bookshelf_real_evidence_audit/closed_loop_results.json`
- Sensitivity: `/tmp/bookshelf_real_evidence_audit/front_to_back_sensitivity.csv`, `release_sensitivity.csv`, `release_sensitivity_summary.json`
- Durable final robustness: `/home/riot/BookshelfFiles/evaluation/july_final_retrain_decision/`
- Durable summary: `final_decision_summary.json`; samples: `joint_offline_samples.csv`; Isaac: `isaac_joint_cases.json`, `isaac_joint_results.json`.
- Helpers: `scripts/sb3/real_policy_evidence_audit.py`, `real_release_counterfactual.py`, `july_corrected_takeover_closed_loop.py`, `july_joint_robustness.py`, `summarize_july_final_robustness.py`, `summarize_model_selection.py`.
- IK diagnostics: `scripts/xarm7_preinsert_ik_branches.py`, `scripts/xarm7_candidate5_transition_plan.py`, `scripts/xarm7_candidate5_servo_diagnostic.py`.
- Paper assets: `/home/chris/Chris/bookshelf-unified/paper/figures/` and generators under `scripts/paper/` on Alienware; both were untracked at snapshot time.

# 30. Project State at 2026-09-04

**DONE**

- Frozen paired Isaac comparison and robustness sweeps.
- July checkpoint/export identity and exact 12-D/6-D contract.
- Singularity-aware reviewed real preinsert workflow.
- Per-grasp held-book transform and semantic gripper mapping.
- Root cause and fixes for stale policy slot and stale cached marker calibration.
- Historical evidence audit, corrected replay, sensitivity, and 8,000 offline/80 closed-loop joint-perturbation tests.
- Fresh F0–F4 training and 179 completed paired profile evaluations.

**CURRENT DECISION:** `KEEP_JULY`, medium confidence. F1-40 remains backup only.

**REMAINING WORK:** commit/freeze the latest Riot safety fixes; conduct clean trials; record stage-separated physical outcomes and final book pose; diagnose PUSH only if INSERT/release are clean.

**FINAL HARDWARE REQUIREMENT:** at least one fully bagged validation, then a small repeated set, with the exact frozen slot, fresh unique per-grasp PASS, July semantic gripper, reviewed motion, and explicit physical INSERT/release/PUSH/final labels.

**PAPER STATUS:** simulation contribution is quantitatively defensible. Historical real diagnostics and protocol fixes are defensible when clearly labeled. Post-fix real task-success and PUSH-reliability claims must wait for clean hardware evidence.
