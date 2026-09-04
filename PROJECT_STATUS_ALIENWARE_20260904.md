# Bookshelf residual-RL project: Alienware snapshot

**Snapshot date:** 2026-09-04 (Australia/Melbourne)  
**Machine:** Alienware PC (`/home/chris`)  
**Scope:** simulation, training, evaluation, checkpoints, diagnostics, and paper-side artifacts available locally. This is a documentation/audit snapshot only. No training, ROS launch, hardware process, or new long evaluation was run.

## Executive status

The best-supported simulation result is the frozen, paired 2,000-scenario evaluation at 3 mm lateral clearance. The nominal controller succeeded in 577/2,000 cases (28.85%); the July residual-PPO checkpoint succeeded in 1,792/2,000 (89.60%, seed 42), 1,781/2,000 (89.05%, seed 123), and 1,883/2,000 (94.15%, seed 2026), for 90.93% mean and 2.80 percentage-point sample standard deviation. PPO-only checkpoints scored 0/2,000 for all three seeds. These are same-simulator, same-protocol results, not real-robot validity.

The deployed/reused residual policy is the 2026-07-08 Panda policy:

- model: `/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/model.zip`
- VecNormalize state: `/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/model_vecnormalize.pkl`
- model SHA-256: `80f7aa2d6675a99f3965b2479bc0b62f5f3320e724a6f3399efacc1640b3b4ed`
- VecNormalize SHA-256: `88670e59194fa5d70743872ea02232acdbea93f827fbce116d1fcb4a0745a635`

The current repo contains later xArm/ROS transfer work, but this snapshot does not treat hardware-side behavior as simulation evidence. The requested labels `F0`–`F4`, `physical-like`, `combined`, `strong`, and `actuation` do not occur in any surviving result manifest, JSON/CSV filename, training command, repository source, or local terminal transcript found by this audit. Their numeric table and label-to-checkpoint mapping therefore cannot be reconstructed responsibly from this PC. Closest surviving robustness and corrected-state results are documented below.

## 1. Git and repository state

Repository:

- path: `/home/chris/Chris/bookshelf-unified`
- remote: `origin = git@github.com:Chris-airobot/Bookshelf-residual-RL.git`
- branch: `simple-real-experiment`
- HEAD: `c6a3c19a0818da7236a220b3a0e41cfcef27bc80`
- HEAD subject/date: `2026-09-03 Add xArm7 offline hardware rehearsal`
- relation to `origin/simple-real-experiment`: local is 6 commits ahead and 0 behind (`0 6` from `git rev-list --left-right --count origin/simple-real-experiment...HEAD`)
- tracked modifications: none
- untracked paths at audit start: `install/`, `log/`, `paper/`, `scripts/paper/`, `scripts/sb3/capture_execution_sequence.py`, and `scripts/sb3/july_corrected_takeover_closed_loop.py`
- this status file is intentionally the only audit-created project file.

Important commits for the simulation/training/evaluation story:

| Commit | Date | Relevance |
|---|---:|---|
| `14125ff` | earlier history | nominal-controller correction |
| `c283946` | earlier history | final training setup |
| `bfb1e64` | earlier history | 3 mm residual curriculum configuration |
| `6f280ce` | earlier history | true PPO resume behavior |
| `95f5838` | earlier history | portable VecNormalize resume |
| `631ac8a` | earlier history | nominal training working state |
| `e5584ea` | 2026-08 | PPO-only baseline task |
| `8a76ebf` | 2026-08 | frozen robustness evaluation and paper artifacts |
| `1a5586e` | 2026-08-23 | full xArm7 residual training setup |
| `10a6f6b` | 2026-08-23 | xArm policy episode simulation and physical handoff |
| `7c48302` | 2026-08-28 | post-release pose evaluation logger |
| `7022e29` | 2026-08-28 | post-release checkpoint-loading fix |
| `3cdcabc` | 2026-08-28 | release-pose estimate used for robust push |
| `8e4e57d`, `1055696`, `05fba2e`, `c6a3c19` | 2026-09-01–03 | real-workflow support and offline rehearsal; outside the paper's core simulation claim |

The current source is later than the July checkpoint. Historical claims must use the copied run configuration under the checkpoint directory, not silently substitute the current Python defaults.

## 2. Simulator, task, and environment

### Simulator stack

- Isaac Sim installation: `/home/chris/isaacsim`
- Isaac Sim version file: `/home/chris/isaacsim/VERSION`
- exact installed version: `5.1.0-rc.19+release.26219.9c81211b.gl`
- Isaac Lab checkout: `/home/chris/IsaacLab`
- Isaac Lab commit: `3e73d6dd79080fd7632488c061052a6edd52e230`
- nearest local description: `training-checkpoints-develop-4-g3e73d6dd7908`
- core extension version in `/home/chris/IsaacLab/source/isaaclab/config/extension.toml`: `0.54.3`
- Isaac Lab checkout state: branch `main`, aligned with `origin/main`; untracked `docker/Dockerfile.bookshelf` and `source/bookshelf` exist.

The project task is registered as `Bookshelf-Residual-Direct-v0` in:

- `/home/chris/Chris/bookshelf-unified/source/bookshelf/bookshelf/tasks/direct/bookshelf/__init__.py`
- environment config: `/home/chris/Chris/bookshelf-unified/source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env_cfg.py`
- environment implementation: `/home/chris/Chris/bookshelf-unified/source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env.py`
- inherited randomized-row geometry: `bookshelf_env_cfg_v5.py` and `bookshelf_env_v5.py`
- inherited base geometry/metrics: `bookshelf_env_cfg_v4.py` and `bookshelf_env_v4.py`

The robot is a Franka Panda with joint-position targets. Physics runs at `dt = 1/120 s`; action decimation is 2, so the policy/controller rate is 60 Hz. The configured episode is 10 s, approximately 600 policy steps. The default scene has 4,096 replicated environments with 2.0 m spacing; training commands may override the environment count.

### Book and shelf geometry

The manipulated book uses `_BOOK_LWH = (0.156, 0.236, 0.034)` m: 156 mm insertion depth (x), 236 mm height (z in the upright pose), and 34 mm lateral thickness (y). Configured mass is 0.45 kg, with static/dynamic friction 1.8/1.5, linear damping 0.2, and angular damping 4.0.

Current residual-task shelf geometry:

- slot mouth/open x: 0.63 m
- slot back x: 0.83 m
- nominal depth: 0.20 m
- slot center y: 0.0 m
- shelf top z: 0.05 m
- shelf deck thickness: 0.02 m
- current residual clearance range: exactly 0.003 m (`min = max = 3 mm`)
- total open width for the 34 mm book: 37 mm, i.e. 1.5 mm free space on either side when centered.

The v5 row has 10 logical book positions and samples one missing position for the target slot. Side books are kinematic rigid objects: their layout changes at reset but they do not move during an episode. One-slot books may be merged into two-slot books with probability 0.35. One-slot heights are `[229, 205, 185, 218, 195, 229, 202, 175, 214]` mm; two-slot heights are `[215, 190, 229, 200]` mm. Top-shelf clearance is 25 mm. The v5 docstring's “23–26 mm physical gap width” predates/inconsistently describes the present 34 mm target-book thickness; the executable geometry and July run snapshot are the authoritative sources.

### Reset randomization and curriculum

The July run's copied `params/env.yaml` and current residual config agree on the relevant curriculum structure:

| Curriculum stage | Progress interval | Arm joints | Grasp x | Grasp y | Grasp z | Grasp yaw |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0–20% | ±1.5° | ±3 mm | ±3 mm | ±1.5 mm | ±3° |
| 2 | 20–50% | ±2° | ±5 mm | ±4 mm | ±2 mm | ±5° |
| 3/final | 50–100% | ±3° | ±8 mm | ±6 mm | ±3 mm | ±8° |

`residual_curriculum_total_steps = 260,416`, with fractions 0.2, 0.5, and 1.0. The environment advances this curriculum using its common/vector step counter, not SB3's scalar transition count. With 4,096 iterations × 32 rollout steps, the July run performed 131,072 vector steps and crossed the 50% boundary only at the end; this is distinct from the 33,554,432 total per-environment transitions reported by SB3. Slot-clearance curriculum and residual-scale curriculum are disabled for the July/current residual task.

### Exact raw observation semantics (12-D)

The policy receives a 12-D raw vector; each component is clipped to `[-1, 1]` after the indicated task scaling, then the saved VecNormalize running statistics are applied and normalized observations are clipped to `[-10, 10]`.

| Index | Raw semantic | Task scaling before VecNormalize |
|---:|---|---|
| 0 | mode: INSERT `0.0`, SCRIPTED `0.5`, PUSH `1.0` | already scalar |
| 1 | rear edge to slot mouth: `book_rear_x - mouth_x` | divide by 0.08 m |
| 2 | book front to slot back: `back_x - book_front_x` | divide by 0.08 m |
| 3 | lateral error: `slot_center_y - book_center_y` | divide by 0.05 m |
| 4 | vertical error from shelf-supported target center | divide by 0.05 m |
| 5 | wrapped book yaw error | divide by 30° |
| 6 | tool x minus book-center x | divide by 0.25 m |
| 7 | tool y minus book-center y | divide by 0.25 m |
| 8 | tool z minus book-center z | divide by 0.25 m |
| 9 | gripper-open fraction | linear map of mean finger position from 0.015 m closed to 0.060 m open |
| 10 | world x component of the rotated local spine/upright axis `(0,1,0)` | direct component |
| 11 | world y component of that same axis | direct component |

Important clipping consequence: the raw target state used by the corrected-state diagnostics has `rear_to_mouth = -0.186211 m` and `front_to_back = 0.230009 m`; both become saturated (`-1` and `+1`) before VecNormalize. The policy therefore cannot distinguish further magnitude changes in those dimensions outside ±80 mm. A July `experiment_spec.json` found in the run directory describes a 16-D schema and has false/obsolete source-file assertions; it is not consistent with the executable 12-D environment or its copied `env.yaml` and should not be cited as the final observation contract.

### Exact action semantics and controller

The policy action is 6-D and is clamped to `[-1, 1]`:

| Index | Semantic | Residual scale per 60 Hz control step |
|---:|---|---:|
| 0 | residual x translation | 2.0 mm |
| 1 | residual y translation | 1.0 mm |
| 2 | residual z translation | 1.5 mm |
| 3 | residual yaw | 0.35° |
| 4 | residual pitch | 0.30° |
| 5 | release request | accepted when `> 0.5` in INSERT |

Nominal and residual commands are added, then final per-step limits are applied: x ±8 mm, y ±3 mm, z ±7 mm, yaw ±0.8°, pitch ±0.6°.

INSERT nominal behavior:

- Translation in x is enabled only when `|lateral| < 6 mm`, `|z error| < 10 mm`, `|yaw| < 6°`, and upright tilt-x `< 0.1`.
- When aligned, nominal x is +1.0 mm/step until `rear_to_mouth > -35 mm`, then +0.7 mm/step.
- Lateral correction is `0.25 × lateral_error`, limited to 1.5 mm/step.
- Vertical correction is `-0.18 × (z_error - 6 mm)`, limited to 1.8 mm/step.
- Yaw correction is `-0.14 × yaw_error`, limited to 0.35°/step.
- Pitch correction is `-0.02 × tilt_x`, limited to 0.25°/step.

PUSH nominal behavior:

- x is +0.8 mm/step;
- y, z, yaw, and pitch gains are 0.35, 0.30, 0.20, and 0.08 respectively;
- y and z corrections are limited to 0.5 and 1.0 mm/step;
- push height is targeted 20% up from the book bottom.

### Release, scripted transition, and PUSH

Current/default `policy_release_guard_mode` is `none`; thus any policy release action above 0.5 in INSERT is accepted without a geometry gate. `premature_release_penalty = 0.5` is configured, but with guard mode `none` there are no blocked release requests and this term is effectively zero. Nominal release assist is disabled.

An observable release mask remains implemented for experiments: inside fraction ≥0.5, `|lateral| < 10 mm`, `|z| < 18 mm`, `|yaw| < 8°`, tilt-x `< 0.12`, and `front_to_back ≥ 15 mm`. It is not the current default guard.

After release, SCRIPTED mode opens the gripper for 3 policy steps, retreats for 6 configured steps using a `-15 mm` x target increment per step, and closes for 5 steps, then changes to PUSH. INSERT and PUSH command the closed gripper. These values are commanded targets; they do not prove the same achieved physical displacement.

### Success and failure definitions

Success can only be asserted in PUSH, after at least 5 push steps, and must hold for 4 consecutive policy steps. The executable gates are:

- `rear_to_mouth ≥ -12 mm`;
- `front_to_back ≤ 55 mm + 0.2 mm epsilon`;
- lateral book extent within half the opening plus 1.5 mm epsilon;
- `|z error| < 15 mm`;
- `|yaw| < 8°`;
- upright dot product ≥0.85;
- velocity limits are disabled (`0` means no threshold).

Drop is based on the lowest book corner. During INSERT it is compared with true ground using a 2 mm threshold; after release it uses the shelf-support/floor threshold (0.042 m unless the support test applies). Out-of-bounds and generic fell terminations are disabled in the current residual config. Episodes otherwise end on success, drop, or timeout. Logged outcome/failure categories include `success`, `drop`, `oob`, `fell`, `not_push`, `depth`, `lateral`, `z`, `yaw`, `upright`, `unstable`, and `timeout`.

## 3. Training state

### July production policy

Run directory:

`/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04`

Recorded command:

`/home/chris/isaacsim/kit/python/bin/python3 scripts/sb3/train.py --task Bookshelf-Residual-Direct-v0 --num_envs 256 --max_iterations 4096 --headless`

The saved command text lacks a space before `--headless` in one concatenated rendering, but the run artifacts show the intended arguments above. Total SB3 transitions were `4,096 × 32 × 256 = 33,554,432`. Intermediate models were saved every 256,000 transitions through 33,536,000, plus the final `model.zip`.

Architecture and PPO hyperparameters from `params/agent.yaml` / `agents/sb3_ppo_cfg.yaml`:

- algorithm: Stable-Baselines3 PPO
- policy: `MlpPolicy`
- actor and critic hidden sizes: 256, 256 with ReLU
- seed: 42
- device: `cuda:0`
- rollout length `n_steps`: 32 per environment
- batch size: 8,192
- epochs/update: 10
- learning rate: `1e-4`
- discount `gamma`: 0.99
- GAE lambda: 0.95
- clip range: 0.2
- entropy coefficient: 0.003
- value coefficient: 1.0
- max gradient norm: 1.0
- observation normalization: enabled via VecNormalize, `clip_obs = 10.0`
- reward normalization: disabled (`normalize_value` is absent, and `train.py` defaults `norm_reward` to false); reward clipping is infinite.

Final row of `training_summary.csv` at 33,554,432 transitions, using the latest 1,000 completed episodes:

- success rate: 0.915
- timeout rate: 0.000
- drop rate: 0.000
- mean return: 87.870546
- mean episode length: 150.738 steps
- mean final lateral error: 2.7639 mm
- mean final z error: 4.3131 mm
- mean final yaw error: 0.6533°
- total episode rows accumulated by then: 231,077

The full `episode_metrics.csv` is 48,733,167 bytes. A cumulative plot summary records 194,157 success, 33,756 `not_push`, 3,153 `depth`, and 11 timeout outcomes. Those cumulative categories and the final rolling 1,000-episode rates answer different questions and must not be merged into one percentage.

### Earlier progression and F0–F4 evidence status

The local training archive contains many historical `Bookshelf-Direct-v1` through `v5` and residual runs under `/home/chris/BookshelfFiles/training_runs/sb3/`. The documented development path was manual demonstrations → behavior cloning → PPO, followed by a nominal+residual task. Earlier notes identify a wide-slot checkpoint with ~99% training/evaluation behavior and a later 3 mm curriculum, but the current paper result is based on the July residual checkpoint and the frozen August evaluation protocol, not those informal early-stage numbers.

Two intact pre-residual milestones are present:

- `/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Direct-v4/2026-05-11_01-15-47/model.zip`, SHA-256 `dfabe60644e31b95278a6eb4a33d122981c545661f6a14f77ff4780d8d878aa1`; BC-initialized PPO, recorded command uses `data/bc/bc_init_sb3.zip` and 64 environments.
- `/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Direct-v4/2026-05-24_12-20-15/model.zip`, SHA-256 `ca8e3201f7f797a1ce1d5e8db439b82a16b40eaa2761718173e0e496851db1c9`; PPO continuation from the May 11 checkpoint. Its command file is truncated after `--max_iterations 24414`, so the on-disk checkpoint is intact but the command provenance is incomplete.

These v4 milestones used the older 10-D observation / 5-D action contract and cannot be loaded as current 12-D / 6-D residual policies without an explicit compatible conversion. Current repo BC assets include `data/bc/bc_init_sb3.zip`, `data/bc/bc_policy_best.pt`, `data/bc/v5/bc_policy_best.pt`, `data/bc/v5_release_guarded/{bc_init_sb3.zip,bc_policy_best.pt,bc_init_sb3_obs_stats.npz}`, and `data/bc/v5_release_weighted/bc_policy_best.pt`. Their presence documents the BC lineage; no current result manifest makes any of them a paper baseline.

No durable manifest on this PC defines F0, F1, F2, F3, or F4, and no artifact connects those labels to checkpoint hashes or to the requested `physical-like`, `combined`, `strong`, or `actuation` scenarios. Consequently:

| Requested item | Recoverable current evidence |
|---|---|
| F0–F4 definitions/checkpoint paths | missing |
| fresh-policy F1/F2/F3/F4 quantitative results | missing |
| physical-like / combined / strong / actuation named table | missing |
| why each named F policy was rejected | no label-safe provenance; cannot assign reasons |

There are, however, surviving fresh xArm training attempts in `/home/chris/Chris/bookshelf-unified/logs/sb3/Bookshelf-XArm7-Residual-Direct-v0/` from 2026-08-23. They are not relabeled as F1–F4 here. Key recorded outcomes are:

- 131,072-transition run `2026-08-23_02-12-52`: 0 success, 100% drop in the final 1,000-episode window.
- 8,192-transition guard-none run `03-36-15`: 0 success, 100% drop.
- 8,192-transition observable-geometry-guard run `03-40-52`: 0 success, 100% drop.
- 132,096-transition no-reset-acceptance-gate run `16-29-44`: 0 success, 100% drop.
- 32,768-transition no-gate run `16-50-29`: 0 success, 100% drop.
- 32,768-transition release-logit-mean `-3.0` run `17-03-48`: 0 success, 100% drop.
- 524,288-transition release-logit-mean `-3.0` run `17-16-17`: 0 success, 100% drop; mean episode length 37.822.
- interrupted `17-34-23` run reached 236,288 transitions: 0 success, 99.9% drop, 0.1% timeout.

The evidence-safe rejection reason for these fresh xArm attempts is simply that none demonstrated any successful episode and nearly all episodes ended in drop. They are not replacements for the July policy. Short 2–4 iteration runs are smoke/diagnostic runs, not trained-policy evaluations.

## 4. Main simulation results

### July nominal and historical evaluation

`/home/chris/BookshelfFiles/evaluation_results/2026-07-07_11-33-28/summary.csv` records 2,000 episodes per row at 3 mm clearance:

| Reset regime | Recorded seeds | Success | Drop | Timeout |
|---|---|---:|---:|---:|
| current noise | 42, 123, 456 | 1,916/2,000 = 95.8% (rounded to 96%) in every row | 79/2,000 = 3.95% | 5/2,000 = 0.25% |
| old reset noise | 42, 123, 456 | 1,759/2,000 = 87.95% (rounded to 88%) in every row | 234/2,000 = 11.7% | 7/2,000 = 0.35% |

All three purported seeds produce identical counts, so this table is not evidence of independent seed variation. The later frozen-bank evaluation is the defensible comparison.

### Frozen 3 mm main comparison

Source: `/home/chris/BookshelfFiles/evaluation_results/paper_simulation_20260816/tables/main_3mm_results.csv` and associated manifests; 2,000 identical bank scenarios per policy/run.

| Method | Seed | Success |
|---|---:|---:|
| nominal only | deterministic | 577/2,000 = 28.85% |
| PPO only | 42 | 0/2,000 = 0.00% |
| PPO only | 123 | 0/2,000 = 0.00% |
| PPO only | 2026 | 0/2,000 = 0.00% |
| residual PPO | 42 | 1,792/2,000 = 89.60% |
| residual PPO | 123 | 1,781/2,000 = 89.05% |
| residual PPO | 2026 | 1,883/2,000 = 94.15% |

Residual mean = 90.933%; sample SD = 2.799 percentage points; reported t-based 95% CI = 83.98–97.89%. The wide CI reflects only three policy seeds.

Paired McNemar counts against nominal on the same scenarios:

| Residual seed | nominal only wins | residual only wins | both succeed | both fail | p-value |
|---:|---:|---:|---:|---:|---:|
| 42 | 61 | 1,276 | 516 | 147 | `1.6985e-296` |
| 123 | 60 | 1,264 | 517 | 159 | `3.6477e-294` |
| 2026 | 26 | 1,332 | 551 | 91 | reported `0.0` from numerical underflow |

### Clearance robustness

Each entry is success percent over the frozen bank. Residual is mean ± sample SD across the three policy seeds. PPO-only is 0% at every clearance.

| Added lateral clearance | Nominal | Residual seeds (42, 123, 2026) | Residual mean ± SD |
|---:|---:|---:|---:|
| 1 mm | 9.70 | 4.80, 9.45, 13.70 | 9.32 ± 4.45 |
| 2 mm | 20.00 | 52.00, 52.65, 60.25 | 54.97 ± 4.59 |
| 3 mm | 28.85 | 89.60, 89.05, 94.15 | 90.93 ± 2.80 |
| 4 mm | 34.20 | 92.15, 97.35, 97.90 | 95.80 ± 3.17 |
| 5 mm | 42.30 | 94.85, 97.40, 97.65 | 96.63 ± 1.55 |

### Reset-offset robustness

| Offset scale | Nominal | Residual seeds (42, 123, 2026) | Residual mean ± SD |
|---:|---:|---:|---:|
| 0.00× | 28.00 | 90.75, 94.10, 96.50 | 93.78 ± 2.89 |
| 0.50× | 26.65 | 90.60, 89.65, 94.45 | 91.57 ± 2.54 |
| 1.00× | 18.10 | 87.60, 86.15, 91.60 | 88.45 ± 2.82 |
| 1.25× | 15.70 | 85.40, 83.65, 87.90 | 85.65 ± 2.14 |
| 1.50× | 15.40 | 81.10, 79.15, 83.35 | 81.20 ± 2.10 |

The offset sweep is the closest durable artifact to a “strong” perturbation result; it must not be renamed to `strong` unless the missing experiment manifest establishes that mapping. No surviving table isolates an “actuation” perturbation.

### Never-release, premature-release, and corrected-state diagnostics

The untracked offline/Isaac diagnostic script is:

`/home/chris/Chris/bookshelf-unified/scripts/sb3/july_corrected_takeover_closed_loop.py`

Its target raw observation is:

`[0, -0.186211, 0.230009, 0.000703, 0.005741, -0.000746, -0.038644, -0.004081, 0.003309, 0.009838, 0.000750, -0.000086]`

Persistent results:

- exact: `/home/chris/BookshelfFiles/evaluation/july_corrected_takeover/exact/result.json`
- 10 perturbations: `/home/chris/BookshelfFiles/evaluation/july_corrected_takeover/perturbed/result.json`

Exact case: success, 163 steps, entered slot, released, entered PUSH, and post-push success. Release occurred at step 65 with `rear_to_mouth = -64.55 mm`, `front_to_back = 75.34 mm`, and inferred insertion depth 102.66 mm. The first rear progress was negative (about -1.35 mm): the policy initially retreated before later moving forward.

Ten perturbed cases: 6/10 success; 10/10 entered the slot, released, and entered PUSH; 6/10 reached post-push success. Release occurred with `rear_to_mouth` from -64.84 to -59.92 mm (mean -63.07 mm). Using the diagnostic's premature threshold of `rear_to_mouth < -78 mm`, this set has 0 premature releases and 0 never-release cases. All ten had negative first rear progress; mean was -0.989 mm.

The achieved observation reconstruction differs from the requested target almost entirely in `front_to_back` by approximately -22.0 mm. Therefore “exact” means an exact requested input state, not exact achieved equality of all 12 raw components in Isaac.

Volatile `/tmp` artifacts generated on this Alienware on 2026-09-04:

- `/tmp/bookshelf_corrected_takeover_states.json`: two corrected raw states derived from Riot logs, both marked `SLOT_STALE`; this is imported real-derived diagnostic input, not an Alienware real-robot result.
- `/tmp/bookshelf_real_evidence_closed_loop.json`: two Isaac reconstructions; 1/2 success, both entered slot/released/entered PUSH, one failed post-push.
- `/tmp/july_isaac_joint_cases.json`: 80 joint-perturbation cases, 50 NORMAL and 30 STRESS.
- `/tmp/july_isaac_joint_results.json`: matching 80 Isaac closed-loop results.

The 80-case result summary is:

| Profile | Cases | Success | Entered slot | Released | Correct release | Premature | Never release | Entered PUSH | Post-push success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NORMAL | 50 | 36 (72.0%) | 47 | 47 | 47 | 0 | 3 | 47 | 36 |
| STRESS | 30 | 24 (80.0%) | 30 | 30 | 30 | 0 | 0 | 30 | 24 |
| total | 80 | 60 (75.0%) | 77 | 77 | 77 | 0 | 3 | 77 | 60 |

NORMAL release `rear_to_mouth` ranged from -71.68 to -59.46 mm; STRESS ranged from -65.00 to -59.32 mm. Maximum initial retreat reached 9.54 mm NORMAL and 12.50 mm STRESS. The counter-intuitive higher STRESS success is only a result on this small, non-identical set (30 vs 50 cases), not evidence that stress improves the policy. These files have no embedded Git commit, model hash, or top-level summary, and `/tmp` is not durable storage.

This is an 80-sample joint perturbation run, not the requested 8,000-sample result. No 8,000-sample JSON/CSV/log was found. Likewise, no durable one-variable sensitivity ranking was found. Source inspection supports one qualitative conclusion only: dimensions 1 and 2 are saturated by the task's ±80 mm pre-normalization clipping at the corrected starting state; it does not support inventing a quantitative ranking of the remaining dimensions.

### Small current-source evaluations

- `/home/chris/Chris/bookshelf-unified/logs/eval_scenarios/2026-08-28_20-10-00_Bookshelf-Residual-Direct-v0_seed42/summary.json`: 20 episodes, 17 success and 3 drop (85% success), July checkpoint, commit `7c48302`.
- `/home/chris/Chris/bookshelf-unified/logs/eval_scenarios/2026-09-04_02-20-30_Bookshelf-Residual-Direct-v0_seed42/summary.json`: 1/1 success, reward 97.547 and length 200 in its episode CSV, July checkpoint, current HEAD `c6a3c19`.

These small runs are smoke/current-source checks and do not supersede the frozen 2,000-scenario bank.

## 5. Checkpoints, result assets, and paper material

Core locations:

- historical training archive: `/home/chris/BookshelfFiles/training_runs/sb3/`
- evaluation archive: `/home/chris/BookshelfFiles/evaluation_results/`
- frozen scenario banks: `/home/chris/BookshelfFiles/evaluation_results/frozen_banks/`
- main three-seed results: `/home/chris/BookshelfFiles/evaluation_results/frozen_multiseed_20260816/`
- clearance sweep: `/home/chris/BookshelfFiles/evaluation_results/clearance_sweep_20260816/`
- reset-offset sweep: `/home/chris/BookshelfFiles/evaluation_results/offset_sweep_20260816/`
- paper aggregation: `/home/chris/BookshelfFiles/evaluation_results/paper_simulation_20260816/`
- current repo evaluations: `/home/chris/Chris/bookshelf-unified/logs/eval_scenarios/`
- xArm fresh-run archive: `/home/chris/Chris/bookshelf-unified/logs/sb3/Bookshelf-XArm7-Residual-Direct-v0/`
- July actor-only shadow export: `/home/chris/Chris/bookshelf-unified/data/policy_exports/bookshelf_residual_2026-07-08_shadow_actor.npz`, SHA-256 `75773dde0edabebcb525469c2e2b1cf868d7724f45a9f661f994cd8847a0ab19`

The paper aggregation manifest was generated at commit `8a76ebf7effcc889a36d9b0be5923eed383b37a5`; it reports successful integrity/audit checks, three residual policy seeds, and 2,000 scenarios per run. The fixed-3-mm scenario SHA-256 is `71282dd1c471ebbcf8c4145b6ee01b47af37b11f86f0686284fbec0111981f1b`.

Untracked paper figure outputs in `/home/chris/Chris/bookshelf-unified/paper/figures/`:

- `main_method_comparison.png` / `.pdf`
- `robustness_clearance_offset.png` / `.pdf`
- `training_curves.png` / `.pdf`
- `training_dynamics.png` / `.pdf`
- `execution_sequence.png` / `.pdf`

Their untracked generators are under `/home/chris/Chris/bookshelf-unified/scripts/paper/`. Because neither directory is committed, they must be preserved explicitly when merging this Alienware snapshot with the Riot snapshot.

## 6. Evidence cautions and missing information

1. **F0–F4 are not recoverable.** No manifest maps those labels to checkpoints, hashes, commands, or results. The requested fresh-policy F1–F4 table and rejection rationale remain a major paper-rewrite blocker.
2. **Named physical-like/combined/strong/actuation results are not recoverable.** The frozen clearance and offset sweeps are available, but renaming them would create unsupported provenance.
3. **The referenced 8,000-sample joint perturbation result is absent.** Only 80-case JSON artifacts survive in `/tmp`.
4. **No saved one-variable release-sensitivity ranking was found.** The code establishes clipping behavior, but not a numerical dimension ranking.
5. **July seed table is not independent.** Its 42/123/456 rows are numerically identical; prefer the later frozen scenario protocol.
6. **Current source is not exactly July source.** Use the July run's copied `params/env.yaml` and model/VecNormalize hashes for historical reproduction. The stale `experiment_spec.json` must not define the observation schema.
7. **Training success is not evaluation success.** The final 91.5% value is a rolling training window; the main frozen result is 90.93% mean across three policy seeds.
8. **Simulation is not hardware evidence.** Corrected Riot-derived input replay on Alienware is useful for diagnosis only. It does not validate frames, calibration, actuation, contact, latency, or real-robot safety.
9. **Volatile evidence needs preservation.** The four `/tmp` JSON files should be copied into a reviewed, provenance-rich result directory before reboot/cleanup, but this audit deliberately did not move or modify them.
10. **No current full regression was run.** Per instruction, this snapshot used static source/config inspection and existing result files only.

## 7. Recommended merge contract for the later paper rewrite

When the Riot snapshot is available, merge evidence by immutable identity:

- checkpoint path plus SHA-256;
- VecNormalize path plus SHA-256;
- Git commit and dirty/untracked state;
- scenario-bank hash and exact scenario count;
- raw 12-D observation definition and frame provenance;
- controller/release configuration active for each result;
- per-case outcomes, not only aggregate percentages;
- explicit separation of simulation, offline replay, shadow/diagnostic, and physical execution.

The paper's defensible current simulation claim is: a nominal controller plus a learned residual policy substantially outperformed nominal-only and PPO-only baselines on an identical frozen Isaac scenario bank, with strong performance across 2–5 mm clearance and up to 1.5× reset offsets. The paper should not claim that this establishes real-robot validity, nor attach F0–F4 or actuation labels until their original manifests are recovered from Riot or another archive.
