# July simulation versus real PUSH termination — 2026-09-07

Mismatch confirmed. Real PUSH stopped after 30 mm of modeled book movement,
although July's task requires a final geometric seating condition. The real
completion gate now checks measured geometry. No hardware or training was run;
no policy, INSERT, release trigger, slot detection, or calibration was changed.

## Provenance and simulation trace

The original July run was located read-only on `alienware` at
`/home/chris/BookshelfFiles/training_runs/sb3/Bookshelf-Residual-Direct-v0/2026-07-08_13-14-04/`.
Its `model.zip` SHA256 is
`80f7aa2d6675a99f3965b2479bc0b62f5f3320e724a6f3399efacc1640b3b4ed`,
matching the deployed July actor's checkpoint metadata. Its saved
`params/env.yaml` SHA256 is
`9a342fa44f9b6314a6062dce9166022f36f00e6af870fa4ed3504dd8640f2c88`.
The saved `experiment_spec.json` contains an incorrect generic 16D observation
description and null source blocks; it was not used as implementation evidence.
The actual saved env configuration specifies 12 observations and 6 actions.

Repository paths below are relative to the repository root:

- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_residual_env.py`:
  `BookshelfEnv` inherits V5; `_get_dones` delegates to V5, apart from optional
  debug overrides. `_apply_action` and `_nominal_cartesian_delta` implement the
  nominal-plus-residual motion in both INSERT and PUSH.
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_v5.py`:
  `_get_dones` controls release/script/PUSH transitions and geometric success;
  `_compute_task_metrics` derives rear/front/corner extent from simulated book
  ground truth. V4 supplies `_upright_ok` and observation mode encoding.
- `source/bookshelf/bookshelf/tasks/direct/bookshelf/bookshelf_env_cfg_v4.py`,
  `bookshelf_env_cfg_v5.py`, and `bookshelf_residual_env_cfg.py`: inherited
  thresholds, scales, nominal controller and script parameters.
- `scripts/sb3/july_corrected_takeover_closed_loop.py:main` evaluates July with
  the current environment. It disables reset curricula and nuisance terms, not
  the task success gate. Its 77/80 correct releases are not 77/80 final successes.

There is a historical distinction: July's saved config includes upper/lower
seating bounds removed by commit `1a5586e` (August 23). The V5 implementation at
`1a5586e^` explicitly evaluates both bounds. Current evaluation uses monotonic
depth checks without the upper rear/lower front bounds. The exact July runtime
source snapshot was not stored in its experiment spec; the saved July config
and pre-change source agree on the bounded condition. Both versions reject the
September 7 old endpoint. Real completion conservatively uses the stricter
original July bounds, stopping on overshoot instead of pushing farther.

Simulation flow:

```text
INSERT + release action > 0.5
  → SCRIPTED (internal mode 1, observation mode 0.5)
  → open 3 steps → retreat 6 steps × −15 mm target offset → close 5 steps
  → PUSH (internal mode 2, observation mode 1.0)
  → same July actor + PUSH nominal controller on each step
  → success geometry held 4 steps, OR drop/failure, OR episode timeout
```

Physics dt = 1/120 s; decimation = 2; policy/control step = 1/60 s;
episode_length_s = 10 (600 steps, timeout at length >=599). Script length is
14 control steps. The −90 mm retreat is a sum of relative target offsets,
not a guarantee of exactly 90 mm measured simulated travel. There is no fixed
PUSH book travel limit or separate successful 30 mm endpoint in the task.

PUSH observation remains
`[mode, rear, front_clearance, lateral, Z, yaw, tool_book_X/Y/Z, gripper, tilt1/2]`.
Scales are `[1,.08,.08,.05,.05,pi/6,.25,.25,.25,1,1,1]`, clipped to the
observation range, then VecNormalize. Physical gripper opening is observed in
PUSH, with empty gripper commanded closed. The sixth action still exists but
cannot request another release in PUSH. Five motion actions remain active.

For clipped action `a`, residual scales are
`[.002,.001,.0015,radians(.35),radians(.30)]`.
`delta = clip(nominal_PUSH + scaled_residual, final_limits)`.
Nominal PUSH has dx=.0008 m, lateral/height/yaw/pitch gains .35/.30/.20/.08;
its height target is 20% up from the book's lowest extent. This is residual RL,
not a policy that stops influencing motion when PUSH begins.

## Exact success gates

Let corner positions be expressed in slot axes; rear `r=min(corner_x)`, front
penetration `d=max(corner_x)`, front clearance `f=slot_depth-d`, lateral extent
`L=max(abs(corner_y))`, book-center height `z`, and depth-axis yaw `psi`.
Every gate below must hold in PUSH on four consecutive steps/samples:

| Gate | July training | Current evaluation | New real |
| --- | --- | --- | --- |
| Rear | −12 mm <= r <= +2 mm | r >= −12 mm | July band |
| Front clearance | −3.2 mm <= f <=55.2 mm | f <=55.2 mm | July band |
| Lateral corners | L <= half gap +1.5 mm | Same | accepted slot width/2 +1.5 mm |
| Center Z | abs(z) <15 mm | Same | Same in accepted slot axes |
| Yaw | abs(psi) <8 degrees | Same | Same |
| Upright | abs(book height-axis vertical dot) >.85 | Same | Same; real local height axis is Z |
| Velocity | disabled (both limits =0) | Same | No invented velocity gate |
| Hold | 4 consecutive simulation steps | Same | 4 unique fresh TF timestamps |
| Ready | episode step >5 | Same | already beyond INSERT + release/retreat |

Simulation book local dimensions are (depth,height,thickness) =
(156,236,34) mm; ROS uses (depth,thickness,height) = (156,34,236) mm.
Its geometric mouth is derived from neighboring book front faces, not simply
`slot_x_open`: with x_open=.63, x_back=.83 and 156 mm neighbors, mouth=.652 m,
so geometric slot depth=.178 m. Real uses its accepted mouth and configured
.200 m depth. Consequently the front-depth floor is .1228 m in that Isaac
geometry and .1448 m in real. The rear threshold often dominates both.
This physical-geometry substitution is explicit, not a hidden constant change.

No standalone tilt-X/Y success limits exist beyond upright and corner extent.
The original training also terminates on unsupported floor drop, with
`book_floor_lowest_z_thresh=.042`; optional OOB/fell termination is disabled.
Newer code refines drop handling by mode. Neither failure rule makes 30 mm
of PUSH a success.

## Real old flow and safety

Implementation: `ros2/bookshelf_simple_experiment_ros/bookshelf_simple_experiment_ros/`
`simple_policy_control_node.py`, `_gripper_goal_result`, `_try_calculate_push`,
`_push_servo_tick`; `post_insert_math.py` provides contact/progress math.

```text
release → gripper open confirmed → measured 90 mm retreat
  → empty gripper close confirmed → PUSH
  → July action + same nominal/residual target calculation → MoveIt Servo
  → modeled book advance reaches 30 mm → push_complete/episode_complete
```

Contact gap is a point-to-oriented-box supporting-plane projection:
`gap = direction·(release_book_center − current_TCP) − sum(abs(R_book.T·direction)*half_size)`.
At gap <=1 mm, contact travel is latched as `max(0,EEF_progress+gap)`.
Old inferred book advance was `clip(EEF_progress-contact_travel,0,.03)`.
It translated the frozen release book pose, and the policy observed that
estimate. It was not a force/contact sensor. `push_x_uncertainty_m=.005` was
logged but not added to this formula. **The old implementation did not evaluate
simulation success and had no explicit maximum PUSH travel guard.**

July continued producing actions at 20 Hz; target updates affect the bounded
30 Hz Servo twist. Existing limits: 25 mm/s linear, .10 rad/s angular,
.2 s command horizon, .5 mm translation tolerance, .25 degree rotation
tolerance, 90 s PUSH timeout; Servo codes 2/4/5 halt, codes 1/6 decelerate.
Previously failed TF lookup returned before checking timeout, potentially
bypassing that timeout while state remained missing. The new order fixes this
PUSH-specific gap.

## September 7 counterfactual

The clean release state's all-corner depth is 102.228893 mm; rear −60.808197 mm;
front clearance 97.771107 mm. A forward translation x must satisfy:

```text
x >= max(−12 − (−60.808197), 97.771107 −55.2, 0) mm
  = max(48.808197,42.571107,0) =48.808197 mm
```

At fixed release orientation, depth would be >=151.037090 mm. July's upper
rear band also implies x <=62.808197 mm. The old 30 mm translation leaves
rear −30.808197 mm, front clearance 67.771107 mm: both depth checks FAIL.
The depth-only additional book advance is **18.808197 mm**. Including old
latched contact travel (21.772181 mm), the estimated minimum total forward
approach+push is **70.580377 mm**, versus old 51.772181 mm.

Independent marker at the old endpoint gives rear −28.290080 mm, depth
134.811015 mm, front clearance65.188985 mm: at least **16.290080 mm** further
translation for the depth checks at that orientation. The fixed 30 mm cutoff
quantitatively explains remaining 28–31 mm protrusion; July permits up to12 mm
protrusion at success, so the shortfall relative to trained success is about
16–19 mm, not necessarily another full 28–31 mm.

This is not a prediction that straight pushing will succeed. At the release
orientation lateral corner extent is20.447423 mm, versus20.281109 mm allowed
by the actual accepted width37.562218 mm plus epsilon. The nearest final
marker has extent27.244605 mm. Forward translation alone cannot fix these
lateral/orientation gates; the actor may need to align, or safety limits may
stop the attempt. Marker noise, common calibration errors, contact/slip and
unobserved future dynamics limit the counterfactual. No missing physical
measurements or successful continued episode are invented.

## Implemented real completion and limits

`push_completion.py` contains the July gate and a fresh-sample monitor.
`_push_servo_tick` now requires measured `base→target_book_center` TF (<=250 ms
old), fresh EEF/TCP transforms, fresh joint input and Servo status. It checks
the geometry in the frozen accepted slot; duplicates cannot advance the
four-sample hold. Missing/stale/future/regressing TF, missing state or depth
overshoot stops with `SAFETY_STOP`. There is no blind extrapolated success.

The actor's PUSH observation construction, network, nominal action, scales,
mode and target math are preserved. Its existing geometric book estimate
continues beyond30 mm, capped only by the safety budget. It is explicitly
separate from the measured success observer. This avoids silently replacing
the actor's observation source/controller as part of a termination fix. That
estimate can still differ from actual book pose after slip or rotation: full
sim2real state parity is not claimed. Measured success plus bounded motion is
the closest available completion equivalent within this scoped change.

While all gates first hold, twist is zero during confirmation, avoiding
overshoot while four distinct measurements arrive. This conservative terminal
hold is not simulation-identical timing: simulation checks four60 Hz physics
steps while continuing control; real verifies four sensor samples.

An explicit **100 mm cumulative measured TCP path** guard (including approach,
vertical/lateral movement and reversal) now stops with `MAX_TRAVEL`, never
success. September7's old measured TCP path was61.84 mm; about19 mm additional
straight travel would fit this conservative budget, but future alignment may
consume it. This is a safety cap, not an arbitrary replacement success travel.
It is not a certified hardware stopping distance: TF sampling/Servo latency
and physical deceleration remain. The90 s timeout is checked before state
lookup; invalid limits fail closed. Existing Servo protections remain active.

Logs at PUSH start and every PUSH policy update include measured depth,
criterion bands, remaining depth, geometry gates, source/timestamp, unique
success count, actual TCP axial/cumulative progress, and the existing raw/
clipped PPO/nominal/final actions. Terminal reasons are `SUCCESS_CRITERION`,
`MAX_TRAVEL`, `TIMEOUT`, or `SAFETY_STOP`. Legacy30 mm remains only a labeled
shadow/config diagnostic; it no longer defines success. Failures do not emit
`push_complete`.

## Validation

- Focused PUSH/rollout/post-insert tests:29 passed.
- Full simple-experiment suite:124 passed,2 skipped (optional external fixtures),
  one existing hppfcl deprecation warning. Tests instantiate no ROS nodes.
- Tests cover old30 mm rejection using September7's actual4×4 transform,
  all alignment gates, four unique samples, stale/duplicate/regressed/future
  timestamps, overshoot, continued nonzero motion, success stop, path budget,
  timeout before unavailable TF and fatal/stale Servo/joint state.
- `PATH=/usr/bin:/bin:$PATH scripts/ros2/build_xarm_experiment.sh`: colcon
  completed all five packages successfully in33.56 s. Build log:
  `.ros2_ws/log/build_2026-09-07_11-13-39/events.log`.
- No hardware launch, simulator execution or retraining was performed.

Files changed for this task: `simple_policy_control_node.py`, new
`push_completion.py`, `config/simple_policy_control.yaml`, new
`test/test_push_completion.py`, `test/test_rollout_control.py`, and this report.
Other dirty simulation/training/slot-detector files were pre-existing and were
not edited for this task. July retraining is not justified by this mismatch.
