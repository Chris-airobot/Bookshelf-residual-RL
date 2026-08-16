# Riot Codex Handoff - Supervised Bookshelf Experiment - 2026-08-16

Read this file first, then read:

1. `REAL_ROBOT_EXPERIMENT_COMMANDS_2026-08-16.md`
2. `REAL_ROBOT_EXPERIMENT_RUNBOOK.md`

The dated command file is the authoritative operator sequence. Do not substitute
older commands from chat history or another checkout.

## Repository Identity

- Canonical Riot source checkout: `/home/riot/Chris/bookshelf-unified`
- Required branch: `combined/bookshelf-20260808`
- Required state: local `HEAD` must equal `origin/combined/bookshelf-20260808`
- Unified ROS install: `/home/riot/Chris/bookshelf_unified_ws/install`
- Depth-fix underlay: `/home/riot/Chris/ros2_ws/install_depth_fix`

The folders under `bookshelf_unified_ws` are build/install artifacts, not extra
source repositories. Make source changes only in the canonical checkout.

Known Git LFS model files can appear modified. Do not stage, restore, or rewrite
them during experiment preparation.

## Current Research State

The software-side frozen simulation evaluation is complete:

- 2,000 scenarios per condition;
- three independent training seeds;
- fixed 3 mm success: nominal 28.85%, PPO-only 0%, residual PPO 90.93 +/- 2.80%;
- clearance sweep: 1-5 mm;
- initial-offset sweep: 0-1.5 times the final training randomization;
- residual PPO remains 81.20 +/- 2.10% at 1.5 times OOD offsets;
- every frozen replay audit passed.

The paper simulation artifacts are frozen on the Alienware. The immediate Riot
objective is real-robot evidence, not more simulator evaluation.

## Physical Experiment Objective

Perform a supervised book insertion sequence:

1. observe the robot, camera, shelf, slot, and planning scene;
2. verify the frozen slot against live RGB-D perception;
3. attach the reviewed book grasp;
4. generate a fresh collision-aware global MoveIt plan to pre-insertion;
5. review the entire path in RViz against the physical setup;
6. execute at most one approved global trajectory;
7. recheck perception and run residual policy shadow mode;
8. review and approve at most one guarded local residual step at a time;
9. preserve automatic logs for successes and failures.

People must remain beside the robot with immediate stop access for every
movement. "One-shot" means one trajectory submission per executor process. It
does not mean unattended operation.

## Fixed Riot Paths

The current reviewed inputs are:

```text
Repository:
  /home/riot/Chris/bookshelf-unified

Trial slot configuration:
  /home/riot/BookshelfFiles/experiment_logs/environment_checks/physical_trial_20260813_01/trial_static_slot.yaml

Physical planning scene:
  /home/riot/BookshelfFiles/experiment_configs/physical_trial_20260813_01_bookshelf_scene.yaml

Shadow actor bundle:
  /home/riot/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz

Activation envelope:
  /home/riot/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json
```

Use a new `TRIAL_NAME` for every attempt. Never reuse a trajectory after the
book, robot, scene, target, or joint state changes.

## Verified Software Capabilities

- `physical_experiment_observation_bringup.launch.py` combines robot, camera,
  TF, MoveIt, automatic logging, slot detection, frozen-slot checking, calibrated
  held-book marker detection, and a live-versus-configured book-pose gate.
- The global planning scene remains unavailable until the marker-derived
  `T_link_tcp_book` agrees with the frozen attached-book transform for 30 stable
  samples. The gate is read-only and never rewrites the scene YAML.
- Observation bringup starts no trajectory executor.
- The calibrated pre-insertion bridge performs seeded collision-aware IK and
  MoveIt planning only.
- The global executor is fail-closed by default and can submit at most one
  trajectory per process after all external approval gates pass.
- The experiment logger is automatic and subscriber-only.
- The residual insertion path is supervised and stepwise; no loop automatically
  chains physical residual actions.

Offline tests and plan-only success are not physical execution authorization.

## Required Source Order in Every Riot Terminal

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_unified_ws/install/local_setup.bash
source /tmp/bookshelf_trial_env.sh
```

Over SSH, use `show_rviz:=false`. RViz review must be performed from the Riot
desktop with `show_rviz:=true` or an equivalent desktop-side RViz session.

Do not assume robot, camera, TF, MoveIt, or logging is already running. When
giving experiment commands, include the observation/hardware bringup command.

## Mandatory Stop Boundaries

Do not execute motion when any of the following is true:

- the live frozen-slot check is not `true`;
- `/compute_ik`, `/plan_kinematic_path`, or `/apply_planning_scene` is missing;
- duplicate or unexpected executor nodes exist;
- the book is absent, loose, or differs from the reviewed grasp;
- the shelf, table, held-book box, target, or path disagrees with reality;
- the plan report is invalid, IK code is not `1`, trajectory sanity fails, or
  the trajectory hash is absent;
- the full animated RViz path has not been reviewed by a person at the robot;
- any transform, scene, token, provenance, joint-drift, or approval gate fails.

Never weaken a failed gate to make the experiment proceed.

## What Codex Should Do First on Riot

1. Verify branch, commit parity with origin, and targeted repository changes.
2. Rebuild `bookshelf_policy_ros`, `bookshelf_shadow_ros`, and
   `bookshelf_guarded_control_ros` into the unified install.
3. Verify all three package prefixes resolve to the unified install.
4. Follow `REAL_ROBOT_EXPERIMENT_COMMANDS_2026-08-16.md` from Section 1.
5. Stop after plan-only inspection until the human physical/RViz review passes.

Do not launch a guarded executor merely because the user says the system is
"ready." Show the plan-only evidence, identify the human review boundary, and
wait for explicit approval for that single movement.

## Evidence to Preserve

For every trial preserve:

- automatic experiment directory and rosbag;
- manifest and ROS graph;
- static-slot status and comparison;
- calibrated target report;
- IK and plan-only report;
- trajectory SHA256;
- external reviewed executor configuration hash, without publishing its token;
- execution report;
- policy activation/shadow reports;
- outcome, duration, intervention, collision/drop, and failure reason.

Failures are experimental evidence. Do not delete or overwrite them.
