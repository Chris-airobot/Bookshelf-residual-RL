# Real PUSH freshness abort and recovery — September 7, 2026

## Result

MARKER_FAILURE_CAUSE: MARKER_TEMPORARY_LATENCY — specifically a temporary
camera/update interruption, not proof of physical marker occlusion.
RECOVERY_BUG_CONFIRMED: YES. The safety stop was correct; the operator failed to
recognize its terminal outcome. Recovery-only fix implemented. No hardware,
training, policy, INSERT, release, perception, calibration, or safety-limit change.

## Inputs and coverage

- Latest policy log automatically selected from `simple_policy_*/policy_step.jsonl`:
  `/home/riot/BookshelfFiles/experiment_logs/simple_policy_20260907_112636/policy_step.jsonl`.
- Latest bag: `/home/riot/BookshelfFiles/experiment_logs/full_real_bags/full_real_20260907_112642/full_real_20260907_112642_0.db3`.
- Bag timestamps: 1788744403.047663–1788744886.025849, September 7,
  11:26:43.048–11:34:46.026 Melbourne (UTC+10). The failure is fully covered,
  with about 10.60 seconds afterward, ending in controller shutdown.
- Release requested 11:34:26.325944, open complete 26.985724;
  retreat complete 33.787472; empty gripper closed 34.761688;
  PUSH start 34.762448; PUSH stopped 35.420871.
- Do not use the `failed` policy row's inherited last-INSERT timestamp as the
  abort time. `push_stopped`, `rollout_complete`, bag statuses and rosout agree.

## Marker/camera measurements

Both `target_book_marker` and `target_book_center` have the same unique update
timestamps. Counting unique stamps, not repeated lookup calls:

| Window (Unix seconds minus 1788744800) | Unique samples | Effective rate* | Median / p95 / max unique gap | Received age median / p95 / max |
| --- | ---: | ---: | --- | --- |
| 20–66.326, pre-release | 680 | 14.70 Hz | 33.36 / 233.50 / 700.51 ms | 38.45 / 43.82 / 55.76 ms |
| 66.326–74.762, release/retreat/close | 233 | 27.59 Hz | 33.36 / 66.72 / 200.15 ms | 38.12 / 41.21 / 46.71 ms |
| 74.762–75.422, PUSH approach | 9 | 19.82 Hz | 33.36 / 88.40 / 100.07 ms** | 42.92 / 45.26 / 45.65 ms |
| 75.422–bag end, after abort | 174 | 17.10 Hz | 33.36 / 133.43 / 300.22 ms | 41.45 / 49.47 / 65.23 ms |

*Rate is (N−1)/(last receipt−first receipt), so does not include the censored
gap at a window end. **The abort-causing gap crosses the PUSH-window boundary
and must be examined separately; this row does not imply uninterrupted PUSH.

Exact abort-causing sequence:

- Last unique marker stamp: 1788744875.146956299; bag arrival 75.192610.
- Failure at approximately 75.4209: age 273.9 ms, correctly exceeding 250 ms.
- Next unique stamp: 75.8474658; arrival 75.8880787.
- Unique measurement gap: **700.51 ms**; arrival gap: **695.47 ms**.
- Marker became stale at 75.396956 and available fresh again at 75.888079:
  approximately **491.1 ms stale**, resuming approximately **466–467 ms after abort**.
- Camera color image stream independently has last pre-gap stamp 75.180314,
  receipt 75.241177; next stamp 75.847466, receipt 75.884268. Thus camera
  capture-stamp gap is **667.15 ms**, recorded arrival gap **643.09 ms**.
  Camera-info shows the same stamp gap and a 654.06 ms receipt gap.
- Received marker messages normally arrive only 35–65 ms old. Sampling the
  latest available marker at 1 ms intervals through PUSH gives held-state ages
  min/median/p95/max approximately 41.9/77.0/240.1/273.0 ms. Received-message
  age alone would hide this outage.

This was not merely one marker packet delayed by 24 ms. The image/marker stream
temporarily lost fresh updates together. The bag cannot distinguish camera
driver, USB, scheduling, transport or recording loss as the underlying cause;
the live controller independently confirms its TF freshness outage. There is
no evidence sufficient to attribute this outage to progressively deeper shelf
occlusion. Tracking recovers while the arm remains stationary. Pre-release
tracking was also intermittent: 25 unique gaps above 250 ms in the inspected
46.3 seconds (maximum 700.51 ms). The outage is isolated within this very short
PUSH, but the whole trial's tracking is not uniformly healthy.

At abort TCP axial travel was only **2.105 mm**, cumulative path **2.783 mm**.
This is early contact approach, not evidence of a deep pushing occlusion.
Last accepted measured book depth was 102.658 mm, rear −56.712 mm, remaining
depth at least 44.712 mm; lateral clearance also failed. No seating success was
reported (zero success samples). These are last-valid measurements, not a fresh
pose at abort; the diagnostic source label `fresh_book_tf` describes their
original acquisition, not continued freshness after the stop.

## Existing abort and controller trace

In `bookshelf_simple_experiment_ros/simple_policy_control_node.py`:

1. `_push_servo_tick` → `_push_fresh_transform` rejects age >0.25 s.
2. `_halt_and_fail` publishes zero Twist and `push_stopped/SAFETY_STOP` log.
3. `_fail` publishes generic `failed`; `_terminate_rollout(error)` writes the
   terminal rollout record and publishes `failed` again.
4. `_begin_visualization_hold` sets `holding_visualization`. `_timer_callback`
   has no motion branch there; `_continuous_policy_tick` returns immediately.
   The July actor no longer controls motion. The Servo process stays alive,
   but receives no new motion commands. The Servo launch's incoming-command
   timeout remains 0.2 s.

Recorded evidence: final nonzero Twist at 75.392375, zero Twist at 75.421164;
no later Twist commands in the bag. Generic failed statuses appear at
75.421557 and 75.422308. Servo trajectory output finishes at 75.600744.
Measured joint positions at 75.583747 and 76.983835 are exactly identical.
Thus this is not an endlessly moving or competing policy controller.

`Policy visualization held for indefinitely` is expected visualization-only
retention. Cached rollout/target display may remain visible, but those caches
are not an active planning/motion lock. A later I resets the completed episode.

In `operator_console_node.py`, the old `OperatorWorkflow.policy_status` handled
only `episode_complete`, not `failed`. The subscriber decoded the recorded
failure messages but had no state transition for them. Therefore state stayed
`policy_running`; H required `push_complete_waiting_return`, G required `start`
or `ready_for_next_book`, and E required a reviewed plan. All were gated out.
The bag does not record a later H/G service request or planning failure and
does not cover attempts after shutdown; the exact later keystrokes cannot be
reconstructed. The reported stuck screen and deterministic gating agree.

## Scan/return prerequisites and diagnosis

Primary cause: **A, state-machine gating**, not demonstrated B planning failure
or C Servo ownership. E, no return to loading pose, is a consequence of H being
blocked, not a required hard-coded starting-joint test preventing H itself.
`preinsert_node._plan_return_loading_callback` calls `_plan_joint_pose` using
current fresh joints, busy-state checks and MoveIt validity/planning checks.
It does not require the robot to already be at the return destination.

Normal return: H plans → review → E executes → trajectory `done` →
`finish_return` requests gripper opening → operator-action `ready` → G available.
Normal scan: G only plans scan → review → E executes → `done` enters `scan` →
S freezes an available slot. `_clear_trial_state_for_scan` removes old frozen
slot, target and plan state. D, stale slot, does not explain the policy-running
gate. Slot detector logs `no valid slot candidate` after 79.097 are real, but
occur without returning to the scan/view pose, and do not themselves prove a
perception fault. No scan planning attempt is demonstrated after the abort.

## Small recovery-only fix

- `_halt_and_fail` captures whether failure occurred in PUSH. After zero command
  and terminal rollout handling it publishes explicit `push_aborted` with the
  safety/timeout/travel reason and `PUSH ABORT ... RETURN AVAILABLE` log.
- Operator adds `push_abort_waiting_return`, distinct from success. This reuses
  the existing H planning, E confirmation, return execution and gripper-open
  sequence. No automatic arm movement or PUSH restart occurs.
- H is accepted in that abort state. Failed return planning preserves it for
  retry. A delayed start-policy service response cannot overwrite the terminal
  abort state with `policy_running`.
- Generic failures (including INSERT) do NOT authorize released-book return.
  Normal `episode_complete` behavior remains unchanged.
- Successful return/open logs `READY_FOR_NEW_SCAN`; G still only plans.

Changed this task: `simple_policy_control_node.py`, `operator_console_node.py`,
`test/test_push_completion.py`, `test/test_operator_console.py`, and this audit.
All other dirty files include prior work and were preserved.

## Freshness decision

**Keep 0.25 s unchanged.** It is comfortably above healthy received ages but
correctly stops motion through a roughly 0.7 s sensing outage. Raising it to
0.30 s would simply abort later in this same gap. Allowing the full gap would
authorize blind motion and is not justified by this evidence. Investigate
camera stream continuity before another PUSH trial. A future admission check
could require several recent unique frames with bounded inter-frame gaps,
while retaining the in-motion freshness stop; that logic is not added here.
Four fresh unique success samples remain mandatory. No automatic reacquisition
and resume is implemented.

## Offline validation

With `/opt/ros/humble/setup.bash` and `.ros2_ws/install/local_setup.bash` sourced,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q`:

- Focused `test_push_completion.py`, `test_operator_console.py`,
  `test_rollout_control.py`: **36 passed**.
- Entire simple-experiment package tests: **128 passed, 2 skipped** (optional
  fixtures), one pre-existing hppfcl deprecation warning.
- Tests cover zero-before-abort status, terminal timer inactivity, stale pose
  never counting as success, H/reviewed E return, fresh G/E/S after return,
  delayed service response, return-plan retry, unchanged normal success,
  maximum travel and timeout. No ROS nodes or hardware are launched by these tests.
- Initial test invocations had incomplete ROS/Python paths; corrected by
  sourcing the canonical overlay. These were collection errors, not test failures.
- `scripts/ros2/build_xarm_experiment.sh`: **5 packages built successfully**;
  `.ros2_ws/log/build_2026-09-07_11-42-43/`; installed launch postchecks passed.
- `git diff --check`: passed.

Raw analysis artifacts and extraction scripts are under `/tmp/push_abort_*`.
No rosbag playback into live ROS, hardware startup or training was performed.

## Next hardware session (operator controlled)

Use the rebuilt overlay/new processes. If PUSH aborts, do not resume it or
bypass marker safety. Inspect the physical book/robot situation, then H to
plan return; review the trajectory and E to execute only if safe. Once return
and gripper opening finish and READY_FOR_NEW_SCAN appears, G plans the scan,
E executes the reviewed scan trajectory, and S freezes a fresh slot. A failed
plan still requires investigation/replanning, never forced execution. This
patch cannot retrospectively update an already-running old operator process.
