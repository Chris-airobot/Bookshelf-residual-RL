# Real PUSH transient book-state recovery — September 7, 2026

## Latest run evidence

- Policy log: `/home/riot/BookshelfFiles/experiment_logs/simple_policy_20260907_114750/policy_step.jsonl`
- Full bag: `/home/riot/BookshelfFiles/experiment_logs/full_real_bags/full_real_20260907_114820/full_real_20260907_114820_0.db3`
- Bag interval: Unix 1788745701.309844–1788745846.411976. It cleanly
  includes PUSH, the failure at 1788745825.272751, recovery of the streams,
  and roughly 21 seconds afterward.
- PUSH began at 1788745824.386909. It stopped 0.886 seconds later with
  `target_book_center ... age=0.282s`. Cumulative TCP path was 3.464 mm and
  axial progress 2.642 mm.

The last accepted `target_book_center` measurement had stamp
1788745824.990161 and arrived at 1788745825.050635. The next unique measurement
had stamp 1788745825.231761 and arrived at 1788745825.287249:

- unique marker stamp gap: **241.600 ms**;
- marker arrival gap: **236.614 ms**;
- fresh marker returned **14.498 ms after the old controller aborted**;
- the old stamp crossed the 250 ms freshness boundary at 1788745825.240161,
  so the unavailable-as-fresh interval was approximately **47.088 ms**.

The color stream shows exactly the same frame stamps. Its corresponding
arrival gap was 246.964 ms, and its first recovered frame arrived 13.658 ms
after the abort. This confirms another temporary camera/frame-stream
interruption. It does not establish physical marker occlusion.

The previous audited outage had a 700.51 ms marker-stamp gap and fresh tracking
returned about 466 ms after abort. Starting at the moment that old state first
became stale, three recovered unique frames would have been available in about
556 ms. In this latest run they would have been available in about 201 ms.
These two recorded outages justify a **1.0 second** bounded grace period: it
covers both observed transients plus margin, but remains tiny relative to the
existing 90 second overall PUSH timeout.

## Implemented behavior

Individual measurement freshness remains exactly **0.25 seconds**.

When only the independent PUSH book TF becomes stale/missing:

1. Publish zero Twist immediately.
2. Enter `push_waiting_for_fresh_state`; stop policy updates and discard the
   prior `target_eef`, so it cannot execute after recovery.
3. Keep publishing zero Twist at the 30 Hz control rate.
4. Continue validating joint state, Servo status, EEF/TCP TF, the original
   overall timeout, and cumulative TCP path. Any non-book safety fault aborts.
5. Observe book TF for at most 1.0 second. Require three fresh, unique,
   monotonically increasing timestamps. Cached duplicates do not count;
   renewed loss resets the recovery count; timestamp regression aborts.
6. Recovery frames prove stream continuity only. They never count toward the
   four independent success samples. The seating-success hold restarts at zero.
7. Resume the same PUSH phase and rollout. A new July target must be calculated
   before motion can resume. PUSH origin, contact estimate, start time, measured
   TCP path and progress are not reset or double-counted.
8. If grace expires, use the existing `push_aborted` →
   `push_abort_waiting_return` → operator-controlled H/review/E return path.

Maximum cumulative TCP travel remains 0.10 m, overall PUSH timeout remains
90 seconds, and Servo protections are unchanged. Stale geometry cannot produce
success. No automatic retry motion or blind dead reckoning was added.

The grace is observation time, not a relaxed freshness threshold. A recovered
sample must independently be no more than 250 ms old.

## Changes and validation

Changed for this task:

- `bookshelf_simple_experiment_ros/simple_policy_control_node.py`
- `config/simple_policy_control.yaml`
- `test/test_push_completion.py`
- this audit

The earlier recovery changes in `operator_console_node.py` and
`test/test_operator_console.py` remain intact and were retested.

Tests cover immediate zero motion, three unique-frame recovery, duplicate
rejection, no success during waiting, cumulative travel across recovery,
grace expiry through the existing abort path, abort/H/E recovery, normal PUSH,
90 s timeout and 100 mm maximum travel.

- Focused tests: **38 passed**.
- Full package pytest: **130 passed, 2 skipped** (optional fixtures), with one
  existing `hppfcl` deprecation warning.
- `colcon test --packages-select bookshelf_simple_experiment_ros`:
  **130 passed, 2 skipped**.
- `scripts/ros2/build_xarm_experiment.sh`: **5 packages succeeded**;
  installed launch checks passed.
- `git diff --check`: passed.

No hardware, training, policy change, bag playback into ROS, freshness change,
or safety-limit weakening was performed.
