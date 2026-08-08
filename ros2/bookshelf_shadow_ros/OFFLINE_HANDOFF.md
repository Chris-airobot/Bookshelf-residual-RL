# Offline Global-to-Local Policy Handoff

The residual PPO policy is a local insertion controller. Isaac Lab placed it
near the pre-insertion region at reset, so simulation did not need a separate
global-motion handoff. The real system does.

The shadow inference node now publishes:

- `/bookshelf_shadow/policy_activation_ready`: `true` only after the complete
  activation test passes for the configured number of consecutive samples;
- `/bookshelf_shadow/policy_activation_debug`: the geometry, simulator-envelope
  outliers, normalized-observation outliers, and stability count;
- `/bookshelf_shadow/inference_valid`: `true` only when activation is ready and
  deterministic PPO inference succeeds.

No topic above authorizes robot execution. The shadow package has no action,
IK, trajectory, controller, gripper, or robot-command client.

## Activation Conditions

All conditions must pass:

1. observation, raw metrics, and validity messages are fresh and paired;
2. normalized observations remain below the global magnitude limit;
3. normalized observations lie inside the reviewed simulator-local envelope;
4. insertion depth, lateral error, vertical error, yaw error, gripper state,
   and mode lie inside configured local-policy limits;
5. the result remains acceptable for `activation_stable_samples` consecutive
   inference cycles.

Actor saturation is intentionally diagnostic only. The trained actor can
saturate in valid simulator states, so saturation must not decide activation.

## Offline Order

1. Generate an activation envelope from the saved simulator equivalence
   samples.
2. Check the independent simulator pre-insertion batch and repeat one
   representative valid sample through the stability tracker.
3. Run unit tests and build the shadow package.
4. Run the existing far-pose audit with `--expect-activation blocked`.
5. Later, record a stationary physical pre-insertion pose and run the same
   check with `--expect-activation ready`.
6. Only after that, run MoveIt plan-only checks with a complete collision scene.

The current recorded pose is expected to remain blocked because its insertion
depth, lateral, vertical, yaw, and normalized observations are outside the
local region. A clean blocked result is the correct behavior.

## Remaining Physical Evidence

Offline work cannot establish:

- whether the traditional planner can reach the intended physical
  pre-insertion pose;
- collision clearance for the real shelf, neighboring books, camera, and
  gripper;
- contact behavior, force safety, or insertion success;
- whether the activation limits require adjustment after a real stationary
  pre-insertion recording.

Keep execution disabled until those checks have been completed and reviewed.

## Automatic Experiment Logging

Every physical trial must start `experiment_logging.launch.py`. It creates a
new timestamped directory, records compact policy/control/TF topics plus
compressed RGB-D streams, and writes `manifest.json`, `events.jsonl`, and
`ros_graph.json`. The manifest includes the Git state and SHA256 hashes of the
policy bundle and activation envelope. The launch refuses to start below its
configured free-space threshold and contains no robot-command node.
