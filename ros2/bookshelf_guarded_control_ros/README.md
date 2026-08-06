# Bookshelf Guarded Control ROS

This package is the deployment boundary between the read-only bookshelf shadow
pipeline and MoveIt. It does not contain a policy network. It consumes the
already checked nominal-plus-residual delta and makes the real xArm
`link_tcp` move as required for the equivalent simulator policy-tool frame.

```text
/bookshelf_shadow/final_delta
  + live validity and provenance
  + T_base_slot
  + T_base_link_tcp
  + T_link_tcp_virtual_policy_tool
        |
        v
target virtual policy-tool pose in slot coordinates
        |
        v
target link_tcp pose in link_base
        |
        v
MoveIt collision-checked path
```

## Two intentionally separate processes

`policy_tool_plan_checker` has no trajectory execution action client. It asks
MoveIt for a path and publishes the target poses, path, and JSON report. The
committed checker accepts the derived, physically unverified tool transform for
inspection but keeps `execution_ready=false` while the planning scene is not
confirmed complete.

`guarded_policy_tool_executor` is the only process in this package capable of
owning an execution action client. It creates that client only when every
static startup gate is already open. Its committed configuration therefore
creates no command interface because all of these gates start closed:

- `allow_unverified_policy_tool: false`
- `planning_scene_complete: false`
- `dry_run: true`
- `allow_execution: false`
- `approval_token: DISABLED`

It has no gripper interface. A valid plan is short-lived, current joints must
still match its start state, known prototype executor nodes must be absent, and
one exact approval token authorizes at most one trajectory.

## Important planning-scene boundary

MoveIt can only collision-check geometry present in its active planning scene.
The checker reports a path while `planning_scene_complete=false`, but that path
does not authorize execution. Before changing the flag, the scene must contain
the shelf shell, neighboring books, held book, robot, camera/fixture geometry,
and any nearby lab obstacles with checked frame IDs and dimensions.

## Build and test

Run these on the target machine. The package depends on the installed
`bookshelf_shadow_ros` package and the xArm MoveIt installation.

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_shadow_ws/install_standard/setup.bash

cd /home/riot/Chris/Bookshelf-residual-RL

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=$PWD/ros2/bookshelf_guarded_control_ros \
python3 -m pytest \
  ros2/bookshelf_guarded_control_ros/test \
  -q

colcon \
  --log-base /home/riot/Chris/bookshelf_guarded_control_ws/log \
  build \
  --base-paths ros2/bookshelf_guarded_control_ros \
  --build-base /home/riot/Chris/bookshelf_guarded_control_ws/build \
  --install-base /home/riot/Chris/bookshelf_guarded_control_ws/install \
  --packages-select bookshelf_guarded_control_ros
```

## Plan-only launch

Start the robot, MoveIt, and the calibrated candidate shadow pipeline first.
Then source this package and launch only the checker:

```bash
source /opt/ros/humble/setup.bash
source /home/riot/Chris/ros2_ws/install_depth_fix/setup.bash
source /home/riot/Chris/bookshelf_shadow_ws/install_standard/setup.bash
source /home/riot/Chris/bookshelf_guarded_control_ws/install/setup.bash

ros2 launch bookshelf_guarded_control_ros policy_tool_plan_only.launch.py
```

Inspect `/bookshelf_guarded/plan_report`. A first report may have
`valid=true`, but it must retain `execution_ready=false` until the planning
scene is independently completed and checked.

Do not launch `guarded_policy_tool_single_step.launch.py` merely to inspect a
plan. Enabling that process requires a separate reviewed experiment-specific
configuration; the committed file is intentionally non-executable.

Do not run the older `policy_to_robot_node`, `cartesian_action_executor_node`,
or `action_executor_node` alongside this package. They are explicitly blocked
because they use prototype frame/action mappings and do not provide this
package's planning, provenance, freshness, or approval gates.
