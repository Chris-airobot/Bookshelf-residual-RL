# External dependency patches

`xarm_moveit_config_bookshelf_named_states.patch` adds the bookshelf-specific
MoveIt named states `pre-insertion` and `pre-target` to the xArm 7 SRDF. It does
not replace or remove `home` or `hold-up`.

Apply it from a clean `xarm_ros2` checkout before building the MoveIt
configuration:

```bash
git -C /path/to/xarm_ros2 apply --check \
  /path/to/bookshelf-unified/patches/xarm_moveit_config_bookshelf_named_states.patch

git -C /path/to/xarm_ros2 apply \
  /path/to/bookshelf-unified/patches/xarm_moveit_config_bookshelf_named_states.patch

source /opt/ros/humble/setup.bash
cd /path/to/ros2_ws
colcon build --packages-select xarm_moveit_config
```

Restart MoveIt/RViz after rebuilding. Selecting a named state does not itself
execute it: plan and inspect the trajectory before choosing Execute.
