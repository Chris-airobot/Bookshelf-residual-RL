#!/usr/bin/env bash

# Shared environment setup for the canonical xArm experiment scripts.

BOOKSHELF_REPO_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  pwd
)"
BOOKSHELF_ROS_BUILD_ROOT="${BOOKSHELF_ROS_BUILD_ROOT:-${BOOKSHELF_REPO_ROOT}/.ros2_ws}"

_bookshelf_source_setup() {
  local setup_file="$1"
  if [[ -f "$setup_file" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$setup_file"
    set -u
  fi
}

# Do not inherit generated overlays from an earlier terminal session. In
# particular, colcon setup files can otherwise retain deleted /tmp underlays.
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset ROS_PACKAGE_PATH
_bookshelf_source_setup /opt/ros/humble/setup.bash

# Riot uses ros2_ws plus install_depth_fix. Alienware uses the official xArm
# checkout in bookshelf_xarm_sim_ws. Missing candidates are simply skipped.
_bookshelf_source_setup "${HOME}/Chris/ros2_ws/install/local_setup.bash"
_bookshelf_source_setup "${HOME}/Chris/ros2_ws/install_depth_fix/local_setup.bash"
_bookshelf_source_setup "${HOME}/Chris/bookshelf_xarm_sim_ws/install/local_setup.bash"

if [[ -n "${BOOKSHELF_ROS_UNDERLAYS:-}" ]]; then
  IFS=: read -r -a bookshelf_underlays <<<"${BOOKSHELF_ROS_UNDERLAYS}"
  for setup_file in "${bookshelf_underlays[@]}"; do
    _bookshelf_source_setup "$setup_file"
  done
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ROS 2 Humble is unavailable after environment setup." >&2
  return 1 2>/dev/null || exit 1
fi

missing_xarm_packages=()
for package_name in xarm_moveit_config xarm_moveit_servo xarm_planner; do
  if ! ros2 pkg prefix "$package_name" >/dev/null 2>&1; then
    missing_xarm_packages+=("$package_name")
  fi
done

if (( ${#missing_xarm_packages[@]} > 0 )); then
  cat >&2 <<'EOF'
ERROR: the official xArm ROS 2 underlay is incomplete.

Set BOOKSHELF_ROS_UNDERLAYS to its setup file, for example:
  export BOOKSHELF_ROS_UNDERLAYS="$HOME/Chris/bookshelf_xarm_sim_ws/install/local_setup.bash"
EOF
  printf 'Missing packages:' >&2
  printf ' %s' "${missing_xarm_packages[@]}" >&2
  printf '\n' >&2
  if [[ -d "${HOME}/Chris/bookshelf_xarm_sim_ws/src/xarm_ros2" ]]; then
    cat >&2 <<'EOF'

On Alienware, build the missing official package with:
  cd "$HOME/Chris/bookshelf_xarm_sim_ws"
  colcon build --symlink-install --packages-select xarm_planner
EOF
  fi
  return 1 2>/dev/null || exit 1
fi
