#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_experiment_common.sh
source "${SCRIPT_DIR}/_experiment_common.sh"

mkdir -p \
  "${BOOKSHELF_ROS_BUILD_ROOT}/build" \
  "${BOOKSHELF_ROS_BUILD_ROOT}/install" \
  "${BOOKSHELF_ROS_BUILD_ROOT}/log"

colcon \
  --log-base "${BOOKSHELF_ROS_BUILD_ROOT}/log" \
  build \
  --symlink-install \
  --base-paths \
    "${BOOKSHELF_REPO_ROOT}/ros2/bookshelf_policy_ros" \
    "${BOOKSHELF_REPO_ROOT}/ros2/bookshelf_shadow_ros" \
    "${BOOKSHELF_REPO_ROOT}/ros2/bookshelf_guarded_control_ros" \
  --build-base "${BOOKSHELF_ROS_BUILD_ROOT}/build" \
  --install-base "${BOOKSHELF_ROS_BUILD_ROOT}/install" \
  --packages-select \
    bookshelf_policy_ros \
    bookshelf_shadow_ros \
    bookshelf_guarded_control_ros

_bookshelf_source_setup "${BOOKSHELF_ROS_BUILD_ROOT}/install/local_setup.bash"

package_prefix="$(ros2 pkg prefix bookshelf_guarded_control_ros)"
launch_file="${package_prefix}/share/bookshelf_guarded_control_ros/launch/xarm7_policy_physical_experiment.launch.py"
if [[ ! -f "$launch_file" ]]; then
  echo "ERROR: canonical physical launch was not installed: ${launch_file}" >&2
  exit 1
fi

echo "Canonical ROS overlay: ${BOOKSHELF_ROS_BUILD_ROOT}/install"
echo "Physical launch: ${launch_file}"
echo "Build complete. Run: scripts/ros2/run_xarm_experiment.sh calculate"
