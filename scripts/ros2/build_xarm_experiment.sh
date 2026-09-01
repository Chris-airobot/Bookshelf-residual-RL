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
    "${BOOKSHELF_REPO_ROOT}/ros2/bookshelf_simple_experiment_ros" \
    "${BOOKSHELF_REPO_ROOT}/ros2/bookshelf_rviz_image_panel" \
  --build-base "${BOOKSHELF_ROS_BUILD_ROOT}/build" \
  --install-base "${BOOKSHELF_ROS_BUILD_ROOT}/install" \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
  --packages-select \
    bookshelf_policy_ros \
    bookshelf_shadow_ros \
    bookshelf_guarded_control_ros \
    bookshelf_simple_experiment_ros \
    bookshelf_rviz_image_panel

_bookshelf_source_setup "${BOOKSHELF_ROS_BUILD_ROOT}/install/local_setup.bash"

package_prefix="$(ros2 pkg prefix bookshelf_simple_experiment_ros)"
launch_file="${package_prefix}/share/bookshelf_simple_experiment_ros/launch/real_experiment_operator.launch.py"
rehearsal_file="${package_prefix}/share/bookshelf_simple_experiment_ros/launch/offline_full_sequence_rehearsal.launch.py"
if [[ ! -f "$launch_file" || ! -f "$rehearsal_file" ]]; then
  echo "ERROR: complete experiment launches were not installed." >&2
  exit 1
fi

echo "Canonical ROS overlay: ${BOOKSHELF_ROS_BUILD_ROOT}/install"
echo "Physical launch: ${launch_file}"
echo "Offline rehearsal: ${rehearsal_file}"
