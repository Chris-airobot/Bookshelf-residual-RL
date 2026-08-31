#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "scan" && "$1" != "loading" ) ]]; then
  echo "Usage: $0 {scan|loading}" >&2
  exit 2
fi

pose_name="$1"
config_root="${BOOKSHELF_EXPERIMENT_CONFIG_ROOT:-${HOME}/BookshelfFiles/experiment_configs}"
pose_dir="${config_root}/operator_joint_poses"
output_path="${pose_dir}/${pose_name}_joint_state.yaml"

mkdir -p "$pose_dir"
temporary_path="$(mktemp "${pose_dir}/.${pose_name}_joint_state.XXXXXX")"
trap 'rm -f "$temporary_path"' EXIT

echo "Waiting for one /joint_states message for the ${pose_name} pose..."
ros2 topic echo --once /joint_states >"$temporary_path"

if [[ ! -s "$temporary_path" ]]; then
  echo "ERROR: /joint_states produced an empty snapshot." >&2
  exit 1
fi

mv "$temporary_path" "$output_path"
trap - EXIT
echo "Saved ${pose_name} pose: ${output_path}"
