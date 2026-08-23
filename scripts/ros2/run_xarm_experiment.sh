#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_experiment_common.sh
source "${SCRIPT_DIR}/_experiment_common.sh"

canonical_setup="${BOOKSHELF_ROS_BUILD_ROOT}/install/local_setup.bash"
if [[ ! -f "$canonical_setup" ]]; then
  echo "ERROR: canonical ROS overlay is not built." >&2
  echo "Run: scripts/ros2/build_xarm_experiment.sh" >&2
  exit 1
fi
_bookshelf_source_setup "$canonical_setup"

operation="${1:-calculate}"
if [[ "$operation" != "calculate" && "$operation" != "control" ]]; then
  echo "Usage: $0 [calculate|control]" >&2
  exit 2
fi

missing_runtime_packages=()
for package_name in realsense2_camera; do
  if ! ros2 pkg prefix "$package_name" >/dev/null 2>&1; then
    missing_runtime_packages+=("$package_name")
  fi
done
if (( ${#missing_runtime_packages[@]} > 0 )); then
  printf 'ERROR: physical experiment runtime is incomplete. Missing packages:' >&2
  printf ' %s' "${missing_runtime_packages[@]}" >&2
  printf '\n' >&2
  echo "Run this launcher on the machine with the physical xArm and RealSense ROS installation." >&2
  exit 1
fi

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

approved_config="${BOOKSHELF_APPROVED_CONFIG:-}"
if [[ -z "$approved_config" ]]; then
  approved_config="$(first_existing_file \
    "${HOME}/BookshelfFiles/experiment_configs/stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml")" || true
fi

policy_bundle="${BOOKSHELF_POLICY_BUNDLE:-}"
if [[ -z "$policy_bundle" ]]; then
  policy_bundle="$(first_existing_file \
    "${HOME}/BookshelfFiles/trained_models/bookshelf_residual_2026-07-08_shadow_actor.npz")" || true
fi

activation_envelope="${BOOKSHELF_ACTIVATION_ENVELOPE:-}"
if [[ -z "$activation_envelope" ]]; then
  activation_envelope="$(first_existing_file \
    "${HOME}/BookshelfFiles/policy_activation_envelopes/simulator_local_2026-08-08.json" \
    "${HOME}/BookshelfFiles/evaluation_results/policy_activation_envelopes/simulator_local_2026-08-08.json")" || true
fi

for required in "$approved_config" "$policy_bundle" "$activation_envelope"; do
  if [[ -z "$required" || ! -f "$required" ]]; then
    cat >&2 <<'EOF'
ERROR: an experiment artifact is missing.

Set these variables to reviewed files:
  BOOKSHELF_APPROVED_CONFIG
  BOOKSHELF_POLICY_BUNDLE
  BOOKSHELF_ACTIVATION_ENVELOPE
EOF
    exit 1
  fi
done

authorization_token="DISABLED"
release_boundary_confirmed="false"
if [[ "$operation" == "control" ]]; then
  authorization_token="${BOOKSHELF_AUTHORIZATION_TOKEN:-}"
  release_boundary_confirmed="${BOOKSHELF_PHYSICAL_RELEASE_BOUNDARY_CONFIRMED:-false}"
  if [[ "$authorization_token" != "I_APPROVE_XARM_FULL_EPISODE" ]]; then
    echo "ERROR: control requires BOOKSHELF_AUTHORIZATION_TOKEN=I_APPROVE_XARM_FULL_EPISODE" >&2
    exit 1
  fi
  if [[ "$release_boundary_confirmed" != "true" ]]; then
    echo "ERROR: control requires BOOKSHELF_PHYSICAL_RELEASE_BOUNDARY_CONFIRMED=true" >&2
    exit 1
  fi
fi

trial_name="${BOOKSHELF_TRIAL_NAME:-xarm_${operation}_$(date +%Y%m%d_%H%M%S)}"
experiment_output_root="${BOOKSHELF_EXPERIMENT_OUTPUT_ROOT:-${HOME}/BookshelfFiles/experiment_logs}"

echo "Repository: ${BOOKSHELF_REPO_ROOT}"
echo "ROS overlay: ${BOOKSHELF_ROS_BUILD_ROOT}/install"
echo "Operation: ${operation}"
echo "Trial: ${trial_name}"
echo "Approved config: ${approved_config}"
echo "Policy: ${policy_bundle}"
echo "Activation envelope: ${activation_envelope}"

exec ros2 launch bookshelf_guarded_control_ros \
  xarm7_policy_physical_experiment.launch.py \
  trial_name:="${trial_name}" \
  approved_config:="${approved_config}" \
  policy_bundle:="${policy_bundle}" \
  activation_envelope:="${activation_envelope}" \
  repository_path:="${BOOKSHELF_REPO_ROOT}" \
  experiment_output_root:="${experiment_output_root}" \
  robot_ip:="${BOOKSHELF_ROBOT_IP:-192.168.1.209}" \
  show_rviz:="${BOOKSHELF_SHOW_RVIZ:-false}" \
  record_camera:="${BOOKSHELF_RECORD_CAMERA:-true}" \
  operation:="${operation}" \
  authorization_token:="${authorization_token}" \
  physical_release_boundary_confirmed:="${release_boundary_confirmed}" \
  start_immediately:="${BOOKSHELF_START_IMMEDIATELY:-false}"
