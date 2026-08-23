#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 XARM_ROS2_ROOT OUTPUT_URDF" >&2
  exit 2
fi

XARM_ROOT=$(realpath "$1")
OUTPUT=$(realpath -m "$2")
DESCRIPTION_ROOT="$XARM_ROOT/xarm_description"
XACRO="$DESCRIPTION_ROOT/urdf/xarm_device.urdf.xacro"

if [[ ! -f "$XACRO" ]]; then
  echo "xArm description was not found: $XACRO" >&2
  exit 1
fi
if ! command -v xacro >/dev/null 2>&1; then
  echo "xacro is unavailable; source ROS 2 and the xArm workspace first" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
TMP=$(mktemp --suffix=.urdf)
trap 'rm -f "$TMP"' EXIT

xacro "$XACRO" \
  add_gripper:=true \
  add_realsense_d435i:=false \
  dof:=7 \
  robot_type:=xarm \
  ros2_control_plugin:=uf_robot_hardware/UFRobotSystemHardware \
  >"$TMP"

# Isaac's importer can now resolve every mesh without a ROS package index.
sed "s#package://xarm_description#file://$DESCRIPTION_ROOT#g" "$TMP" >"$OUTPUT"

if grep -q 'package://xarm_description' "$OUTPUT"; then
  echo "Failed to replace all xarm_description mesh paths in: $OUTPUT" >&2
  exit 1
fi

grep -q '<link name="link_tcp"' "$OUTPUT"
grep -q '<joint name="drive_joint" type="revolute"' "$OUTPUT"
grep -q '<joint name="joint7" type="revolute"' "$OUTPUT"

echo "xArm7 Isaac URDF: $OUTPUT"
echo "export BOOKSHELF_XARM7_URDF_PATH=$OUTPUT"
