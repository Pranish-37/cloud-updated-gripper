#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ROBOT_IP="192.168.0.244"

# Source ROS + workspace environment
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/workspace_env.sh"
cloudgripper_source_env "${REPO_ROOT}" "[pick_place] "

# Kill any stale xarm driver so the robot is not left in a weird state
if pgrep -f 'xarm_driver/xarm_driver_node' >/dev/null 2>&1; then
    echo "[pick_place] Found stale xarm_driver_node — stopping it."
    pkill -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.3
    pkill -9 -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.5
fi

echo "[pick_place] Starting xarm_driver_node in safe blocking mode..."
ros2 run xarm_driver xarm_driver_node \
    --ros-args \
    --params-file "${REPO_ROOT}/config/safety.yaml" \
    -p robot_ip:="${ROBOT_IP}" \
    -p servo_mode:=false &
DRIVER_PID=$!

# Give the driver time to connect and initialize
sleep 3
if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "[pick_place] xarm_driver_node exited unexpectedly."
    exit 1
fi

echo "[pick_place] Starting pick/place sequence..."
python3 "${REPO_ROOT}/scripts/teleop_pick_place_xyz.py"

# Ensure driver is cleaned up when the python script exits
kill "$DRIVER_PID" 2>/dev/null || true
wait "$DRIVER_PID" 2>/dev/null || true
