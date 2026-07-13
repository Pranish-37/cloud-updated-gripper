#!/usr/bin/env bash
# =============================================================================
# launch_teleop.sh — Start xarm_driver (servo mode) + teleop_node
#
# Usage:
#   ./scripts/launch_teleop.sh [OPTIONS]
#
# Options:
#   --ip <ip>         Robot IP address       (default: 192.168.0.244)
#   --no-spacemouse   Disable SpaceMouse, use keyboard-only rotation
#   --help            Show this help message
#
# Example:
#   ./scripts/launch_teleop.sh --ip 192.168.0.245 --no-spacemouse
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ROBOT_IP="192.168.0.244"
USE_SPACEMOUSE="true"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)            ROBOT_IP="$2"; shift 2 ;;
        --no-spacemouse) USE_SPACEMOUSE="false"; shift ;;
        --help)
            sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Source ROS 2 and workspace
# ---------------------------------------------------------------------------
ROS_DISTRO="${ROS_DISTRO:-humble}"

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    echo "[ERROR] ROS 2 (${ROS_DISTRO}) not found at /opt/ros/${ROS_DISTRO}/"
    echo "        Set ROS_DISTRO env var to match your installation."
    exit 1
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

INSTALL_SETUP="${REPO_ROOT}/install/setup.bash"
if [[ ! -f "$INSTALL_SETUP" ]]; then
    echo "[ERROR] Workspace not built. Run first:"
    echo "        cd ${REPO_ROOT} && colcon build"
    exit 1
fi
set +u
source "$INSTALL_SETUP"
set -u

# ---------------------------------------------------------------------------
# Cleanup on exit — kill background nodes when script exits or Ctrl-C
# ---------------------------------------------------------------------------
PIDS=()

cleanup() {
    echo ""
    echo "[teleop] Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[teleop] Done."
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Launch xarm_driver_node (servo mode, background)
# ---------------------------------------------------------------------------
echo "[teleop] Starting xarm_driver_node  ip=${ROBOT_IP}  servo_mode=true ..."

ros2 run xarm_driver xarm_driver_node \
    --ros-args \
    -p robot_ip:="${ROBOT_IP}" \
    -p servo_mode:=true &

PIDS+=($!)
DRIVER_PID="${PIDS[-1]}"

# Give the driver time to connect to the robot
echo "[teleop] Waiting for xarm_driver to connect (3 s)..."
sleep 3

if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "[ERROR] xarm_driver_node exited unexpectedly. Check robot connection."
    exit 1
fi
echo "[teleop] xarm_driver_node running (pid=${DRIVER_PID})."

# ---------------------------------------------------------------------------
# Launch teleop_node (foreground — blocks until Esc)
# ---------------------------------------------------------------------------
echo ""
echo "[teleop] Starting teleop_node  spacemouse=${USE_SPACEMOUSE}"
echo "[teleop] -------------------------------------------------------"
echo "[teleop]  W/S=x   A/D=y   Space/Ctrl=z"
if [[ "$USE_SPACEMOUSE" == "true" ]]; then
    echo "[teleop]  SpaceMouse cap = roll / pitch / yaw"
else
    echo "[teleop]  J/L=roll   I/K=pitch   U/O=yaw"
fi
echo "[teleop]  Q=gripper open   E=gripper close   Esc=quit"
echo "[teleop] -------------------------------------------------------"
echo ""

ros2 run teleop_node teleop_node \
    --ros-args \
    -p use_spacemouse:="${USE_SPACEMOUSE}"

# Script exits here → trap fires → driver node is killed
