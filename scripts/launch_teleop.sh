#!/usr/bin/env bash
# =============================================================================
# launch_teleop.sh — Start xarm_driver (servo mode) + teleop_node
#
# Usage:
#   ./scripts/launch_teleop.sh [OPTIONS]
#
# Options:
#   --ip <ip>         Robot IP address       (default: 192.168.0.244)
#   --dof <3|6>       DoF mode               (default: 3)
#   --no-spacemouse   Disable SpaceMouse, use full keyboard control
#   --help            Show this help message
#
# Example:
#   ./scripts/launch_teleop.sh --ip 192.168.0.245 --dof 6 --no-spacemouse
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ROBOT_IP="192.168.0.244"
DOF_MODE="3"
USE_SPACEMOUSE="true"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)            ROBOT_IP="$2"; shift 2 ;;
        --dof)           DOF_MODE="$2"; shift 2 ;;
        --no-spacemouse) USE_SPACEMOUSE="false"; shift ;;
        --help)
            sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$DOF_MODE" != "3" && "$DOF_MODE" != "6" ]]; then
    echo "[ERROR] --dof must be 3 or 6, got: ${DOF_MODE}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Source ROS 2 and workspace
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/workspace_env.sh"
cloudgripper_source_env "${REPO_ROOT}" "[teleop] "

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
    # Hard-kill any leftover drivers this session (or previous aborted runs).
    pkill -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.2
    pkill -9 -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[teleop] Done."
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Ensure no stale drivers hold the robot TCP connection
# ---------------------------------------------------------------------------
if pgrep -f 'xarm_driver/xarm_driver_node' >/dev/null 2>&1; then
    echo "[teleop] Found leftover xarm_driver_node process(es) — killing them first."
    pkill -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.3
    pkill -9 -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.5
fi

# ---------------------------------------------------------------------------
# Launch xarm_driver_node (servo mode, background)
# ---------------------------------------------------------------------------
echo "[teleop] Starting xarm_driver_node  ip=${ROBOT_IP}  servo_mode=true ..."

ros2 run xarm_driver xarm_driver_node \
    --ros-args \
    --params-file "${REPO_ROOT}/config/safety.yaml" \
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
echo "[teleop] Starting teleop_node  dof=${DOF_MODE}  spacemouse=${USE_SPACEMOUSE}"
echo "[teleop] -------------------------------------------------------"
if [[ "$USE_SPACEMOUSE" == "true" ]]; then
    echo "[teleop]  SpaceMouse x/y/z   = translation"
    if [[ "$DOF_MODE" == "6" ]]; then
        echo "[teleop]  SpaceMouse r/p/y   = rotation"
    else
        echo "[teleop]  Rotation locked to home RPY (3-DoF)"
    fi
    echo "[teleop]  SpaceMouse L/R    = gripper open/close"
    echo "[teleop]  Q/E               = gripper open/close (backup)"
else
    echo "[teleop]  W/S=x   A/D=y   Space/Ctrl=z"
    if [[ "$DOF_MODE" == "6" ]]; then
        echo "[teleop]  J/L=roll   I/K=pitch   U/O=yaw"
    else
        echo "[teleop]  Rotation locked to home RPY (3-DoF)"
    fi
    echo "[teleop]  Q=gripper open   E=gripper close"
fi
echo "[teleop]  Esc=quit"
echo "[teleop]  (SpaceMouse must be centered/released before control arms)"
echo "[teleop] -------------------------------------------------------"
echo ""

ros2 run teleop_node teleop_node \
    --ros-args \
    --params-file "${REPO_ROOT}/config/teleop.yaml" \
    --params-file "${REPO_ROOT}/config/safety.yaml" \
    -p dof_mode:="${DOF_MODE}" \
    -p use_spacemouse:="${USE_SPACEMOUSE}" \
    -p enable_recording:=false

# Script exits here → trap fires → driver node is killed
