#!/usr/bin/env bash
# =============================================================================
# launch_teleop_record.sh — same as launch_teleop.sh, but adds a recording step
# for saving start / pick / air / drop joint states while you teleop.
#
# Usage:
#   ./scripts/launch_teleop_record.sh [OPTIONS]
#
# After the teleop app starts, open a second terminal and run:
#   python3 scripts/record_pick_place_positions.py
#
# Then use keys:
#   s = save start
#   1 = save pick
#   2 = save air transfer
#   3 = save drop
#   q = quit and save JSON file
#
# Saved file:
#   scripts/pick_place_positions.json
# =============================================================================

set -euo pipefail

ROBOT_IP="192.168.0.244"
DOF_MODE="3"
USE_SPACEMOUSE="true"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)            ROBOT_IP="$2"; shift 2 ;;
        --dof)           DOF_MODE="$2"; shift 2 ;;
        --no-spacemouse) USE_SPACEMOUSE="false"; shift ;;
        --help)
            sed -n '1,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$DOF_MODE" != "3" && "$DOF_MODE" != "6" ]]; then
    echo "[ERROR] --dof must be 3 or 6, got: ${DOF_MODE}"
    exit 1
fi

# Source ROS 2 and the workspace
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/workspace_env.sh"
cloudgripper_source_env "${REPO_ROOT}" "[teleop] "

PIDS=()
cleanup() {
    echo ""
    echo "[teleop] Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.2
    pkill -9 -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[teleop] Done."
}
trap cleanup EXIT INT TERM

if pgrep -f 'xarm_driver/xarm_driver_node' >/dev/null 2>&1; then
    echo "[teleop] Found leftover xarm_driver_node process(es) — killing them first."
    pkill -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.3
    pkill -9 -f 'xarm_driver/xarm_driver_node' 2>/dev/null || true
    sleep 0.5
fi

echo "[teleop] Starting xarm_driver_node  ip=${ROBOT_IP}  servo_mode=true ..."
ros2 run xarm_driver xarm_driver_node \
    --ros-args \
    --params-file "${REPO_ROOT}/config/safety.yaml" \
    -p robot_ip:="${ROBOT_IP}" \
    -p servo_mode:=true &
PIDS+=("$!")
DRIVER_PID="${PIDS[-1]}"

sleep 3
if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "[ERROR] xarm_driver_node exited unexpectedly. Check robot connection."
    exit 1
fi

echo "[teleop] xarm_driver_node running (pid=${DRIVER_PID})."
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
echo "[teleop] -------------------------------------------------------"
echo ""
echo "[record] To save the poses, open another terminal and run:"
echo "[record]   python3 ${REPO_ROOT}/scripts/record_pick_place_positions.py"
echo "[record]"
echo "[record] Keys inside that recorder:"
echo "[record]   s = save start"
echo "[record]   1 = save pick"
echo "[record]   2 = save air transfer"
echo "[record]   3 = save drop"
echo "[record]   q = quit and save JSON"

echo "[teleop] Now starting teleop in the background."
ros2 run teleop_node teleop_node \
    --ros-args \
    --params-file "${REPO_ROOT}/config/teleop.yaml" \
    --params-file "${REPO_ROOT}/config/safety.yaml" \
    -p dof_mode:="${DOF_MODE}" \
    -p use_spacemouse:="${USE_SPACEMOUSE}" \
    -p enable_recording:=false &
PIDS+=("$!")
TELEOP_PID="${PIDS[-1]}"

echo "[record] Starting joint-state recorder in the same terminal."
echo "[record] Use the keys below while teleop is active:"
echo "[record]   s = save start"
echo "[record]   1 = save pick"
echo "[record]   2 = save air transfer"
echo "[record]   3 = save drop"
echo "[record]   q = quit and save JSON"
python3 "${REPO_ROOT}/scripts/record_pick_place_positions.py" &
RECORDER_PID="$!"
PIDS+=("$RECORDER_PID")

wait "$RECORDER_PID"
kill "$TELEOP_PID" 2>/dev/null || true
wait "$TELEOP_PID" 2>/dev/null || true
