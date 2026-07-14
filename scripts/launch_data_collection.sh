#!/usr/bin/env bash
# =============================================================================
# launch_data_collection.sh — Launch full teleop data collection stack
#
# Starts: realsense2_camera + ae_awb_controller + xarm_driver +
#         obs_aggregator + recorder_node + teleop_node
#
# Usage:
#   ./scripts/launch_data_collection.sh [OPTIONS]
#
# Options:
#   --dataset <path>      Dataset root directory     (default: from YAML)
#   --repo-id <id>        HuggingFace-style repo id  (default: from YAML)
#   --episodes <n>        Total episodes to collect  (default: from YAML)
#   --dof <3|6>           DoF mode                   (default: from YAML)
#   --task <string>       Task description string    (default: from YAML)
#   --ip <ip>             Robot IP address           (default: 192.168.0.244)
#   --no-spacemouse       Use keyboard rotation only
#   --push-to-hub         Upload dataset to HuggingFace after collection (default: off)
#   --help                Show this help message
#
# Examples:
#   ./scripts/launch_data_collection.sh
#   ./scripts/launch_data_collection.sh --episodes 50 --dof 3
#   ./scripts/launch_data_collection.sh \
#       --dataset ~/datasets/pnp-3dof \
#       --repo-id MyName/pnp-3dof-200 \
#       --episodes 200 \
#       --dof 3 \
#       --task "pick and place cube"
#
# Controls (once the pygame window opens):
#   R              Toggle recording on / off
#   N              Save episode, start next
#   D              Discard episode, restart same index
#   ESC            Finalize dataset and quit
#   W/S A/D        EE +x/-x  +y/-y
#   Space/Ctrl     EE +z/-z
#   SpaceMouse     Roll / pitch / yaw
#   Q / E          Gripper open / close
# =============================================================================

set -euo pipefail

# Dataset files land on the host mount; keep them world-readable.
umask 022

# ---------------------------------------------------------------------------
# Defaults — overridable via CLI flags
# ---------------------------------------------------------------------------
ROBOT_IP="192.168.0.244"
USE_SPACEMOUSE="true"
DATASET_ROOT=""       # empty = use YAML value
REPO_ID=""
EPISODES=""
DOF_MODE=""
TASK=""
PUSH_TO_HUB=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LAUNCH_FILE="${REPO_ROOT}/launch/data_collection.launch.py"
CONFIG_FILE="${REPO_ROOT}/config/data_collection.yaml"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET_ROOT="$2";  shift 2 ;;
        --repo-id)       REPO_ID="$2";       shift 2 ;;
        --episodes)      EPISODES="$2";      shift 2 ;;
        --dof)           DOF_MODE="$2";      shift 2 ;;
        --task)          TASK="$2";          shift 2 ;;
        --ip)            ROBOT_IP="$2";      shift 2 ;;
        --no-spacemouse) USE_SPACEMOUSE="false"; shift ;;
        --push-to-hub)   PUSH_TO_HUB="true";    shift ;;
        --help)
            sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate required files
# ---------------------------------------------------------------------------
if [[ ! -f "$LAUNCH_FILE" ]]; then
    echo "[ERROR] Launch file not found: ${LAUNCH_FILE}"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_FILE}"
    echo "        Expected at: config/data_collection.yaml"
    exit 1
fi

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
    echo "        cd ${REPO_ROOT} && colcon build --symlink-install"
    exit 1
fi
set +u
source "$INSTALL_SETUP"
set -u

# ---------------------------------------------------------------------------
# Print session summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            CloudGripper V2 — Data Collection                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf  "║  config    : %-49s║\n" "config/data_collection.yaml"
printf  "║  robot ip  : %-49s║\n" "${ROBOT_IP}"
[[ -n "$DATASET_ROOT" ]] && printf "║  dataset   : %-49s║\n" "${DATASET_ROOT}"
[[ -n "$EPISODES"     ]] && printf "║  episodes  : %-49s║\n" "${EPISODES}"
[[ -n "$DOF_MODE"     ]] && printf "║  dof       : %-49s║\n" "${DOF_MODE}"
[[ -n "$TASK"         ]] && printf "║  task      : %-49s║\n" "${TASK}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Controls (pygame window)                                    ║"
echo "║    R          toggle recording on/off                        ║"
echo "║    N          save episode, start next                       ║"
echo "║    Backspace  discard episode, restart same index            ║"
echo "║    ESC        finalize dataset and quit                      ║"
echo "║    W/S A/D    EE +x/-x  +y/-y                               ║"
echo "║    Space/Ctrl EE +z/-z                                       ║"
echo "║    SpaceMouse roll / pitch / yaw                             ║"
echo "║    Q / E      gripper open / close                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  After launch: call the lock service once camera converges:  ║"
echo "║    ros2 service call /camera/lock_exposure std_srvs/srv/Trigger ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Build ros2 launch argument list
# ---------------------------------------------------------------------------
LAUNCH_ARGS=()

# Use if-blocks instead of [[ ]] && to avoid set -e triggering on false conditions
if [[ -n "$DATASET_ROOT" ]]; then LAUNCH_ARGS+=("dataset_root:=${DATASET_ROOT}"); fi
if [[ -n "$REPO_ID"      ]]; then LAUNCH_ARGS+=("dataset_repo_id:=${REPO_ID}");   fi
if [[ -n "$EPISODES"     ]]; then LAUNCH_ARGS+=("total_episodes:=${EPISODES}");   fi
if [[ -n "$DOF_MODE"     ]]; then LAUNCH_ARGS+=("dof_mode:=${DOF_MODE}");         fi
if [[ -n "$TASK"         ]]; then LAUNCH_ARGS+=("task:=${TASK}");                 fi
if [[ -n "$PUSH_TO_HUB"  ]]; then LAUNCH_ARGS+=("push_to_hub:=${PUSH_TO_HUB}");  fi

# Warn if CLI --ip differs from what is in the YAML (key is now indented under ros__parameters)
YAML_IP=$(grep 'robot_ip:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' || true)
if [[ -n "$YAML_IP" && "$ROBOT_IP" != "$YAML_IP" ]]; then
    echo "[WARN] --ip ${ROBOT_IP} differs from robot_ip in YAML (${YAML_IP})."
    echo "       Update config/data_collection.yaml to persist this change."
    echo ""
fi

YAML_SM=$(grep 'use_spacemouse:' "$CONFIG_FILE" | awk '{print $2}' || true)
if [[ "$USE_SPACEMOUSE" == "false" && "$YAML_SM" != "false" ]]; then
    echo "[WARN] --no-spacemouse passed but use_spacemouse: true in YAML."
    echo "       Add 'use_spacemouse: false' to config/data_collection.yaml to persist."
    echo ""
fi

# ---------------------------------------------------------------------------
# lerobot PYTHONPATH — ros2 launch does not inherit the interactive shell's
# PYTHONPATH set by docker_rc.bash, so we must export it explicitly here.
# ---------------------------------------------------------------------------
LEROBOT_WS="/workspace/lerobot"
if [[ -d "${LEROBOT_WS}/src/lerobot" ]]; then
    export PYTHONPATH="${LEROBOT_WS}/src:${PYTHONPATH:-}"
    echo "[data_collection] lerobot PYTHONPATH (src layout): ${LEROBOT_WS}/src"
elif [[ -d "${LEROBOT_WS}/lerobot" ]]; then
    export PYTHONPATH="${LEROBOT_WS}:${PYTHONPATH:-}"
    echo "[data_collection] lerobot PYTHONPATH (flat layout): ${LEROBOT_WS}"
else
    echo "[WARN] /workspace/lerobot not found — recorder_node will fail to import lerobot."
    echo "       Run: bash /workspace/scripts/docker_setup.sh"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[data_collection] Launching..."
echo ""

exec ros2 launch "$LAUNCH_FILE" "${LAUNCH_ARGS[@]}"
