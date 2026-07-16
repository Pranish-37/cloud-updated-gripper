#!/usr/bin/env bash
# =============================================================================
# launch_evaluation.sh — Launch policy evaluation stack
#
# Starts: realsense2_camera + ae_awb_controller + xarm_driver +
#         obs_aggregator + inference_node + episode_manager
#
# Usage:
#   ./scripts/launch_evaluation.sh [OPTIONS]
#
# Options:
#   --policy <path>       Path to pretrained_model/          (default: from YAML)
#   --dataset-meta <path> Training dataset root with meta/   (default: from YAML)
#   --output <path>       Eval log output directory          (default: from YAML)
#   --dof <3|6>           DoF mode                           (default: from YAML)
#   --ip <ip>             Robot IP address                   (default: from YAML)
#   --device <cuda|cpu>   Inference device                   (default: from YAML)
#   --help                Show this help message
#
# After launch:
#   1. Lock camera exposure once converged:
#        ros2 service call /camera/lock_exposure std_srvs/srv/Trigger
#   2. Follow stdin prompts in the episode_manager terminal
# =============================================================================

set -euo pipefail

umask 022

POLICY_PATH=""
DATASET_META_PATH=""
EVAL_OUTPUT_DIR=""
DOF_MODE=""
ROBOT_IP=""
DEVICE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LAUNCH_FILE="${REPO_ROOT}/launch/evaluation.launch.py"
CONFIG_FILE="${REPO_ROOT}/config/evaluation.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy)        POLICY_PATH="$2";        shift 2 ;;
        --dataset-meta)  DATASET_META_PATH="$2";  shift 2 ;;
        --output)        EVAL_OUTPUT_DIR="$2";    shift 2 ;;
        --dof)           DOF_MODE="$2";           shift 2 ;;
        --ip)            ROBOT_IP="$2";           shift 2 ;;
        --device)        DEVICE="$2";             shift 2 ;;
        --help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$LAUNCH_FILE" ]]; then
    echo "[ERROR] Launch file not found: ${LAUNCH_FILE}"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/workspace_env.sh"
cloudgripper_source_env "${REPO_ROOT}" "[evaluation] "

RESOLVED_OUTPUT="${EVAL_OUTPUT_DIR:-}"
if [[ -z "$RESOLVED_OUTPUT" ]]; then
    RESOLVED_OUTPUT="$(grep 'eval_output_dir:' "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')"
fi
if [[ -n "$RESOLVED_OUTPUT" ]]; then
    mkdir -p "$RESOLVED_OUTPUT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            CloudGripper V2 — Policy Evaluation              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf  "║  config    : %-49s║\n" "config/evaluation.yaml"
[[ -n "$POLICY_PATH"       ]] && printf "║  policy    : %-49s║\n" "${POLICY_PATH}"
[[ -n "$DATASET_META_PATH" ]] && printf "║  dataset   : %-49s║\n" "${DATASET_META_PATH}"
[[ -n "$EVAL_OUTPUT_DIR"   ]] && printf "║  output    : %-49s║\n" "${EVAL_OUTPUT_DIR}"
[[ -n "$DOF_MODE"          ]] && printf "║  dof       : %-49s║\n" "${DOF_MODE}"
[[ -n "$DEVICE"            ]] && printf "║  device    : %-49s║\n" "${DEVICE}"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  After launch: lock camera exposure when converged:         ║"
echo "║    ros2 service call /camera/lock_exposure std_srvs/srv/Trigger ║"
echo "║  Follow stdin prompts in the episode_manager terminal.      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

LAUNCH_ARGS=()
if [[ -n "$POLICY_PATH"       ]]; then LAUNCH_ARGS+=("policy_path:=${POLICY_PATH}"); fi
if [[ -n "$DATASET_META_PATH" ]]; then LAUNCH_ARGS+=("dataset_meta_path:=${DATASET_META_PATH}"); fi
if [[ -n "$EVAL_OUTPUT_DIR"   ]]; then LAUNCH_ARGS+=("eval_output_dir:=${EVAL_OUTPUT_DIR}"); fi
if [[ -n "$DOF_MODE"          ]]; then LAUNCH_ARGS+=("dof_mode:=${DOF_MODE}"); fi
if [[ -n "$ROBOT_IP"          ]]; then LAUNCH_ARGS+=("robot_ip:=${ROBOT_IP}"); fi
if [[ -n "$DEVICE"            ]]; then LAUNCH_ARGS+=("device:=${DEVICE}"); fi

echo "[evaluation] Launching..."
echo ""

exec ros2 launch "$LAUNCH_FILE" "${LAUNCH_ARGS[@]}"
