#!/usr/bin/env bash
# =============================================================================
# docker_shell.sh — Launch an interactive ROS 2 Humble Docker environment
#
# Usage:
#   ./scripts/docker_shell.sh [OPTIONS]
#
# Options:
#   --no-gpu      Disable GPU passthrough (default: enabled if available)
#   --root        Run as root inside the container (rare — e.g. extra apt installs)
#   --help        Show this help message
#
# On first entry, run docker_setup.sh inside the container to colcon build.
# Subsequent runs reuse build artifacts on the mounted volume.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE="cloudgripper_ros:humble"
CONTAINER_NAME="cloudgripper_ros"
USE_GPU="true"
RUN_AS_ROOT="false"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-gpu) USE_GPU="false"; shift ;;
        --root)   RUN_AS_ROOT="true"; shift ;;
        --help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Build GPU flags
# ---------------------------------------------------------------------------
GPU_FLAGS=()
if [[ "$USE_GPU" == "true" ]]; then
    if sudo docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        GPU_FLAGS=(--gpus all)
        echo "[docker] NVIDIA runtime detected — GPU passthrough enabled."
    else
        echo "[docker] NVIDIA runtime not found — running without GPU."
    fi
fi

# ---------------------------------------------------------------------------
# Build image if not present
# ---------------------------------------------------------------------------
if ! sudo docker image inspect "$IMAGE" &>/dev/null; then
    echo "[docker] Image '${IMAGE}' not found — building from Dockerfile (first time, ~5 min)..."
    sudo docker build -t "$IMAGE" "${REPO_ROOT}"
fi

# ---------------------------------------------------------------------------
# Run container
# ---------------------------------------------------------------------------
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(id -un)"

ENTRYPOINT_ARGS=()
ENV_FLAGS=(
    -e "TERM=xterm-256color"
    -e "DISPLAY=${DISPLAY:-:0}"
)

if [[ "$RUN_AS_ROOT" == "true" ]]; then
    echo "[docker] Running as root."
    ENV_FLAGS+=(-e "XDG_RUNTIME_DIR=/tmp/runtime-root")
else
    echo "[docker] Running as host user ${HOST_USER} (${HOST_UID}:${HOST_GID})."
    ENV_FLAGS+=(
        -e "HOST_UID=${HOST_UID}"
        -e "HOST_GID=${HOST_GID}"
        -e "HOST_USER=${HOST_USER}"
        -e "XDG_RUNTIME_DIR=/tmp/runtime-${HOST_UID}"
    )
    ENTRYPOINT_ARGS=(
        --entrypoint /workspace/scripts/docker_entrypoint.sh
    )
fi

echo "[docker] Starting container '${CONTAINER_NAME}'..."
echo "[docker] Workspace mounted at /workspace"
echo "[docker] First time: bash /workspace/scripts/docker_setup.sh  (colcon build only)"
echo ""

sudo docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --network host \
    --privileged \
    "${GPU_FLAGS[@]}" \
    "${ENTRYPOINT_ARGS[@]}" \
    -v /dev:/dev \
    -v "${REPO_ROOT}:/workspace" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /workspace \
    "${ENV_FLAGS[@]}" \
    "$IMAGE" \
    bash --rcfile /workspace/scripts/docker_rc.bash
