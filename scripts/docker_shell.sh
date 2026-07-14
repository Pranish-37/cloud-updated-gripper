#!/usr/bin/env bash
# =============================================================================
# docker_shell.sh — Launch an interactive ROS 2 Humble Docker environment
#
# Usage:
#   ./scripts/docker_shell.sh [OPTIONS]
#
# Options:
#   --no-gpu      Disable GPU passthrough (default: enabled if available)
#   --root        Run as root inside the container (needed for docker_setup.sh)
#   --help        Show this help message
#
# On first run, the container will build the workspace automatically.
# Subsequent runs reuse the build artifacts via the mounted volume.
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
USER_FLAGS=()
if [[ "$RUN_AS_ROOT" == "true" ]]; then
    echo "[docker] Running as root (for apt install / first-time setup)."
else
    USER_FLAGS=(-u "$(id -u):$(id -g)")
    echo "[docker] Running as host user $(id -u):$(id -g) (use --root for docker_setup.sh)."
fi

echo "[docker] Starting container '${CONTAINER_NAME}'..."
echo "[docker] Workspace mounted at /workspace"
echo "[docker] Run 'bash /workspace/scripts/docker_setup.sh' on first launch to build."
echo ""

sudo docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --network host \
    --privileged \
    "${GPU_FLAGS[@]}" \
    "${USER_FLAGS[@]}" \
    -v /dev:/dev \
    -v "${REPO_ROOT}:/workspace" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /workspace \
    -e "TERM=xterm-256color" \
    -e "DISPLAY=${DISPLAY:-:0}" \
    -e "XDG_RUNTIME_DIR=/tmp/runtime-root" \
    "$IMAGE" \
    bash --rcfile /workspace/scripts/docker_rc.bash
