#!/usr/bin/env bash
# =============================================================================
# docker_setup.sh — First-time workspace setup inside the Docker container
#
# Run this ONCE after launching the container for the first time:
#   bash /workspace/scripts/docker_setup.sh
#
# Works as the default (non-root) container user — apt steps are skipped
# because the Docker image already includes ROS packages. Use
# ./scripts/docker_shell.sh --root only if you need to apt-install extras.
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
ROS_DISTRO="${ROS_DISTRO:-humble}"
AS_ROOT=false
[[ "$(id -u)" -eq 0 ]] && AS_ROOT=true

echo ""
echo "============================================================"
echo " CloudGripper V2 — Docker first-time setup"
echo " ROS distro : ${ROS_DISTRO}"
echo " Workspace  : ${WORKSPACE}"
echo " User       : $(id -un 2>/dev/null || echo "uid $(id -u)") (root=${AS_ROOT})"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. ROS 2 apt packages (root only — already baked into the Docker image)
# ---------------------------------------------------------------------------
if [[ "$AS_ROOT" == true ]]; then
    echo "[setup] Installing ROS 2 apt packages..."
    apt-get update -q
    apt-get install -y -q \
        ros-${ROS_DISTRO}-rclpy \
        ros-${ROS_DISTRO}-std-msgs \
        ros-${ROS_DISTRO}-std-srvs \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-rcl-interfaces \
        ros-${ROS_DISTRO}-realsense2-camera \
        ros-${ROS_DISTRO}-launch-ros \
        ros-${ROS_DISTRO}-rosidl-default-generators \
        ros-${ROS_DISTRO}-rosidl-default-runtime \
        libspnav-dev \
        libhidapi-hidraw0 \
        libhidapi-libusb0
    echo "[setup] ROS 2 packages installed."
else
    echo "[setup] Skipping apt install (not root)."
    echo "         ROS packages are already in the cloudgripper_ros:humble image."
    echo "         To install extra apt packages, re-enter with:"
    echo "           ./scripts/docker_shell.sh --root"
fi

# ---------------------------------------------------------------------------
# 2. Python packages
# ---------------------------------------------------------------------------
echo "[setup] Installing Python packages..."
if [[ "$AS_ROOT" == true ]]; then
    apt-get install -y -q python3-pip
fi
pip3 install --quiet -r "${WORKSPACE}/requirements.txt"
echo "[setup] Python packages installed."

# ---------------------------------------------------------------------------
# 3. lerobot v0.4.4 — last release with Python >=3.10 support
#    lerobot 0.5.0+ requires Python >=3.12 and uses PEP-695 generic syntax
#    which cannot be backported; v0.4.4 runs natively on Python 3.10.
# ---------------------------------------------------------------------------
LEROBOT_VERSION="v0.4.4"
LEROBOT_DIR="${WORKSPACE}/lerobot"
_expected_lerobot_version() {
    if [[ -f "${LEROBOT_DIR}/pyproject.toml" ]]; then
        grep -m1 '^version' "${LEROBOT_DIR}/pyproject.toml" \
            | sed -E 's/.*"([^"]+)".*/\1/'
    else
        echo "unknown"
    fi
}

if [[ ! -d "${LEROBOT_DIR}/.git" ]]; then
    echo "[setup] Cloning lerobot ${LEROBOT_VERSION} into ${LEROBOT_DIR}..."
    git clone --depth 1 --branch "${LEROBOT_VERSION}" \
        https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
    echo "[setup] lerobot ${LEROBOT_VERSION} cloned."
else
    CURRENT_VER=$(_expected_lerobot_version)
    if [[ "${CURRENT_VER}" != "0.4.4" ]]; then
        echo "[setup] Found lerobot ${CURRENT_VER} — re-cloning ${LEROBOT_VERSION} for Python 3.10 compat..."
        rm -rf "$LEROBOT_DIR"
        git clone --depth 1 --branch "${LEROBOT_VERSION}" \
            https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
        echo "[setup] lerobot ${LEROBOT_VERSION} cloned."
    else
        echo "[setup] lerobot ${LEROBOT_VERSION} already present — skipping clone."
    fi
fi

# Prevent colcon from treating lerobot as a ROS package.
touch "${LEROBOT_DIR}/COLCON_IGNORE"

# Upgrade pip only (not setuptools — colcon-core requires setuptools<80).
echo "[setup] Upgrading pip..."
pip3 install --quiet --upgrade pip
echo "[setup] pip upgraded."

# sympy 1.9 in the base image is distutils-installed; pip cannot uninstall it
# when lerobot pulls in a newer version.  Pre-install to shadow it.
echo "[setup] Pre-installing sympy to avoid distutils conflict..."
pip3 install --quiet --ignore-installed "sympy>=1.13"

# v0.4.4 declares Python >=3.10 natively — no --ignore-requires-python needed.
# The [dataset] extra does not exist in v0.4.4; datasets is a base dependency.
echo "[setup] Installing lerobot from ${LEROBOT_DIR} (editable)..."
rm -rf "${LEROBOT_DIR}"/UNKNOWN.egg-info "${LEROBOT_DIR}"/lerobot.egg-info
pip3 install -e "${LEROBOT_DIR}"
echo "[setup] lerobot installed."

# ---------------------------------------------------------------------------
# 4. Build workspace
# ---------------------------------------------------------------------------
echo "[setup] Building colcon workspace..."
set +u  # ROS setup.bash references unset vars, temporarily allow it
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
cd "$WORKSPACE"
# Remove stale build artifacts so cmake re-detects the numpy include path from
# scratch.  lerobot upgrades numpy 1.x -> 2.x, which moves the headers; old
# CMakeCache.txt entries pointing at the 1.x path cause build failures.
# Modern setuptools (>=71) also removed the legacy "setup.py --uninstall"
# command that colcon --cmake-clean-cache relies on, so a manual clean is
# the only safe approach.
rm -rf "${WORKSPACE}/build" "${WORKSPACE}/install" "${WORKSPACE}/log"
colcon build --symlink-install
echo "[setup] Build complete."

echo ""
echo "============================================================"
echo " Setup done! The workspace is ready."
echo " ROS environment is sourced automatically on next entry."
echo "============================================================"
echo ""
