#!/usr/bin/env bash
# =============================================================================
# docker_setup.sh — First-time workspace setup inside the Docker container
#
# Run this ONCE after launching the container for the first time:
#   bash /workspace/scripts/docker_setup.sh
#
# This script will:
#   1. Install required ROS 2 apt packages
#   2. Install Python dependencies (requirements.txt)
#   3. Build the colcon workspace
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
ROS_DISTRO="${ROS_DISTRO:-humble}"

echo ""
echo "============================================================"
echo " CloudGripper V2 — Docker first-time setup"
echo " ROS distro : ${ROS_DISTRO}"
echo " Workspace  : ${WORKSPACE}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. ROS 2 apt packages
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. Python packages
# ---------------------------------------------------------------------------
echo "[setup] Installing Python packages..."
apt-get install -y -q python3-pip
pip3 install --quiet -r "${WORKSPACE}/requirements.txt"
echo "[setup] Python packages installed."

# ---------------------------------------------------------------------------
# 3. Build workspace
# ---------------------------------------------------------------------------
echo "[setup] Building colcon workspace..."
set +u  # ROS setup.bash references unset vars, temporarily allow it
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
cd "$WORKSPACE"
colcon build --symlink-install
echo "[setup] Build complete."

echo ""
echo "============================================================"
echo " Setup done! The workspace is ready."
echo " ROS environment is sourced automatically on next entry."
echo "============================================================"
echo ""
