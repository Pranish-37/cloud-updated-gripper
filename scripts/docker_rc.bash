# docker_rc.bash — sourced automatically when entering the container via docker_shell.sh

ROS_DISTRO="${ROS_DISTRO:-humble}"
WORKSPACE="/workspace"

# Source ROS 2 base
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# Source workspace overlay if already built
if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
    source "${WORKSPACE}/install/setup.bash"
    echo "[docker] Workspace sourced: ${WORKSPACE}/install/setup.bash"
else
    echo "[docker] Workspace not built yet. Run:"
    echo "         bash ${WORKSPACE}/scripts/docker_setup.sh"
fi

# Useful aliases
alias cb='cd /workspace && colcon build --symlink-install && source install/setup.bash'
alias cs='source /workspace/install/setup.bash'

echo "[docker] ROS 2 ${ROS_DISTRO} ready. Workspace: ${WORKSPACE}"
