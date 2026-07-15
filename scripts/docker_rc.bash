# docker_rc.bash — auto-sourced when entering the container via docker_shell.sh

WORKSPACE="/workspace"
umask 022

# shellcheck disable=SC1091
source "${WORKSPACE}/scripts/lib/workspace_env.sh"

cloudgripper_source_env "${WORKSPACE}" "[docker] " false || true

if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
    echo "[docker] Workspace sourced: ${WORKSPACE}/install/setup.bash"
else
    echo "[docker] Workspace not built yet. Run:"
    echo "         bash ${WORKSPACE}/scripts/docker_setup.sh"
fi

alias cb='cloudgripper_colcon_build /workspace'
alias cs='source /workspace/install/setup.bash'
alias cv='cloudgripper_verify_env /workspace'

echo "[docker] ROS 2 ${CLOUDGRIPPER_ROS_DISTRO} ready. Workspace: ${WORKSPACE}"
