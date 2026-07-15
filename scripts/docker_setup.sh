#!/usr/bin/env bash
# =============================================================================
# docker_setup.sh — colcon build inside Docker (deps are in the image / Dockerfile)
#
#   ./scripts/docker_shell.sh
#   bash /workspace/scripts/docker_setup.sh
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
umask 022

# shellcheck disable=SC1091
source "${WORKSPACE}/scripts/lib/workspace_env.sh"

_require_writable() {
    local path="$1"
    if [[ -e "$path" && ! -w "$path" ]]; then
        echo "[setup][ERROR] ${path} not writable by uid $(id -u)." >&2
        echo "             On host: sudo chown -R \$(id -u):\$(id -g) build install log" >&2
        exit 1
    fi
}

echo ""
echo "============================================================"
echo " CloudGripper V2 — Workspace build"
echo " User: $(id -un 2>/dev/null || echo "uid $(id -u)")"
echo "============================================================"
echo ""

_require_writable "${WORKSPACE}"
for _d in build install log; do _require_writable "${WORKSPACE}/${_d}"; done

cloudgripper_source_env "${WORKSPACE}" "[setup] " false
echo "[setup] python3: $(command -v python3) ($(python3 --version))"
echo ""

cloudgripper_colcon_build "${WORKSPACE}"

if [[ "$(id -u)" -eq 0 ]]; then
    _uid="$(stat -c '%u' "${WORKSPACE}")"
    _gid="$(stat -c '%g' "${WORKSPACE}")"
    if [[ "$_uid" -ne 0 ]]; then
        chown -R "${_uid}:${_gid}" "${WORKSPACE}/build" "${WORKSPACE}/install" "${WORKSPACE}/log" 2>/dev/null || true
    fi
fi

echo ""
echo "============================================================"
echo " Build done!"
echo " Verify: cloudgripper_verify_env /workspace   (or: cv)"
echo "============================================================"
echo ""
