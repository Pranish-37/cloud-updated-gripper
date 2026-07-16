#!/usr/bin/env bash
# docker_entrypoint.sh — create a passwd entry for the host user, then drop privileges.
#
# Called by docker_shell.sh (container starts as root, then exec runuser).
# Keeps file ownership on /workspace aligned with the host UID/GID.


set -euo pipefail

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
HOST_USER="${HOST_USER:-dev}"

# Ensure runtime dir exists for GUI apps (teleop pygame).
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    mkdir -p "${XDG_RUNTIME_DIR}"
    chown "${HOST_UID}:${HOST_GID}" "${XDG_RUNTIME_DIR}" 2>/dev/null || chmod 700 "${XDG_RUNTIME_DIR}"
fi

if ! getent group "${HOST_GID}" >/dev/null; then
    if getent group "${HOST_USER}" >/dev/null; then
        groupmod -g "${HOST_GID}" "${HOST_USER}"
    else
        groupadd -g "${HOST_GID}" "${HOST_USER}"
    fi
fi
GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"

if ! getent passwd "${HOST_UID}" >/dev/null; then
    if getent passwd "${HOST_USER}" >/dev/null; then
        usermod -u "${HOST_UID}" -g "${GROUP_NAME}" -d "/home/${HOST_USER}" -s /bin/bash "${HOST_USER}"
    else
        useradd -u "${HOST_UID}" -g "${GROUP_NAME}" -d "/home/${HOST_USER}" -m -s /bin/bash "${HOST_USER}"
    fi
fi
CONTAINER_USER="$(getent passwd "${HOST_UID}" | cut -d: -f1)"

chown -R "${HOST_UID}:${HOST_GID}" "/home/${CONTAINER_USER}" 2>/dev/null || true

exec runuser -u "${CONTAINER_USER}" -w /workspace -- "$@"
