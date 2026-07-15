# workspace_env.sh — shared ROS / Python environment for Docker and launch scripts.
#
# Usage:
#   source /path/to/scripts/lib/workspace_env.sh
#   cloudgripper_source_env /path/to/repo "[prefix]" [require_install]
#
# Optional checks (README Step 1 / Step 2 verify):
#   cloudgripper_verify_env /path/to/repo

: "${CLOUDGRIPPER_ROS_DISTRO:=${ROS_DISTRO:-humble}}"
: "${CLOUDGRIPPER_LEROBOT_VERSION:=0.4.4}"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_cloudgripper_strip_conda_from_path() {
    local part
    local -a kept=()
    local IFS=':'
    for part in ${PATH:-}; do
        [[ "$part" == "/opt/conda/bin" ]] && continue
        kept+=("$part")
    done
    PATH="$(IFS=':'; echo "${kept[*]}")"
    export PATH
}

_cloudgripper_ensure_system_python() {
    _cloudgripper_strip_conda_from_path
    local py
    py="$(command -v python3)"
    if ! python3 -c "import em" 2>/dev/null; then
        echo "${1}[ERROR] python3 at ${py} cannot import em (empy)." >&2
        echo "${1}        Rebuild image: sudo docker build -t cloudgripper_ros:humble ." >&2
        return 1
    fi
}

_cloudgripper_lerobot_version_from_pyproject() {
    local toml="$1"
    [[ -f "$toml" ]] || { echo "unknown"; return; }
    grep -m1 '^version' "$toml" | sed -E 's/.*"([^"]+)".*/\1/'
}

_cloudgripper_check_numpy_scipy() {
    local prefix="${1:-}"
    if ! python3 -c "
import numpy, scipy
path = scipy.__file__
assert '/usr/lib/python3/dist-packages' not in path, path
" 2>/dev/null; then
        python3 -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__file__)" 2>/dev/null || true
        echo "${prefix}[ERROR] scipy is the apt build (incompatible with lerobot numpy 2.x)." >&2
        echo "${prefix}        Fix: pip3 install scipy==1.15.3  (or rebuild image)" >&2
        return 1
    fi
}

_cloudgripper_setup_lerobot() {
    local repo_root="$1"
    local prefix="${2:-}"
    local ws="${repo_root}/lerobot"
    local ver=""

    # Default: pip-installed /opt/lerobot (stable deps). Opt-in dev override:
    #   export CLOUDGRIPPER_LEROBOT_DEV=1
    if [[ "${CLOUDGRIPPER_LEROBOT_DEV:-0}" == "1" && -d "${ws}/src/lerobot" ]]; then
        ver=$(_cloudgripper_lerobot_version_from_pyproject "${ws}/pyproject.toml")
        if [[ "$ver" == "$CLOUDGRIPPER_LEROBOT_VERSION" ]]; then
            export PYTHONPATH="${ws}/src:${PYTHONPATH:-}"
            echo "${prefix}lerobot ${ver} (workspace dev)"
            return 0
        fi
        echo "${prefix}[WARN] workspace lerobot ${ver} != ${CLOUDGRIPPER_LEROBOT_VERSION}, using image" >&2
    fi

    if python3 -c "import lerobot" 2>/dev/null; then
        ver="$(python3 -c "import lerobot; print(lerobot.__version__)" 2>/dev/null || echo unknown)"
        if [[ "$ver" == "$CLOUDGRIPPER_LEROBOT_VERSION" ]]; then
            echo "${prefix}lerobot ${ver} (image /opt/lerobot)"
            return 0
        fi
        echo "${prefix}[ERROR] lerobot ${ver} != ${CLOUDGRIPPER_LEROBOT_VERSION}. Rebuild image." >&2
        return 1
    fi

    echo "${prefix}[WARN] lerobot not importable" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Source ROS distro + workspace overlay; fix PATH for colcon/ros2; configure lerobot.
#   $1  repo root path
#   $2  log prefix (optional, e.g. "[docker] ")
#   $3  require install/setup.bash — "true" (default) or "false"
cloudgripper_source_env() {
    local repo_root="$1"
    local prefix="${2:-}"
    local require_install="${3:-true}"

    if [[ ! -f "/opt/ros/${CLOUDGRIPPER_ROS_DISTRO}/setup.bash" ]]; then
        echo "${prefix}[ERROR] ROS ${CLOUDGRIPPER_ROS_DISTRO} not found under /opt/ros/" >&2
        return 1
    fi

    set +u
    # shellcheck disable=SC1091
    source "/opt/ros/${CLOUDGRIPPER_ROS_DISTRO}/setup.bash"

    local install_setup="${repo_root}/install/setup.bash"
    if [[ -f "$install_setup" ]]; then
        # shellcheck disable=SC1090
        source "$install_setup"
    elif [[ "$require_install" == true ]]; then
        set -u
        echo "${prefix}[ERROR] Workspace not built. Run: bash ${repo_root}/scripts/docker_setup.sh" >&2
        return 1
    fi
    set -u

    _cloudgripper_ensure_system_python "$prefix" || return 1
    _cloudgripper_check_numpy_scipy "$prefix" || return 1
    _cloudgripper_setup_lerobot "$repo_root" "$prefix" || true
    return 0
}

# Rebuild colcon workspace (interactive shell alias target).
cloudgripper_colcon_build() {
    local repo_root="${1:-/workspace}"
    cloudgripper_source_env "$repo_root" "" false
    cd "$repo_root"
    rm -rf build install log
    colcon build --symlink-install
    set +u
    # shellcheck disable=SC1091
    source "${repo_root}/install/setup.bash"
    set -u
}

# README verification helper — run inside container after image build / docker_setup.
cloudgripper_verify_env() {
    local repo_root="${1:-/workspace}"
    local prefix="[verify] "

    cloudgripper_source_env "$repo_root" "$prefix" false || return 1

    echo "${prefix}PASS python3=$(command -v python3) ($(python3 --version))"
    python3 -c "import em; print('${prefix}PASS import em')"
    python3 -c "import lerobot; assert lerobot.__version__=='${CLOUDGRIPPER_LEROBOT_VERSION}'; print('${prefix}PASS lerobot', lerobot.__version__)"
    ffmpeg -version | head -1 | grep -q ffmpeg && echo "${prefix}PASS ffmpeg"

    if [[ -f "${repo_root}/install/setup.bash" ]]; then
        cloudgripper_source_env "$repo_root" "$prefix" true || return 1
        ros2 pkg list | grep -qx cloudgripper_msgs && echo "${prefix}PASS cloudgripper_msgs"
        _cloudgripper_check_numpy_scipy "$prefix" || return 1
        python3 -c "import numpy, scipy; print('${prefix}PASS scipy', scipy.__version__, 'numpy', numpy.__version__)"
        python3 -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('${prefix}PASS LeRobotDataset')"
    else
        echo "${prefix}SKIP workspace build (run docker_setup.sh)"
    fi
    echo "${prefix}All checks passed."
}

# Ensure dataset_root exists and is writable by the current user (host mount permissions).
cloudgripper_ensure_dataset_writable() {
    local path="$1"
    local prefix="${2:-}"

    path="${path/#\~/$HOME}"
    if [[ "$path" != /* ]]; then
        path="$(cd "$(dirname "$path")" 2>/dev/null && pwd)/$(basename "$path")" || true
    fi

    if [[ -e "$path" ]]; then
        if [[ ! -w "$path" ]]; then
            local owner perm
            owner="$(stat -c '%U:%G' "$path" 2>/dev/null || echo unknown)"
            perm="$(stat -c '%a' "$path" 2>/dev/null || echo ?)"
            echo "${prefix}[ERROR] Dataset path not writable: ${path}" >&2
            echo "${prefix}        owner=${owner} mode=${perm}  uid=$(id -u)" >&2
            echo "${prefix}        Fix on host: sudo chown -R \$(id -u):\$(id -g) ${path}" >&2
            return 1
        fi
        return 0
    fi

    local parent
    parent="$(dirname "$path")"
    if [[ ! -d "$parent" ]] || [[ ! -w "$parent" ]]; then
        echo "${prefix}[ERROR] Cannot create dataset — parent not writable: ${parent}" >&2
        echo "${prefix}        Fix on host: sudo chown -R \$(id -u):\$(id -g) ${parent}" >&2
        return 1
    fi
    # Do not mkdir dataset_root — LeRobotDataset.create() requires a non-existent path.
}
