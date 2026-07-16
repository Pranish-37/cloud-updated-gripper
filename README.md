# CloudGripper V2 ROS

ROS 2–based robotic manipulation system integrating teleoperated data collection and policy deployment within the LeRobot framework.


---

## Setup

Two paths depending on your OS:

- **Ubuntu 22.04**  [Native install](#native-ubuntu-2204)
- **Other / newer Ubuntu**  [Docker (recommended)](#docker-recommended)

---

### Docker (recommended)

Tested on Ubuntu 25.10 with Docker 29 + NVIDIA RTX 4090.

#### Prerequisites

On the **host**:

```bash
docker --version          # Docker installed
sudo docker info          # sudo access to Docker
git clone <repo-url> CloudGripper_V2_ROS
cd CloudGripper_V2_ROS
```

#### How the build is split

| Step | Where | What it does |
|------|-------|--------------|
| `docker build` | Host | Bakes ROS apt packages, Python deps, ffmpeg, **lerobot v0.4.4** into image `cloudgripper_ros:humble` |
| `docker_setup.sh` | Inside container | **`colcon build` only** — compiles this repo's 6 ROS packages into `build/` / `install/` on the mounted volume |

The host repo is mounted at `/workspace` every time you enter the container.  
`./scripts/docker_shell.sh` does **not** re-run the Dockerfile if the image already exists.

Shell scripts share one library: `scripts/lib/workspace_env.sh` (ROS source, conda PATH fix, lerobot 0.4.4).

#### First-time setup

##### Step 0 — (Optional) Start from a clean slate

Skip this on a truly fresh clone. Use it when re-testing the full flow:

```bash
# Exit any running container first (type 'exit' inside it)
sudo docker stop cloudgripper_ros 2>/dev/null || true
sudo docker rmi cloudgripper_ros:humble 2>/dev/null || true
rm -rf build install log
# Optional: remove host lerobot clone so runtime uses image /opt/lerobot (recommended)
rm -rf lerobot
```

> **Do not** keep an old `lerobot/` directory unless developing lerobot itself  
> (`export CLOUDGRIPPER_LEROBOT_DEV=1`). A stale clone can pull wrong Python deps.

##### Step 1 — Build the Docker image (host, run once)

```bash
cd /path/to/CloudGripper_V2_ROS
sudo docker build -t cloudgripper_ros:humble .
```

First build takes ~5 minutes. The image is cached locally until you delete it or change the Dockerfile.

**Verify on the host:**

```bash
# Image exists
sudo docker image inspect cloudgripper_ros:humble >/dev/null && echo "PASS: image exists"

# lerobot 0.4.4 is importable inside the image (no workspace mount needed)
sudo docker run --rm cloudgripper_ros:humble \
  python3 -c "import lerobot; print('lerobot', lerobot.__version__)"
# Expected output: lerobot 0.4.4
```

If the version is not `0.4.4`, rebuild the image (Step 0 + Step 1).

##### Step 2 — Build the colcon workspace (inside container, run once)

```bash
# Host — allow Docker GUI windows (needed later for teleop)
xhost +local:docker

# Host — enter container as your user (not root)
./scripts/docker_shell.sh
```

You should see lines like:

```text
[docker] Running as host user jin (1001:1001).
[docker] Workspace not built yet. Run:
         bash /workspace/scripts/docker_setup.sh
[docker] lerobot 0.4.4 (image /opt/lerobot)
[docker] ROS 2 humble ready. Workspace: /workspace
jin@agi:/workspace$
```

Inside the container:

```bash
bash /workspace/scripts/docker_setup.sh
```

This removes any old `build/` / `install/` / `log/` and runs `colcon build --symlink-install`.  
Root is **not** required.

**Verify inside the container:**

```bash
cv
# or: cloudgripper_verify_env /workspace
```

Expected: `PASS` lines for python3, em, lerobot 0.4.4, ffmpeg; after setup also cloudgripper_msgs + LeRobotDataset.

Type `exit` to leave the container. `build/` and `install/` remain on the host.

##### Step 3 — Every subsequent session

```bash
xhost +local:docker       # host, once per login session
./scripts/docker_shell.sh # no --root needed

# Workspace + lerobot are sourced automatically. Examples:
./scripts/launch_teleop.sh
./scripts/launch_data_collection.sh
```

On entry you should now see `[docker] Workspace sourced: /workspace/install/setup.bash`.

> **No GPU needed** for teleop / camera / robot control.  
> GPU passthrough (`--gpus all`) is enabled automatically if nvidia-container-toolkit is present.  
> Disable with `./scripts/docker_shell.sh --no-gpu`.

#### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Permission denied` on `build/` or `lerobot/` | Previously ran setup as root. On host: `sudo chown -R $(id -u):$(id -g) build install log lerobot` |
| `lerobot X.X.X — expected 0.4.4` | Rebuild image: Step 0 + Step 1 |
| `ModuleNotFoundError: No module named 'em'` | Old image had conda python first in PATH — rebuild image (Step 0 + Step 1) |
| `numpy` / `scipy` / `_ARRAY_API` in recorder | numpy 2.x vs apt scipy — rebuild image; remove host `lerobot/` dir |
| `No RealSense devices were found` | Camera not plugged in or USB not passed to container (check `lsusb`, `--privileged`) |
| Changed `Dockerfile` but behavior unchanged | `docker_shell.sh` reuses old image — run Step 0 + Step 1 |
| `cloudgripper_ros` container name in use | `sudo docker stop cloudgripper_ros` then retry |

---

### Native (Ubuntu 22.04)

#### 1. ROS 2

Install ROS 2 (Humble or later): https://docs.ros.org/en/humble/Installation.html

Then install required ROS 2 packages:

```bash
sudo apt install -y \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-std-srvs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-rcl-interfaces \
    ros-${ROS_DISTRO}-realsense2-camera \
    ros-${ROS_DISTRO}-launch-ros \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-rosidl-default-runtime
```

> On Linux with SpaceMouse, also install the spacenav daemon:
> ```bash
> sudo apt install -y libspnav-dev spacenavd
> sudo systemctl enable spacenavd --now
> ```

#### 2. Python packages + lerobot

```bash
pip install -r requirements.txt
git clone --depth 1 --branch v0.4.4 https://github.com/huggingface/lerobot.git
pip install -e lerobot
touch lerobot/COLCON_IGNORE
```

#### 3. Build the workspace

```bash
cd /path/to/CloudGripper_V2_ROS
colcon build --symlink-install
source install/setup.bash
```

---

## Package Overview

| Package | Type | Description |
|---------|------|-------------|
| `cloudgripper_msgs` | CMake | Custom message definitions (`ObsFrame`) |
| `xarm_driver` | Python | Robot driver — publishes state, executes actions |
| `camera_node` | Python | RealSense AE/AWB lock controller |
| `obs_aggregator` | Python | Assembles 5 Hz `ObsFrame` from robot + camera topics |
| `recorder` | Python | LeRobot v3 dataset writer + episode services |
| `teleop_node` | Python | Keyboard + SpaceMouse teleoperation |

---

## Teleop

```bash
./scripts/launch_teleop.sh                        # default IP, SpaceMouse
./scripts/launch_teleop.sh --ip 192.168.0.245     # custom robot IP
./scripts/launch_teleop.sh --no-spacemouse        # keyboard-only rotation
```

Controls:

| Key | Action |
|-----|--------|
| W / S | EE +x / -x |
| A / D | EE +y / -y |
| Space / Ctrl | EE +z / -z |
| SpaceMouse cap | roll / pitch / yaw |
| J/L  I/K  U/O | roll / pitch / yaw (keyboard mode) |
| Q | gripper open |
| E | gripper close |
| Esc | stop |

---

## Data collection

```bash
./scripts/launch_data_collection.sh
./scripts/launch_data_collection.sh --episodes 50 --dof 3
./scripts/launch_data_collection.sh --ip 192.168.0.245 --no-spacemouse
```

Session config: `config/data_collection.yaml`

---

## ROS Topic Map

```
xarm_driver   publish: robot/joint_states      sensor_msgs/JointState
              publish: robot/ee_pose           geometry_msgs/PoseStamped
              publish: robot/gripper_pos       std_msgs/Float64
              subscribe: policy/action_cmd     std_msgs/Float64MultiArray

obs_aggregator subscribe: robot/* + camera/color/image
               publish: observation/frame      cloudgripper_msgs/ObsFrame

recorder      subscribe: observation/frame
              services: recorder/toggle_recording, save_episode, ...

teleop_node   publish: policy/action_cmd
              subscribe: robot/ee_pose, robot/gripper_pos
```

All topic names are relative — prefix with a namespace for multi-robot setups  
(e.g. `/robot_1/robot/joint_states`).

---

## Hardware config (default)

| Component | Value |
|-----------|-------|
| Robot | xArm7 |
| Robot IP | `192.168.0.244` (see `config/data_collection.yaml`) |
| Camera | Intel RealSense (serial `843112070951`) |
| Camera resolution | 1920 × 1080 RGB @ 15 Hz |
| Control FPS | 5 Hz (obs/recording), 50 Hz (teleop) |
