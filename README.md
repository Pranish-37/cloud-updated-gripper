# CloudGripper V2 ROS

ROS 2 port of the LeRobot xArm7 pick-and-place pipeline  
(xArm7 + Intel RealSense, ACT / Diffusion / π₀ policies).

---

## Setup

Two paths depending on your OS:

- **Ubuntu 22.04**  [Native install](#native-ubuntu-2204)
- **Other / newer Ubuntu**  [Docker (recommended)](#docker-recommended)

---

### Docker (recommended)

Tested on Ubuntu 25.10 with Docker 29 + NVIDIA RTX 4090.  
Requires Docker installed (`docker --version`) and `sudo` access.

**Step 1 — Build the image (run once)**

All ROS 2 apt packages and Python dependencies are baked into the image via `Dockerfile`.

```bash
sudo docker build -t cloudgripper_ros:humble .
```

This takes ~5 minutes on first build. The image is cached locally and survives container restarts.

**Step 2 — Build the colcon workspace (run once)**

```bash
# Allow Docker to open GUI windows (needed for teleop keyboard input)
xhost +local:docker

# Enter the container
./scripts/docker_shell.sh

# Inside the container: build all 5 ROS packages
bash /workspace/scripts/docker_setup.sh
```

**Step 3 — Every subsequent session**

```bash
xhost +local:docker       # on the host (once per login session)
./scripts/docker_shell.sh # enter container

# ROS + workspace are sourced automatically. Run teleop:
./scripts/launch_teleop.sh
```

> **No GPU needed for teleop/camera/robot control.**  
> GPU passthrough (`--gpus all`) is enabled automatically if nvidia-container-toolkit is present.  
> Disable with `./scripts/docker_shell.sh --no-gpu`.

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

#### 2. Python packages

```bash
pip install -r requirements.txt
```

#### 3. Build the workspace

```bash
cd /path/to/CloudGripper_V2_ROS
colcon build
source install/setup.bash
```

---

## Package Overview

| Package | Type | Description |
|---------|------|-------------|
| `cloudgripper_msgs` | CMake | Custom message definitions (`ObsFrame`) |
| `xarm_driver` | Python | Robot driver node — publishes state, executes actions |
| `camera_node` | Python | RealSense camera node + AE/AWB lock service |
| `obs_aggregator` | Python | Assembles 5 Hz `ObsFrame` from robot + camera topics |
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

## ROS Topic Map

```
xarm_driver publish: robot/joint_states      sensor_msgs/JointState
            publish: robot/ee_pose           geometry_msgs/PoseStamped
            publish: robot/gripper_pos       std_msgs/Float64

camera_node publish: camera/color/image_raw  sensor_msgs/Image

obs_aggregator subscibe: robot/* + camera/*
               publish: observation/frame    cloudgripper_msgs/ObsFrame

teleop_node publish: action_cmd       std_msgs/Float64MultiArray
xarm_driver subscribe: policy/action_cmd
```

All topic names are relative — prefix with a namespace for multi-robot setups  
(e.g. `/robot_1/robot/joint_states`).

---

## Hardware config (default)

| Component | Value |
|-----------|-------|
| Robot | xArm7 |
| Robot IP | `192.168.0.244` |
| Camera | Intel RealSense (serial `843112070951`) |
| Camera resolution | 640 × 480 RGB @ 15 Hz |
| Control FPS | 5 Hz (obs/inference), 50 Hz (teleop) |
