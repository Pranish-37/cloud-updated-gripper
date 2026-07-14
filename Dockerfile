FROM osrf/ros:humble-desktop

# Install ROS 2 packages + system utilities
RUN apt-get update -q && apt-get install -y -q \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-std-srvs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-rcl-interfaces \
    ros-humble-realsense2-camera \
    ros-humble-launch-ros \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    libspnav-dev \
    libhidapi-hidraw0 \
    libhidapi-libusb0 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --quiet -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Miniforge (conda) — needed for ffmpeg=7.1.1 from conda-forge
# ffmpeg 8.x is not yet supported by lerobot
# ---------------------------------------------------------------------------
RUN wget -q \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install -y -c conda-forge "ffmpeg=7.1.1" \
    && conda clean -afy

# ---------------------------------------------------------------------------
# lerobot — installed from source into the image as a base install.
# /workspace/lerobot (host-mounted) takes priority at runtime if present.
# ---------------------------------------------------------------------------
RUN git clone --depth 1 --branch v0.4.4 https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip3 install --upgrade pip \
    && pip3 install --ignore-installed "sympy>=1.13" \
    && pip3 install -e /opt/lerobot \
    && pip3 cache purge

WORKDIR /workspace
