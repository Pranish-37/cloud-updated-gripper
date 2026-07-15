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
    python3-empy \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --quiet -r /tmp/requirements.txt

# Miniforge — ffmpeg=7.1.1 only. Do NOT prepend conda to PATH (breaks rosidl/em).
RUN wget -q \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy \
    && /opt/conda/bin/conda install -y -c conda-forge "ffmpeg=7.1.1" \
    && ln -sf /opt/conda/bin/ffmpeg /usr/local/bin/ffmpeg \
    && /opt/conda/bin/conda clean -afy

RUN python3 -c "import em" && ffmpeg -version | head -1

# lerobot v0.4.4 — install deps first, then pip scipy (apt scipy is numpy-1.x only).
RUN git clone --depth 1 --branch v0.4.4 https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip3 install --upgrade pip \
    && pip3 install --ignore-installed "sympy>=1.13" \
    && pip3 install -e /opt/lerobot \
    && pip3 install "scipy==1.15.3" "setuptools>=71.0.0,<80.0.0" \
    && pip3 cache purge \
    && python3 -c "import lerobot; assert lerobot.__version__=='0.4.4', lerobot.__version__" \
    && python3 -c "import numpy, scipy; assert '/usr/lib/python3/dist-packages' not in scipy.__file__, scipy.__file__; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

WORKDIR /workspace
