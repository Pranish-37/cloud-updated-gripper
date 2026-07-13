FROM osrf/ros:humble-desktop

# Install ROS 2 packages
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
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --quiet -r /tmp/requirements.txt

WORKDIR /workspace
