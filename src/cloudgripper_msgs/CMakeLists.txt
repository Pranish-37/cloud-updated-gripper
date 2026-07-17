cmake_minimum_required(VERSION 3.8)
project(cloudgripper_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObsFrame.msg"
  "srv/PredictActionChunk.srv"
  "srv/ExecuteAction.srv"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_package()
