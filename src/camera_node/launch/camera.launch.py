"""
camera.launch.py — launches realsense2_camera + optional ae_awb_controller_node.

Args (launch arguments):
  camera_serial      default '843112070951'
  camera_name        default 'camera'     — becomes the realsense2_camera node name
                                            and the namespace prefix for its topics
  color_width        default 640
  color_height       default 480
  color_fps          default 15.0
  enable_ae_awb_lock default 'true'       — set 'false' to skip the controller node
                                            (AE/AWB will stay in auto mode)

Multi-robot usage: include this file inside a GroupAction + PushRosNamespace block
in the top-level full_system.launch.py. All nodes inherit the parent namespace, so
topics are scoped automatically (e.g. /robot_1/camera/color/image_raw).
Standalone launch (single robot, no namespace):
  ros2 launch camera_node camera.launch.py camera_serial:=843112070951
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_serial      = LaunchConfiguration('camera_serial')
    camera_name        = LaunchConfiguration('camera_name')
    color_width        = LaunchConfiguration('color_width')
    color_height       = LaunchConfiguration('color_height')
    color_fps          = LaunchConfiguration('color_fps')
    enable_ae_awb_lock = LaunchConfiguration('enable_ae_awb_lock')

    return LaunchDescription([
        DeclareLaunchArgument('camera_serial', default_value='843112070951',
                              description='RealSense device serial number'),
        DeclareLaunchArgument('camera_name', default_value='camera',
                              description='Node/topic name prefix for realsense2_camera'),
        DeclareLaunchArgument('color_width',  default_value='640',
                              description='Color stream width in pixels'),
        DeclareLaunchArgument('color_height', default_value='480',
                              description='Color stream height in pixels'),
        DeclareLaunchArgument('color_fps',    default_value='15.0',
                              description='Color stream frame rate'),
        DeclareLaunchArgument('enable_ae_awb_lock', default_value='true',
                              description='Launch AE/AWB controller node (true/false)'),

        # realsense2_camera node — always launched
        # Topics published (relative, scoped by namespace):
        #   camera/color/image_raw   sensor_msgs/Image  BGR8  @ color_fps Hz
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name=camera_name,
            namespace=camera_name,
            parameters=[{
                'serial_no':                  camera_serial,
                'enable_color':               True,
                'enable_depth':               False,
                'enable_infra1':              False,
                'enable_infra2':              False,
                'color_width':                color_width,
                'color_height':               color_height,
                'color_fps':                  color_fps,
                'color_auto_exposure':        True,
                'color_auto_white_balance':   True,
            }],
            output='screen',
        ),

        # AE/AWB controller — optional, skipped when enable_ae_awb_lock:=false
        Node(
            package='camera_node',
            executable='ae_awb_controller_node',
            name='ae_awb_controller',
            parameters=[{'realsense_node_name': camera_name}],
            output='screen',
            condition=IfCondition(enable_ae_awb_lock), # skip the controller node if enable_ae_awb_lock is False
        ),
    ])
