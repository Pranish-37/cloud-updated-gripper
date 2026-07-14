"""
data_collection.launch.py — full teleop data collection stack.

All session parameters live in config/data_collection.yaml.
This launch file only wires nodes together; CLI launch args override a
small subset (dataset path, episodes, dof, task, push_to_hub, robot_ip,
use_spacemouse).

Usage:
  ros2 launch launch/data_collection.launch.py
  ros2 launch launch/data_collection.launch.py total_episodes:=50 dof_mode:=3
  ros2 launch launch/data_collection.launch.py robot_ip:=192.168.0.245 use_spacemouse:=false
"""

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = str(_REPO_ROOT / 'config' / 'data_collection.yaml')


def _yaml_defaults() -> dict:
    with open(_CONFIG) as f:
        data = yaml.safe_load(f)
    return data.get('/**', {}).get('ros__parameters', {})


def generate_launch_description():
    cfg = _yaml_defaults()

    # CLI-overridable launch args (defaults from YAML)
    total_episodes  = LaunchConfiguration('total_episodes')
    dof_mode        = LaunchConfiguration('dof_mode')
    dataset_root    = LaunchConfiguration('dataset_root')
    dataset_repo_id = LaunchConfiguration('dataset_repo_id')
    task            = LaunchConfiguration('task')
    push_to_hub     = LaunchConfiguration('push_to_hub')
    robot_ip        = LaunchConfiguration('robot_ip')
    use_spacemouse  = LaunchConfiguration('use_spacemouse')

    return LaunchDescription([
        DeclareLaunchArgument(
            'total_episodes', default_value=str(cfg.get('total_episodes', 100))),
        DeclareLaunchArgument(
            'dof_mode', default_value=str(cfg.get('dof_mode', 3))),
        DeclareLaunchArgument(
            'dataset_root', default_value=str(cfg.get('dataset_root', ''))),
        DeclareLaunchArgument(
            'dataset_repo_id', default_value=str(cfg.get('dataset_repo_id', ''))),
        DeclareLaunchArgument(
            'task', default_value=str(cfg.get('task', ''))),
        DeclareLaunchArgument(
            'push_to_hub', default_value=str(cfg.get('push_to_hub', False)).lower()),
        DeclareLaunchArgument(
            'robot_ip', default_value=str(cfg.get('robot_ip', '192.168.0.244'))),
        DeclareLaunchArgument(
            'use_spacemouse',
            default_value=str(cfg.get('use_spacemouse', True)).lower()),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            parameters=[_CONFIG],
            output='screen',
        ),

        Node(
            package='camera_node',
            executable='ae_awb_controller_node',
            name='ae_awb_controller',
            parameters=[_CONFIG],
            output='screen',
        ),

        Node(
            package='xarm_driver',
            executable='xarm_driver_node',
            name='xarm_driver_node',
            parameters=[
                _CONFIG,
                {'robot_ip': robot_ip},
            ],
            output='screen',
        ),

        # realsense2_camera uses namespace+name 'camera', so the image topic is
        # /camera/camera/color/image_raw — remap to our canonical name.
        Node(
            package='obs_aggregator',
            executable='obs_aggregator_node',
            name='obs_aggregator_node',
            parameters=[
                _CONFIG,
                {'dof_mode': dof_mode},  # CLI override only
            ],
            remappings=[
                ('camera/color/image', '/camera/camera/color/image_raw'),
            ],
            output='screen',
        ),

        Node(
            package='recorder',
            executable='recorder_node',
            name='recorder_node',
            parameters=[
                _CONFIG,
                {
                    'dataset_root':    dataset_root,
                    'dataset_repo_id': dataset_repo_id,
                    'total_episodes':  total_episodes,
                    'dof_mode':        dof_mode,
                    'task':            task,
                    'push_to_hub':     push_to_hub,
                },
            ],
            output='screen',
        ),

        Node(
            package='teleop_node',
            executable='teleop_node',
            name='teleop_node',
            parameters=[
                _CONFIG,
                {'use_spacemouse': use_spacemouse},
            ],
            output='screen',
        ),
    ])
