"""
data_collection.launch.py — full teleop data collection stack.

Session parameters live in:
  config/data_collection.yaml  — dataset, camera, robot, recorder, obs
  config/teleop.yaml           — teleop_node controls (SpaceMouse / keyboard)
  config/safety.yaml           — XYZ/RPY envelope + anti-爆冲 max_step_*

CLI launch args override a small subset (dataset path, episodes, dof, task,
push_to_hub, robot_ip, use_spacemouse).

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
_TELEOP_CONFIG = str(_REPO_ROOT / 'config' / 'teleop.yaml')
_SAFETY_CONFIG = str(_REPO_ROOT / 'config' / 'safety.yaml')


def _yaml_defaults(path: str) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get('/**', {}).get('ros__parameters', {})


def generate_launch_description():
    cfg = _yaml_defaults(_CONFIG)
    teleop_cfg = _yaml_defaults(_TELEOP_CONFIG)

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
            default_value=str(teleop_cfg.get('use_spacemouse', True)).lower()),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            parameters=[
                _CONFIG,
                # Explicit inline override: the /**-wildcard YAML does not reliably deliver
                # dot-notation sub-namespace keys (rgb_camera.*) to namespaced nodes.
                {'rgb_camera.color_profile': cfg.get('rgb_camera.color_profile', '640x480x15')},
            ],
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
                _SAFETY_CONFIG,
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
                    # Explicit overrides: ensure recorder feature shape matches the
                    # camera profile set above, bypassing /**-wildcard delivery issues.
                    'image_height':    cfg.get('image_height', 480),
                    'image_width':     cfg.get('image_width', 640),
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
                _TELEOP_CONFIG,
                _SAFETY_CONFIG,
                {
                    'dof_mode': dof_mode,
                    'use_spacemouse': use_spacemouse,
                    'enable_recording': True,
                },
            ],
            output='screen',
        ),
    ])
