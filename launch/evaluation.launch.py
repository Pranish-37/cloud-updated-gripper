"""
evaluation.launch.py — policy evaluation stack.

Starts: realsense2_camera + ae_awb_controller + xarm_driver +
        obs_aggregator + inference_node + episode_manager

Usage:
  ros2 launch launch/evaluation.launch.py
  ros2 launch launch/evaluation.launch.py policy_path:=/path/to/pretrained_model dof_mode:=3
"""

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = str(_REPO_ROOT / 'config' / 'evaluation.yaml')


def _yaml_defaults() -> dict:
    with open(_CONFIG) as f:
        data = yaml.safe_load(f)
    return data.get('/**', {}).get('ros__parameters', {})


def generate_launch_description():
    cfg = _yaml_defaults()

    policy_path        = LaunchConfiguration('policy_path')
    dataset_meta_path  = LaunchConfiguration('dataset_meta_path')
    eval_output_dir    = LaunchConfiguration('eval_output_dir')
    dof_mode           = LaunchConfiguration('dof_mode')
    robot_ip           = LaunchConfiguration('robot_ip')
    device             = LaunchConfiguration('device')

    return LaunchDescription([
        DeclareLaunchArgument(
            'policy_path',
            default_value=str(cfg.get('policy_path', '')),
        ),
        DeclareLaunchArgument(
            'dataset_meta_path',
            default_value=str(cfg.get('dataset_meta_path', '')),
        ),
        DeclareLaunchArgument(
            'eval_output_dir',
            default_value=str(cfg.get('eval_output_dir', '')),
        ),
        DeclareLaunchArgument(
            'dof_mode',
            default_value=str(cfg.get('dof_mode', 3)),
        ),
        DeclareLaunchArgument(
            'robot_ip',
            default_value=str(cfg.get('robot_ip', '192.168.0.244')),
        ),
        DeclareLaunchArgument(
            'device',
            default_value=str(cfg.get('device', 'cuda')),
        ),

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

        Node(
            package='obs_aggregator',
            executable='obs_aggregator_node',
            name='obs_aggregator_node',
            parameters=[
                _CONFIG,
                {'dof_mode': dof_mode},
            ],
            remappings=[
                ('camera/color/image', '/camera/camera/color/image_raw'),
            ],
            output='screen',
        ),

        Node(
            package='inference_node',
            executable='inference_node',
            name='inference_node',
            parameters=[
                _CONFIG,
                {
                    'policy_path': policy_path,
                    'dataset_meta_path': dataset_meta_path,
                    'device': device,
                },
            ],
            output='screen',
        ),

        Node(
            package='episode_manager',
            executable='episode_manager_node',
            name='episode_manager_node',
            parameters=[
                _CONFIG,
                {
                    'eval_output_dir': eval_output_dir,
                    'dof_mode': dof_mode,
                },
            ],
            output='screen',
        ),
    ])
