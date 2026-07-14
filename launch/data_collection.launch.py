"""
data_collection.launch.py — full teleop data collection stack.

Nodes launched:
  1. realsense2_camera        — RGB stream @ color_fps Hz (via camera_node)
  2. ae_awb_controller        — AE/AWB lock after convergence
  3. xarm_driver_node         — robot driver (servo mode for teleop)
  4. obs_aggregator_node      — assembles ObsFrame at fps Hz
  5. recorder_node            — writes LeRobot v3 dataset
  6. teleop_node              — keyboard + SpaceMouse control + recording key bindings

Usage:
  # 1. Edit config/data_collection.yaml (dataset path, DoF, total_episodes, etc.)
  # 2. Build the workspace:  cd ~/ros2_ws && colcon build --symlink-install
  # 3. Source:               source install/setup.bash
  # 4. Launch:
  ros2 launch cloudgripper_ros2 data_collection.launch.py

Override any YAML parameter at the command line, e.g.:
  ros2 launch cloudgripper_ros2 data_collection.launch.py total_episodes:=50 dof_mode:=3
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Path to config file (relative to this launch file's repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = str(_REPO_ROOT / 'config' / 'data_collection.yaml')


def generate_launch_description():
    # -----------------------------------------------------------------------
    # Overridable launch arguments (mirror the YAML keys you most often change)
    # -----------------------------------------------------------------------
    total_episodes  = LaunchConfiguration('total_episodes')
    dof_mode        = LaunchConfiguration('dof_mode')
    dataset_root    = LaunchConfiguration('dataset_root')
    dataset_repo_id = LaunchConfiguration('dataset_repo_id')
    task            = LaunchConfiguration('task')

    return LaunchDescription([
        # --- overridable args (defaults come from YAML, these just expose them) ---
        DeclareLaunchArgument('total_episodes',  default_value='100'),
        DeclareLaunchArgument('dof_mode',        default_value='6'),
        DeclareLaunchArgument('dataset_root',    default_value='/tmp/cloudgripper_datasets/my_dataset'),
        DeclareLaunchArgument('dataset_repo_id', default_value='CloudGripper/my-dataset'),
        DeclareLaunchArgument('task',            default_value=''),

        # -----------------------------------------------------------------------
        # 1a. realsense2_camera — full parameter set from YAML
        # -----------------------------------------------------------------------
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            parameters=[
                _CONFIG,
                {
                    'serial_no':                  '843112070951',
                    'enable_color':               True,
                    'enable_depth':               False,
                    'enable_infra1':              False,
                    'enable_infra2':              False,
                    # stream geometry — read from YAML
                    # 'color_width', 'color_height', 'color_fps', 'color_format'
                    # are loaded via _CONFIG above
                    # AE/AWB initial state — loaded via _CONFIG
                    # Manual override values — loaded via _CONFIG
                },
            ],
            output='screen',
        ),

        # -----------------------------------------------------------------------
        # 1b. AE/AWB controller — call /camera/lock_exposure after convergence
        # -----------------------------------------------------------------------
        Node(
            package='camera_node',
            executable='ae_awb_controller_node',
            name='ae_awb_controller',
            parameters=[{'realsense_node_name': 'camera'}],
            output='screen',
        ),

        # -----------------------------------------------------------------------
        # 2. xarm_driver_node — servo mode for low-latency teleop
        # -----------------------------------------------------------------------
        Node(
            package='xarm_driver',
            executable='xarm_driver_node',
            name='xarm_driver_node',
            parameters=[
                _CONFIG,
                {'servo_mode': True},
            ],
            output='screen',
        ),

        # -----------------------------------------------------------------------
        # 3. obs_aggregator_node
        # -----------------------------------------------------------------------
        Node(
            package='obs_aggregator',
            executable='obs_aggregator_node',
            name='obs_aggregator_node',
            parameters=[
                _CONFIG,
                {
                    'publish_rate': 5.0,
                    'dof_mode':     dof_mode,
                },
            ],
            output='screen',
        ),

        # -----------------------------------------------------------------------
        # 4. recorder_node
        # -----------------------------------------------------------------------
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
                },
            ],
            output='screen',
        ),

        # -----------------------------------------------------------------------
        # 5. teleop_node — recording mode enabled
        # -----------------------------------------------------------------------
        Node(
            package='teleop_node',
            executable='teleop_node',
            name='teleop_node',
            parameters=[
                _CONFIG,
                {'enable_recording': True},
            ],
            output='screen',
        ),
    ])
