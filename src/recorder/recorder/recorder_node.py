"""
recorder_node — LeRobot dataset writer for CloudGripper V2 teleop data collection.
Targets lerobot v0.4.4 (last release supporting Python 3.10).

Subscribes:
  observation/frame    cloudgripper_msgs/ObsFrame   (from obs_aggregator)

Services (all std_srvs/Trigger):
  recorder/toggle_recording  — R key: start or pause buffering within current episode
  recorder/save_episode      — N key: save current buffer, episode_index +1, start new episode
  recorder/discard_episode   — D key: discard current buffer, episode_index unchanged
  recorder/finalize_dataset  — ESC : write meta/info.json stats, shutdown node

Parameters:
  dataset_root          (string)  — absolute or relative path to dataset directory
  dataset_repo_id       (string)  — e.g. "YourName/dataset-name"
  total_episodes        (int)     — target number of episodes to collect
  fps                   (float,  default 5.0)
  dof_mode              (int,    default 3)  — 3 or 6
  task                  (string, default "")  — task description written per episode
  image_writer_threads  (int,    default 4)

Action labeling (next-state supervision):
  action[t] = state[t+1][7:]
    3-DoF: state[7:11] = [ee_x, ee_y, ee_z, gripper]            → action (4,)
    6-DoF: state[7:14] = [ee_x, ee_y, ee_z, roll, pitch, yaw, gripper] → action (7,)

  The last frame of every episode has no next-state label and is silently dropped.

Dataset create vs resume:
  - If dataset_root/meta/info.json exists AND num_episodes < total_episodes → resume
  - If dataset_root/meta/info.json exists AND num_episodes >= total_episodes → log, do nothing
  - Otherwise → create new dataset
"""

from pathlib import Path
import shutil

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from cloudgripper_msgs.msg import ObsFrame


def _build_features(dof: int, img_h: int, img_w: int) -> dict:
    """Construct the LeRobot v3 features dict matching xarm_lerobot.py conventions."""
    if dof == 3:
        state_names = [
            'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
            'ee_x', 'ee_y', 'ee_z', 'gripper',
        ]
        action_names = ['ee_x', 'ee_y', 'ee_z', 'gripper']
        action_shape = (4,)
        state_shape = (11,)
    elif dof == 6:
        state_names = [
            'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
            'ee_x', 'ee_y', 'ee_z', 'ee_roll', 'ee_pitch', 'ee_yaw', 'gripper',
        ]
        action_names = ['ee_x', 'ee_y', 'ee_z', 'ee_roll', 'ee_pitch', 'ee_yaw', 'gripper']
        action_shape = (7,)
        state_shape = (14,)
    else:
        raise ValueError(f'dof_mode must be 3 or 6, got {dof}')

    return {
        'observation.state': {
            'dtype': 'float32',
            'shape': state_shape,
            'names': state_names,
        },
        'observation.images.front': {
            'dtype': 'video',
            'shape': (img_h, img_w, 3),
            'names': ['height', 'width', 'channels'],
        },
        'action': {
            'dtype': 'float32',
            'shape': action_shape,
            'names': action_names,
        },
    }


class RecorderNode(Node):
    def __init__(self):
        super().__init__('recorder_node')

        # --- parameters ---
        self.declare_parameter('dataset_root', '')
        self.declare_parameter('dataset_repo_id', 'CloudGripper/dataset')
        self.declare_parameter('total_episodes', 100)
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('dof_mode', 3)
        self.declare_parameter('task', '')
        self.declare_parameter('image_writer_threads', 4)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('push_to_hub', False)

        def _s(n): return self.get_parameter(n).get_parameter_value().string_value
        def _i(n): return self.get_parameter(n).get_parameter_value().integer_value
        def _f(n): return self.get_parameter(n).get_parameter_value().double_value
        def _b(n): return self.get_parameter(n).get_parameter_value().bool_value

        dataset_root_str = _s('dataset_root')
        if not dataset_root_str:
            raise RuntimeError("Parameter 'dataset_root' must be set.")

        self._repo_id        = _s('dataset_repo_id')
        self._total_episodes = _i('total_episodes')
        self._fps            = _f('fps')
        self._dof            = _i('dof_mode')
        self._task           = _s('task')
        self._img_threads    = _i('image_writer_threads')
        self._img_h          = _i('image_height')
        self._img_w          = _i('image_width')
        self._push_to_hub    = _b('push_to_hub')

        self._dataset_root = Path(dataset_root_str).expanduser().resolve()

        # --- internal state ---
        self._recording = False # True = actively buffering frames
        self._pending_frame: dict | None = None  # last received frame, awaiting next-state label
        self._dataset = None # LeRobotDataset instance
        self._ready = False # False until dataset is loaded and episodes remain

        # --- build / resume dataset ---
        self._init_dataset()

        # --- subscriber ---
        self.create_subscription(ObsFrame, 'observation/frame', self._on_obs_frame, 1)

        # --- services ---
        self.create_service(Trigger, 'recorder/toggle_recording', self._svc_toggle_recording)
        self.create_service(Trigger, 'recorder/save_episode',     self._svc_save_episode)
        self.create_service(Trigger, 'recorder/discard_episode',  self._svc_discard_episode)
        self.create_service(Trigger, 'recorder/finalize_dataset', self._svc_finalize_dataset)

        self.get_logger().info(
            f'RecorderNode ready — {self._dof}-DoF, fps={self._fps}, '
            f'episodes {self._dataset.num_episodes}/{self._total_episodes}, '
            f'dataset: {self._dataset_root}'
        )
        if self._ready:
            self.get_logger().info(
                'Episode ready — call recorder/toggle_recording to start buffering.'
            )

    # ------------------------------------------------------------------
    # Dataset init (create / resume)
    # ------------------------------------------------------------------

    def _has_pending_frames(self) -> bool:
        if self._dataset is None or self._dataset.episode_buffer is None:
            return False
        return self._dataset.episode_buffer.get('size', 0) > 0

    def _has_episode_metadata(self) -> bool:
        """True when meta/episodes contains at least one parquet file."""
        episodes_dir = self._dataset_root / 'meta' / 'episodes'
        if not episodes_dir.is_dir():
            return False
        return any(episodes_dir.rglob('*.parquet'))

    def _dataset_is_resumable(self) -> bool:
        """True when a locally recorded dataset has at least one fully saved episode."""
        meta = self._dataset_root / 'meta'
        return (
            (meta / 'info.json').exists()
            and (meta / 'tasks.parquet').exists()
            and self._has_episode_metadata()
        )

    def _clear_incomplete_dataset(self) -> None:
        """Remove a crashed/empty dataset that only has partial metadata."""
        if not (self._dataset_root / 'meta' / 'info.json').exists():
            return
        if self._dataset_is_resumable():
            return
        self.get_logger().warn(
            f'Incomplete or corrupted dataset at {self._dataset_root} '
            f'— removing and starting fresh.'
        )
        shutil.rmtree(self._dataset_root)

    def _fix_dataset_fps(self, dataset) -> None:
        """PyAV video encoding requires integer fps; JSON stores it as float."""
        dataset.meta.info['fps'] = int(round(dataset.meta.info['fps']))

    def _init_dataset(self) -> None:
        # Import here so lerobot is only required at runtime, not at import time.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self._dataset_is_resumable():
            # v0.4.4: load existing dataset via constructor, then enable writers.
            dataset = LeRobotDataset(
                repo_id=self._repo_id,
                root=self._dataset_root,
            )
            dataset.start_image_writer(0, self._img_threads)
            n = dataset.num_episodes
            self.get_logger().info(
                f'Resumed dataset: {self._dataset_root}  ({n} episodes already on disk)'
            )
            if n >= self._total_episodes:
                self.get_logger().warn(
                    f'Dataset already has {n}/{self._total_episodes} episodes. '
                    f'Nothing to record. Increase total_episodes or use a different dataset_root.'
                )
                self._fix_dataset_fps(dataset)
                self._dataset = dataset
                self._ready = False
                return
        else:
            self._clear_incomplete_dataset()
            features = _build_features(self._dof, self._img_h, self._img_w)
            dataset = LeRobotDataset.create(
                repo_id=self._repo_id,
                root=self._dataset_root,
                fps=int(round(self._fps)),
                features=features,
                robot_type='xarm',
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=self._img_threads,
            )
            self.get_logger().info(f'Created new dataset: {self._dataset_root}')

        self._fix_dataset_fps(dataset)
        self._dataset = dataset
        self._ready = True

    # ------------------------------------------------------------------
    # Frame subscription — action labeling via next-state supervision
    # ------------------------------------------------------------------

    def _on_obs_frame(self, msg: ObsFrame) -> None:
        if not self._recording or not self._ready:
            return

        state = np.array(msg.state, dtype=np.float32)
        # action[t] = state[t+1][7:]  — works for both DoF modes without branching:
        #   3-DoF: state (11,) → state[7:] = [ee_x, ee_y, ee_z, gripper]        shape (4,)
        #   6-DoF: state (14,) → state[7:] = [ee_x, ee_y, ee_z, roll, pitch, yaw, gripper] shape (7,)
        # The first 7 elements are always joint angles; everything after is the action vector.
        action = state[7:].copy()

        # Convert RGB image message to numpy (obs_aggregator already outputs rgb8)
        img = np.frombuffer(msg.image.data, dtype=np.uint8).reshape(
            msg.image.height, msg.image.width, 3
        ).copy()

        # Label the pending frame with the current state's action component, then commit it
        if self._pending_frame is not None:
            frame = self._pending_frame
            frame['action'] = action
            frame['task'] = self._task
            self._dataset.add_frame(frame)

        # Park current frame as pending (will be labeled when next frame arrives)
        self._pending_frame = {
            'observation.state': state,
            'observation.images.front': img,
            'task': self._task,
            # 'action' filled in on next callback
        }

    # ------------------------------------------------------------------
    # Service handlers
    # ------------------------------------------------------------------

    def _svc_toggle_recording(self, _req, res: Trigger.Response) -> Trigger.Response:
        if not self._ready:
            res.success = False
            res.message = 'Dataset is full or not initialized.'
            return res

        self._recording = not self._recording
        if self._recording:
            self.get_logger().info(
                f'[Episode {self._dataset.num_episodes}] Recording STARTED.'
            )
            res.message = 'Recording started.'
        else:
            self.get_logger().info(
                f'[Episode {self._dataset.num_episodes}] Recording PAUSED.'
            )
            res.message = 'Recording paused.'
        res.success = True
        return res

    def _svc_save_episode(self, _req, res: Trigger.Response) -> Trigger.Response:
        if not self._ready:
            res.success = False
            res.message = 'Dataset is full or not initialized.'
            return res

        self._recording = False
        self._pending_frame = None  # drop last unlabeled frame

        if self._has_pending_frames():
            self._fix_dataset_fps(self._dataset)
            self._dataset.save_episode()
            n = self._dataset.num_episodes
            self.get_logger().info(
                f'Episode saved. {n}/{self._total_episodes}'
            )
            res.message = f'Episode saved ({n}/{self._total_episodes}).'
            res.success = True

            if n >= self._total_episodes:
                self.get_logger().info('Target episode count reached. Finalizing dataset...')
                self._finalize_and_shutdown()
        else:
            self.get_logger().warn(
                'save_episode called but buffer is empty — episode discarded.'
            )
            self._dataset.clear_episode_buffer()
            res.success = False
            res.message = 'Buffer was empty; episode discarded. Press R to start recording.'

        return res

    def _svc_discard_episode(self, _req, res: Trigger.Response) -> Trigger.Response:
        self._recording = False
        self._pending_frame = None
        if self._has_pending_frames():
            self._dataset.clear_episode_buffer()
        n = self._dataset.num_episodes if self._dataset else 0
        self.get_logger().info(
            f'Episode discarded. Restarting episode {n} (index unchanged).'
        )
        res.success = True
        res.message = f'Episode {n} discarded. Press R to start recording again.'
        return res

    def _svc_finalize_dataset(self, _req, res: Trigger.Response) -> Trigger.Response:
        self._finalize_and_shutdown()
        res.success = True
        res.message = 'Dataset finalized. Node shutting down.'
        return res

    # ------------------------------------------------------------------
    # Finalize helper
    # ------------------------------------------------------------------

    def _finalize_and_shutdown(self) -> None:
        self._recording = False
        self._pending_frame = None

        if self._dataset is None:
            rclpy.shutdown()
            return

        if self._has_pending_frames():
            self.get_logger().info('Saving in-progress episode before finalizing...')
            self._fix_dataset_fps(self._dataset)
            self._dataset.save_episode()

        self.get_logger().info(
            f'Finalizing dataset — {self._dataset.num_episodes} episodes total.'
        )
        self._dataset.finalize()
        self.get_logger().info('Dataset finalized.')

        if self._push_to_hub:
            self.get_logger().info(f'Uploading dataset to HuggingFace Hub: {self._repo_id} ...')
            self._dataset.push_to_hub()
            self.get_logger().info('Upload complete.')
        else:
            self.get_logger().info('push_to_hub=false — dataset saved locally only.')

        self.get_logger().info('Shutting down recorder node.')
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self.get_logger().info('RecorderNode shutting down.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
