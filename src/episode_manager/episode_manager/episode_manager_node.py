"""
episode_manager — real-world policy evaluation control loop.

Subscribes:
  observation/frame    cloudgripper_msgs/ObsFrame

Publishes:
  policy/action_cmd    std_msgs/Float64MultiArray   (7-vector robot command)

Clients:
  inference/predict_action_chunk   cloudgripper_msgs/PredictActionChunk
  inference/reset                  std_srvs/Trigger
  robot/move_home                  std_srvs/Trigger

Logs (eval_output_dir):
  actions.jsonl          — one 7-vector action per control step
  episode_dataset.json   — per-episode summary list

Interactive stdin flow (per episode):
  1. Place cube, press Enter to start
  2. Reset policy, run up to max_steps_per_episode at fps
  3. Prompt success (y/n)
  4. Move robot home
  5. Prompt continue (y/n) or quit
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from cloudgripper_msgs.msg import ObsFrame
from cloudgripper_msgs.srv import PredictActionChunk


def _prompt_yes_no(prompt: str, default: bool | None = None) -> bool:
    suffix = ' [y/n]: '
    if default is True:
        suffix = ' [Y/n]: '
    elif default is False:
        suffix = ' [y/N]: '
    while True:
        answer = input(f'{prompt}{suffix}').strip().lower()
        if not answer and default is not None:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print('Please answer y or n.')


def _prompt_enter(message: str) -> None:
    input(f'{message} [Enter]')


def _pad_rpy(action: list[float], rpy: list[float]) -> list[float]:
    """Pad 4-vec [x,y,z,gripper] to 7-vec robot command."""
    return [action[0], action[1], action[2], rpy[0], rpy[1], rpy[2], action[3]]


class EpisodeManagerNode(Node):
    def __init__(self):
        super().__init__('episode_manager')

        self.declare_parameter('eval_output_dir', '')
        self.declare_parameter('dof_mode', 3)
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('chunk_size', 16)
        self.declare_parameter('n_action_steps', 8)
        self.declare_parameter('max_steps_per_episode', 90)
        self.declare_parameter('default_rpy', [math.pi, 0.0, 0.0])

        eval_output_dir = self.get_parameter('eval_output_dir').get_parameter_value().string_value
        if not eval_output_dir:
            raise RuntimeError("Parameter 'eval_output_dir' must be set.")

        self._dof = self.get_parameter('dof_mode').get_parameter_value().integer_value
        self._fps = self.get_parameter('fps').get_parameter_value().double_value
        self._chunk_size = self.get_parameter('chunk_size').get_parameter_value().integer_value
        self._n_action_steps = self.get_parameter('n_action_steps').get_parameter_value().integer_value
        self._max_steps = self.get_parameter('max_steps_per_episode').get_parameter_value().integer_value
        self._default_rpy = list(
            self.get_parameter('default_rpy').get_parameter_value().double_array_value
        )

        if self._dof not in (3, 6):
            raise ValueError(f'dof_mode must be 3 or 6, got {self._dof}')

        self._output_dir = Path(eval_output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._actions_path = self._output_dir / 'actions.jsonl'
        self._episodes_path = self._output_dir / 'episode_dataset.json'

        self._latest_obs: ObsFrame | None = None
        self._episode_index = 0
        self._episodes: list[dict] = []
        self._load_episode_summaries()

        self.create_subscription(ObsFrame, 'observation/frame', self._on_obs_frame, 1)
        self._pub_action = self.create_publisher(Float64MultiArray, 'policy/action_cmd', 1)

        self._predict_client = self.create_client(
            PredictActionChunk, 'inference/predict_action_chunk',
        )
        self._reset_client = self.create_client(Trigger, 'inference/reset')
        self._home_client = self.create_client(Trigger, 'robot/move_home')

        self.get_logger().info(
            f'EpisodeManager ready — {self._dof}-DoF, fps={self._fps}, '
            f'max_steps={self._max_steps}, output={self._output_dir}'
        )

    def _load_episode_summaries(self) -> None:
        if self._episodes_path.exists():
            with open(self._episodes_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                self._episodes = data
                self._episode_index = len(data)

    def _save_episode_summaries(self) -> None:
        with open(self._episodes_path, 'w') as f:
            json.dump(self._episodes, f, indent=2)

    def _append_action_log(self, record: dict) -> None:
        with open(self._actions_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def _on_obs_frame(self, msg: ObsFrame) -> None:
        self._latest_obs = msg

    def _wait_for_services(self, timeout_sec: float = 30.0) -> bool:
        deadline = time.time() + timeout_sec
        clients = [
            ('inference/predict_action_chunk', self._predict_client),
            ('inference/reset', self._reset_client),
            ('robot/move_home', self._home_client),
        ]
        for name, client in clients:
            while not client.wait_for_service(timeout_sec=1.0):
                if time.time() >= deadline:
                    self.get_logger().error(f'Service not available: {name}')
                    return False
                rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def _wait_for_obs(self, timeout_sec: float = 10.0) -> ObsFrame | None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_obs is not None:
                return deepcopy(self._latest_obs)
        return None

    def _call_trigger(self, client, label: str) -> bool:
        future = client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            self.get_logger().error(f'{label} timed out.')
            return False
        result = future.result()
        if result is None or not result.success:
            msg = result.message if result else 'no response'
            self.get_logger().error(f'{label} failed: {msg}')
            return False
        self.get_logger().info(f'{label}: {result.message}')
        return True

    def _predict_chunk(self, frame: ObsFrame) -> list[list[float]] | None:
        req = PredictActionChunk.Request()
        req.frame = frame
        future = self._predict_client.call_async(req)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            self.get_logger().error('predict_action_chunk timed out.')
            return None

        result = future.result()
        if result is None or not result.success:
            msg = result.message if result else 'no response'
            self.get_logger().error(f'predict_action_chunk failed: {msg}')
            return None

        if result.chunk_size != self._chunk_size:
            self.get_logger().error(
                f'Unexpected chunk_size {result.chunk_size} (expected {self._chunk_size}).'
            )
            return None

        expected_action_dim = 4 if self._dof == 3 else 7
        if result.action_dim != expected_action_dim:
            self.get_logger().error(
                f'Unexpected action_dim {result.action_dim} '
                f'(expected {expected_action_dim} for {self._dof}-DoF).'
            )
            return None

        flat = list(result.actions)
        return [
            flat[i * result.action_dim:(i + 1) * result.action_dim]
            for i in range(result.chunk_size)
        ]

    def _to_robot_action(self, policy_action: list[float]) -> list[float]:
        if self._dof == 3:
            return _pad_rpy(policy_action, self._default_rpy)
        return list(policy_action)

    def _run_episode(self) -> bool:
        """Run one episode. Returns False if user wants to quit."""
        print()
        print(f'=== Episode {self._episode_index} ===')
        _prompt_enter('Place the cube, then press Enter to start this episode.')

        if not self._call_trigger(self._reset_client, 'inference/reset'):
            return True

        cached_chunk: list[list[float]] | None = None
        step_dt = 1.0 / self._fps
        executed_steps = 0

        for step in range(self._max_steps):
            loop_start = time.monotonic()

            if step % self._n_action_steps == 0:
                obs = self._wait_for_obs(timeout_sec=5.0)
                if obs is None:
                    self.get_logger().error('No observation frame available — aborting episode.')
                    break
                cached_chunk = self._predict_chunk(obs)
                if cached_chunk is None:
                    self.get_logger().error('Inference failed — aborting episode.')
                    break

            chunk_idx = step % self._n_action_steps
            policy_action = cached_chunk[chunk_idx]
            robot_action = self._to_robot_action(policy_action)

            msg = Float64MultiArray()
            msg.data = [float(v) for v in robot_action]
            self._pub_action.publish(msg)

            self._append_action_log({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'episode_index': self._episode_index,
                'step': step,
                'policy_action': policy_action,
                'robot_action': robot_action,
            })
            executed_steps += 1

            elapsed = time.monotonic() - loop_start
            sleep_time = step_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            rclpy.spin_once(self, timeout_sec=0.0)

        success = _prompt_yes_no('Episode success?', default=False)

        self._episodes.append({
            'episode_index': self._episode_index,
            'success': success,
            'num_steps': executed_steps,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        self._save_episode_summaries()
        self._episode_index += 1

        print('Moving robot to home...')
        self._call_trigger(self._home_client, 'robot/move_home')

        return _prompt_yes_no('Run another episode?', default=True)

    def run_eval_loop(self) -> None:
        if not self._wait_for_services():
            raise RuntimeError('Required services not available.')

        print()
        print('CloudGripper policy evaluation')
        print(f'Output directory: {self._output_dir}')
        print(f'Control rate: {self._fps} Hz, max steps/episode: {self._max_steps}')
        print('Press Ctrl+C to stop at any time.')
        print()

        _prompt_enter('Waiting for camera/robot. Lock exposure if needed, then press Enter.')

        try:
            while rclpy.ok():
                if not self._run_episode():
                    print('Evaluation finished.')
                    break
        except KeyboardInterrupt:
            print('\nEvaluation interrupted.')


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeManagerNode()
    try:
        node.run_eval_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
