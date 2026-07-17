"""
episode_manager — real-world policy evaluation control loop.

Subscribes:
  observation/frame    cloudgripper_msgs/ObsFrame

Publishes: (none — actions go through blocking robot/execute_action)

Clients:
  inference/predict_action_chunk   cloudgripper_msgs/PredictActionChunk
  inference/reset                  std_srvs/Trigger
  robot/move_home                  std_srvs/Trigger
  robot/execute_action             cloudgripper_msgs/ExecuteAction
      — blocking EE+gripper move (matches old send_action wait=True)

Interactive control (pygame window — same pattern as teleop data collection):
  Enter / Space  confirm (begin eval / start episode)
  Y               success → home → next episode
  N               failure → home → next episode
  S               abort episode (fail) → home → next episode
  Esc             quit evaluation

Logs (eval_output_dir):
  actions.jsonl          — one action + image path per control step
  episode_dataset.json   — per-episode summary list
  episodes/episode_XXXXXX/frame_YYYYYY.jpg  — observation images
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image
from rclpy.node import Node
from std_srvs.srv import Trigger

from cloudgripper_msgs.msg import ObsFrame
from cloudgripper_msgs.srv import ExecuteAction, PredictActionChunk
from episode_manager.eval_keyboard import EvalAbortEpisode, EvalKeyboard, EvalQuit


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

        self._kb: EvalKeyboard | None = None

        self.create_subscription(ObsFrame, 'observation/frame', self._on_obs_frame, 1)

        self._predict_client = self.create_client(
            PredictActionChunk, 'inference/predict_action_chunk',
        )
        self._reset_client = self.create_client(Trigger, 'inference/reset')
        self._home_client = self.create_client(Trigger, 'robot/move_home')
        self._exec_client = self.create_client(ExecuteAction, 'robot/execute_action')

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

    def _episode_image_dir(self, episode_index: int) -> Path:
        return self._output_dir / 'episodes' / f'episode_{episode_index:06d}'

    def _save_obs_image(self, episode_index: int, step: int, frame: ObsFrame) -> str:
        """Save ObsFrame RGB to JPEG; return path relative to eval_output_dir."""
        img = np.frombuffer(frame.image.data, dtype=np.uint8).reshape(
            frame.image.height, frame.image.width, 3,
        )
        rel = f'episodes/episode_{episode_index:06d}/frame_{step:06d}.jpg'
        path = self._output_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path, format='JPEG', quality=90)
        return rel

    def _on_obs_frame(self, msg: ObsFrame) -> None:
        self._latest_obs = msg

    def _spin_and_poll(self) -> None:
        rclpy.spin_once(self, timeout_sec=0.05)
        if self._kb is None:
            return
        ev = self._kb.poll()
        if ev.quit:
            raise EvalQuit()
        if ev.stop_episode:
            raise EvalAbortEpisode()

    def _wait_confirm(self, status: str) -> None:
        assert self._kb is not None
        self._kb.set_prompt(status, 'Press Enter / Space to confirm')
        self.get_logger().info(f'{status}  [Enter/Space in Eval window]')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            ev = self._kb.poll()
            if ev.quit:
                raise EvalQuit()
            if ev.confirm:
                return

    def _wait_outcome(self, status: str) -> bool:
        """Y → success True, N or S → success False. Esc quits."""
        assert self._kb is not None
        self._kb.set_prompt(status, 'Y success / N failure / S abort')
        self.get_logger().info(f'{status}  [Y/N/S in Eval window]')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            ev = self._kb.poll()
            if ev.quit:
                raise EvalQuit()
            if ev.yes:
                return True
            if ev.no or ev.stop_episode:
                return False
        raise EvalQuit()

    def _wait_for_services(self, timeout_sec: float = 30.0) -> bool:
        deadline = time.time() + timeout_sec
        clients = [
            ('inference/predict_action_chunk', self._predict_client),
            ('inference/reset', self._reset_client),
            ('robot/move_home', self._home_client),
            ('robot/execute_action', self._exec_client),
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
            self._spin_and_poll()
            if self._latest_obs is not None:
                return deepcopy(self._latest_obs)
        return None

    def _call_trigger(self, client, label: str) -> bool:
        future = client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            self._spin_and_poll()
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

    def _execute_robot_action(self, robot_action: list[float]) -> bool:
        """Blocking EE+gripper move — matches old real_e2e send_action(wait=True)."""
        req = ExecuteAction.Request()
        req.command = [float(v) for v in robot_action]
        future = self._exec_client.call_async(req)
        while rclpy.ok() and not future.done():
            self._spin_and_poll()
        if not future.done():
            self.get_logger().error('robot/execute_action timed out.')
            return False
        result = future.result()
        if result is None or not result.success:
            msg = result.message if result else 'no response'
            self.get_logger().error(f'robot/execute_action failed: {msg}')
            return False
        return True

    def _predict_chunk(self, frame: ObsFrame) -> list[list[float]] | None:
        req = PredictActionChunk.Request()
        req.frame = frame
        future = self._predict_client.call_async(req)
        while rclpy.ok() and not future.done():
            self._spin_and_poll()
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

    def _run_episode(self) -> None:
        assert self._kb is not None
        self._kb.set_prompt(f'Episode {self._episode_index}', 'Waiting to start...')
        self.get_logger().info(f'=== Episode {self._episode_index} ===')
        self._wait_confirm(
            f'Episode {self._episode_index}: place the cube, then confirm to start'
        )

        if not self._call_trigger(self._reset_client, 'inference/reset'):
            return

        self._kb.set_prompt(f'Episode {self._episode_index} running...', 'S abort / Esc quit')
        cached_chunk: list[list[float]] | None = None
        step_dt = 1.0 / self._fps
        executed_steps = 0
        episode_idx = self._episode_index
        self._episode_image_dir(episode_idx).mkdir(parents=True, exist_ok=True)

        aborted = False
        try:
            for step in range(self._max_steps):
                self._spin_and_poll()

                if step % self._n_action_steps == 0:
                    obs = self._wait_for_obs(timeout_sec=5.0)
                    if obs is None:
                        self.get_logger().error(
                            'No observation frame available — aborting episode.'
                        )
                        break
                    cached_chunk = self._predict_chunk(obs)
                    if cached_chunk is None:
                        self.get_logger().error('Inference failed — aborting episode.')
                        break

                frame = deepcopy(self._latest_obs) if self._latest_obs is not None else None
                image_rel = ''
                state: list[float] = []
                if frame is not None:
                    image_rel = self._save_obs_image(episode_idx, step, frame)
                    state = [float(v) for v in frame.state]

                chunk_idx = step % self._n_action_steps
                policy_action = cached_chunk[chunk_idx]
                robot_action = self._to_robot_action(policy_action)

                if not self._execute_robot_action(robot_action):
                    self.get_logger().error('Action execution failed — aborting episode.')
                    break

                self._append_action_log({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'episode_index': episode_idx,
                    'step': step,
                    'observation.state': state,
                    'image': image_rel,
                    'policy_action': policy_action,
                    'robot_action': robot_action,
                })
                executed_steps += 1

                if step_dt > 0:
                    # Poll S/Esc during post-move sleep (old precise_sleep(dt))
                    t_end = time.monotonic() + step_dt
                    while time.monotonic() < t_end:
                        self._spin_and_poll()
                        time.sleep(min(0.05, max(0.0, t_end - time.monotonic())))

            success = self._wait_outcome(
                f'Episode {episode_idx} done — mark success (Y) or failure (N)'
            )
        except EvalAbortEpisode:
            aborted = True
            success = False
            self.get_logger().info(
                f'Episode {episode_idx} stopped by user (S) — marking failure.'
            )

        self._episodes.append({
            'episode_index': episode_idx,
            'success': success,
            'aborted': aborted,
            'num_steps': executed_steps,
            'images_dir': f'episodes/episode_{episode_idx:06d}',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        self._save_episode_summaries()
        self._episode_index += 1

        self.get_logger().info(
            f'Episode labeled success={success}. Moving robot to home...'
        )
        self._kb.set_prompt('Moving to home...', 'Please wait')
        self._call_trigger(self._home_client, 'robot/move_home')

    def run_eval_loop(self) -> None:
        if not self._wait_for_services():
            raise RuntimeError('Required services not available.')

        self._kb = EvalKeyboard()
        self.get_logger().info('CloudGripper policy evaluation')
        self.get_logger().info(f'Output directory: {self._output_dir}')
        self.get_logger().info(
            f'Control rate: {self._fps} Hz, max steps/episode: {self._max_steps}'
        )
        self.get_logger().info('Focus the Eval pygame window for keyboard control.')

        try:
            self._wait_confirm('Press Enter / Space to begin evaluation')
            while rclpy.ok():
                self._run_episode()
        except EvalQuit:
            self.get_logger().info('Evaluation quit by user.')
        finally:
            if self._kb is not None:
                self._kb.close()
                self._kb = None


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
