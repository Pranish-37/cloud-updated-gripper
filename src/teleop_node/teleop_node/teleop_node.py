"""
teleop_node — ROS 2 keyboard + SpaceMouse teleoperation node.

Control mapping (SpaceMouse available):
  SpaceMouse x/y/z         → EE translation
  SpaceMouse roll/pitch/yaw → EE rotation (6-DoF only; ignored in 3-DoF)
  SpaceMouse left / right  → gripper open / close
  Q / E                    → gripper open / close (backup)
  Esc                      → stop  [+ finalize dataset when enable_recording:=true]

Control mapping (SpaceMouse unavailable / use_spacemouse:=false):
  W/S, A/D, Space/Ctrl     → EE translation
  J/L I/K U/O              → roll/pitch/yaw (6-DoF only; disabled in 3-DoF)
  Q / E                    → gripper open / close

3-DoF: RPY is locked to robot_home_position for the whole session.

Recording control (only active when enable_recording:=true):
  R             → toggle recording on/off within current episode
  N             → save episode, start next
  Backspace     → discard episode, restart same index

Subscribes (relative, namespace-scoped):
  robot/ee_pose       geometry_msgs/PoseStamped   — to seed initial target pose
  robot/gripper_pos   std_msgs/Float64            — to seed initial gripper value

Publishes (relative, namespace-scoped):
  policy/action_cmd   std_msgs/Float64MultiArray  (7-vec: x,y,z mm · roll,pitch,yaw rad · gripper)

Parameters:
  dof_mode                    (int,   default 3) — 3 or 6
  control_hz                  (float, default 50.0)
  translation_speed           (float, default 50.0 mm/s, keyboard)
  rotation_speed              (float, default 10.0 deg/s, keyboard)
  use_spacemouse              (bool,  default True)
  spacemouse_translation_speed (float, default -1 → use translation_speed)
  spacemouse_rotation_speed   (float, default 30.0)
  spacemouse_translation_deadzone (float, default 0.05)
  spacemouse_rotation_deadzone    (float, default 0.05)
  invert_x / invert_y         (bool,  default True)  — keyboard translation
  spacemouse_invert_x/y       (bool,  default False) — SpaceMouse translation
  invert_roll/pitch/yaw       (bool,  default False)
  roll/pitch/yaw_scale        (float, default 1.0)
  x/y/z_limits                ([min,max] mm)
  roll/pitch/yaw_limits       ([min,max] deg) — 6-DoF only; roll in [0,360)
  max_step_x/y/z_mm           (float, per-tick absolute mm cap)
  max_step_roll/pitch/yaw_deg (float, per-tick absolute deg cap)

Note: xarm_driver_node must be started with servo_mode:=true for real-time response.
"""

import math
import time

import pygame
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger

from teleop_node.input_handler import TeleopConfig, KeyboardInput, SpaceMouseInput


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_deg(v: float) -> float:
    """Wrap angle to [-180, 180] degrees (pitch / yaw)."""
    return ((v + 180.0) % 360.0) - 180.0


def _wrap_roll_deg(v: float) -> float:
    """Wrap roll to [0, 360) degrees."""
    return v % 360.0


def _nearest_deg(value: float, reference: float) -> float:
    """Return value + 360*k closest to reference (continuous branch pick)."""
    delta = (value - reference + 180.0) % 360.0 - 180.0
    return reference + delta


def _clamp_delta(prev: float, new: float, max_delta: float) -> float:
    """Limit |new - prev| to max_delta (startup / glitch safety)."""
    d = new - prev
    if abs(d) <= max_delta:
        return new
    return prev + math.copysign(max_delta, d)


def _clamp_delta_angle_deg(prev: float, new: float, max_delta: float) -> float:
    """Limit angular step using shortest-path delta (degrees)."""
    d = (new - prev + 180.0) % 360.0 - 180.0
    if abs(d) <= max_delta:
        return prev + d
    return prev + math.copysign(max_delta, d)


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('translation_speed', 50.0)
        self.declare_parameter('rotation_speed', 10.0)
        self.declare_parameter('dof_mode', 3)
        self.declare_parameter('use_spacemouse', True)
        # -1.0 means "mirror translation_speed"
        self.declare_parameter('spacemouse_translation_speed', -1.0)
        self.declare_parameter('spacemouse_rotation_speed', 30.0)
        self.declare_parameter('spacemouse_translation_deadzone', 0.05)
        self.declare_parameter('spacemouse_rotation_deadzone', 0.05)
        self.declare_parameter('invert_x', True)
        self.declare_parameter('invert_y', True)
        self.declare_parameter('spacemouse_invert_x', False)
        self.declare_parameter('spacemouse_invert_y', False)
        self.declare_parameter('invert_roll', False)
        self.declare_parameter('invert_pitch', False)
        self.declare_parameter('invert_yaw', False)
        self.declare_parameter('roll_scale', 1.0)
        self.declare_parameter('pitch_scale', 1.0)
        self.declare_parameter('yaw_scale', 1.0)
        # Safety envelopes — [min, max]; XYZ mm, RPY degrees
        self.declare_parameter('x_limits', [200.0, 700.0])
        self.declare_parameter('y_limits', [-400.0, 400.0])
        self.declare_parameter('z_limits', [6.0, 600.0])
        self.declare_parameter('roll_limits', [90.0, 270.0])
        self.declare_parameter('pitch_limits', [-45.0, 45.0])
        self.declare_parameter('yaw_limits', [-90.0, 90.0])
        self.declare_parameter('max_step_x_mm', 5.0)
        self.declare_parameter('max_step_y_mm', 5.0)
        self.declare_parameter('max_step_z_mm', 5.0)
        self.declare_parameter('max_step_roll_deg', 2.0)
        self.declare_parameter('max_step_pitch_deg', 2.0)
        self.declare_parameter('max_step_yaw_deg', 2.0)

        def _f(name): return self.get_parameter(name).get_parameter_value().double_value
        def _b(name): return self.get_parameter(name).get_parameter_value().bool_value
        def _i(name): return self.get_parameter(name).get_parameter_value().integer_value

        def _lim(name: str) -> tuple[float, float]:
            v = list(self.get_parameter(name).get_parameter_value().double_array_value)
            if len(v) != 2:
                raise ValueError(f'{name} must be [min, max], got {v}')
            lo, hi = float(v[0]), float(v[1])
            if lo > hi:
                raise ValueError(f'{name}: min ({lo}) > max ({hi})')
            return lo, hi

        self._hz = _f('control_hz')
        use_sm = _b('use_spacemouse')
        self._dof = _i('dof_mode')
        if self._dof not in (3, 6):
            raise ValueError(f'dof_mode must be 3 or 6, got {self._dof}')

        translation_speed = _f('translation_speed')
        sm_ts = _f('spacemouse_translation_speed')
        if sm_ts < 0.0:
            sm_ts = translation_speed

        x_lim, y_lim, z_lim = _lim('x_limits'), _lim('y_limits'), _lim('z_limits')
        roll_lim, pitch_lim, yaw_lim = (
            _lim('roll_limits'), _lim('pitch_limits'), _lim('yaw_limits'),
        )

        cfg = TeleopConfig(
            translation_speed=translation_speed,
            rotation_speed=_f('rotation_speed'),
            control_hz=self._hz,
            dof_mode=self._dof,
            invert_x=_b('invert_x'),
            invert_y=_b('invert_y'),
            x_limits=x_lim,
            y_limits=y_lim,
            z_limits=z_lim,
            roll_limits=roll_lim,
            pitch_limits=pitch_lim,
            yaw_limits=yaw_lim,
            max_step_x_mm=_f('max_step_x_mm'),
            max_step_y_mm=_f('max_step_y_mm'),
            max_step_z_mm=_f('max_step_z_mm'),
            max_step_roll_deg=_f('max_step_roll_deg'),
            max_step_pitch_deg=_f('max_step_pitch_deg'),
            max_step_yaw_deg=_f('max_step_yaw_deg'),
            spacemouse_translation_speed=sm_ts,
            spacemouse_rotation_speed=_f('spacemouse_rotation_speed'),
            spacemouse_translation_deadzone=_f('spacemouse_translation_deadzone'),
            spacemouse_rotation_deadzone=_f('spacemouse_rotation_deadzone'),
            spacemouse_invert_x=_b('spacemouse_invert_x'),
            spacemouse_invert_y=_b('spacemouse_invert_y'),
            invert_roll=_b('invert_roll'),
            invert_pitch=_b('invert_pitch'),
            invert_yaw=_b('invert_yaw'),
            roll_scale=_f('roll_scale'),
            pitch_scale=_f('pitch_scale'),
            yaw_scale=_f('yaw_scale'),
        )

        self.declare_parameter('enable_recording', False)
        self._enable_recording = _b('enable_recording')

        self.declare_parameter('robot_home_position', [330.0, 0.0, 160.0, 3.14159, 0.0, 0.0, 850.0])
        _home_raw = list(
            self.get_parameter('robot_home_position').get_parameter_value().double_array_value
        )  # [x, y, z, roll, pitch, yaw (rad), gripper]
        self._home_target = [
            _home_raw[0], _home_raw[1], _home_raw[2],
            math.degrees(_home_raw[3]),
            math.degrees(_home_raw[4]),
            math.degrees(_home_raw[5]),
        ]
        self._home_gripper = _home_raw[6]
        self._homing = False

        self._cfg = cfg
        self._keyboard = KeyboardInput(cfg)
        self._spacemouse: SpaceMouseInput | None = SpaceMouseInput(cfg) if use_sm else None

        # target pose [x,y,z mm, roll,pitch,yaw deg]
        self._target = [
            300.0, 0.0, 300.0,
            self._home_target[3], self._home_target[4], self._home_target[5],
        ]
        self._locked_rpy = list(self._home_target[3:6])  # frozen in 3-DoF after seed
        self._gripper = cfg.gripper_close_val
        self._target_seeded = False
        self._gripper_seeded = False
        self._is_recording = False

        self.create_subscription(PoseStamped, 'robot/ee_pose', self._on_ee_pose, 1)
        self.create_subscription(Float64, 'robot/gripper_pos', self._on_gripper, 1)
        self._pub = self.create_publisher(Float64MultiArray, 'policy/action_cmd', 1)

        self._rec_toggle_client = None
        self._rec_save_client = None
        self._rec_discard_client = None
        self._rec_finalize_client = None
        self._move_home_client = None

        if self._enable_recording:
            self._rec_toggle_client = self.create_client(Trigger, 'recorder/toggle_recording')
            self._rec_save_client = self.create_client(Trigger, 'recorder/save_episode')
            self._rec_discard_client = self.create_client(Trigger, 'recorder/discard_episode')
            self._rec_finalize_client = self.create_client(Trigger, 'recorder/finalize_dataset')
            self._move_home_client = self.create_client(Trigger, 'robot/move_home')
            self.get_logger().info(
                'Recording mode enabled — R=toggle  N=save  Bksp=discard  ESC=finalize+quit'
            )

        self.get_logger().info(
            f'TeleopNode ready — {self._dof}-DoF, {self._hz} Hz, '
            f'SpaceMouse={"on" if use_sm else "off"}, '
            f'sm_trans={sm_ts:.1f} mm/s, '
            f'recording={"on" if self._enable_recording else "off"}.'
        )
        self.get_logger().info(
            f'Safety limits — xyz mm: x={x_lim}, y={y_lim}, z={z_lim}; '
            f'rpy deg: roll={roll_lim}, pitch={pitch_lim}, yaw={yaw_lim}'
        )
        self.get_logger().info(
            f'Max step/tick — xyz mm: '
            f'({cfg.max_step_x_mm}, {cfg.max_step_y_mm}, {cfg.max_step_z_mm}); '
            f'rpy deg: ({cfg.max_step_roll_deg}, {cfg.max_step_pitch_deg}, '
            f'{cfg.max_step_yaw_deg})'
        )

    # ------------------------------------------------------------------
    # Pose seeding from robot state
    # ------------------------------------------------------------------

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        if self._target_seeded:
            return
        q = msg.pose.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        p = msg.pose.position

        # Keep roll on the branch nearest home (≈180°) so we never command a
        # ±π discontinuity into servo mode on the first tick.
        r_deg = _nearest_deg(_wrap_roll_deg(math.degrees(roll)), self._home_target[3])
        p_deg = math.degrees(pitch)
        y_deg = math.degrees(yaw)

        self._target = [p.x, p.y, p.z, r_deg, p_deg, y_deg]
        self._locked_rpy = [r_deg, p_deg, y_deg]
        self._target_seeded = True
        self.get_logger().info(
            f'Target seeded: xyz=({p.x:.1f},{p.y:.1f},{p.z:.1f}) mm  '
            f'rpy=({r_deg:.1f},{p_deg:.1f},{y_deg:.1f}) deg'
            + (' [3-DoF RPY locked]' if self._dof == 3 else '')
        )

    def _on_gripper(self, msg: Float64) -> None:
        if self._gripper_seeded:
            return
        self._gripper = msg.data
        self._gripper_seeded = True
        self.get_logger().info(f'Gripper seeded: {msg.data:.0f}')

    # ------------------------------------------------------------------
    # Recorder service helpers
    # ------------------------------------------------------------------

    def _call_recorder(self, client, name: str) -> None:
        if client is None or not client.service_is_ready():
            self.get_logger().warn(f'Recorder service not ready: {name}')
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda f: self._on_recorder_response(f, name))

    def _on_recorder_response(self, future, name: str) -> None:
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f'[recorder/{name}] {res.message}')
            else:
                self.get_logger().warn(f'[recorder/{name}] {res.message}')
        except Exception as e:
            self.get_logger().error(f'[recorder/{name}] call failed: {e}')

    def _toggle_recording(self) -> None:
        self._is_recording = not self._is_recording
        self._call_recorder(self._rec_toggle_client, 'toggle_recording')

    def _save_episode(self) -> None:
        self._is_recording = False
        if self._rec_save_client and self._rec_save_client.service_is_ready():
            self._call_recorder(self._rec_save_client, 'save_episode')
            self._reset_to_home()
        else:
            self.get_logger().warn('Recorder not ready — save skipped.')

    def _discard_episode(self) -> None:
        self._is_recording = False
        if self._rec_discard_client and self._rec_discard_client.service_is_ready():
            self._call_recorder(self._rec_discard_client, 'discard_episode')
            self._reset_to_home()
        else:
            self.get_logger().warn('Recorder not ready — discard skipped.')

    def _reset_to_home(self) -> None:
        if self._homing:
            self.get_logger().warn('move_home already in flight — ignoring duplicate request.')
            return
        if not (self._move_home_client and self._move_home_client.service_is_ready()):
            self.get_logger().warn('robot/move_home service not ready — skipping home reset.')
            return

        self._homing = True
        self._target = list(self._home_target)
        self._locked_rpy = list(self._home_target[3:6])
        self._gripper = self._home_gripper
        self.get_logger().info(
            f'Resetting robot to home: xyz=({self._home_target[0]},{self._home_target[1]},'
            f'{self._home_target[2]}) mm  '
            f'rpy=({self._home_target[3]:.1f},{self._home_target[4]:.1f},'
            f'{self._home_target[5]:.1f}) deg  gripper={self._home_gripper}'
        )
        future = self._move_home_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_move_home_done)

    def _on_move_home_done(self, future) -> None:
        self._homing = False
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f'[move_home] {res.message}')
            else:
                self.get_logger().warn(f'[move_home] failed: {res.message}')
        except Exception as exc:
            self.get_logger().error(f'[move_home] call failed: {exc}')

    def _finalize_dataset(self) -> None:
        self._is_recording = False
        self._call_recorder(self._rec_finalize_client, 'finalize_dataset')

    # ------------------------------------------------------------------
    # Main blocking run — called from main()
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking control loop. Returns when Esc is pressed."""
        self._keyboard.start()
        if self._spacemouse:
            try:
                self._spacemouse.start()
                self.get_logger().info('SpaceMouse opened — waiting for center (release cap)...')
            except Exception as e:
                self.get_logger().warn(
                    f'SpaceMouse unavailable: {e} — falling back to keyboard control.'
                )
                self._spacemouse = None

        dt = 1.0 / self._hz
        cfg = self._cfg

        recorder_callbacks: dict = {}
        if self._enable_recording:
            recorder_callbacks = {
                pygame.K_r: self._toggle_recording,
                pygame.K_n: self._save_episode,
                pygame.K_BACKSPACE: self._discard_episode,
                pygame.K_ESCAPE: self._finalize_dataset,
            }

        self.get_logger().info('Waiting for robot state seed (ee_pose + gripper_pos)...')
        while rclpy.ok() and not (self._target_seeded and self._gripper_seeded):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Robot state seeded.')

        # Arm SpaceMouse only after it reports centered — avoids startup 爆冲 from
        # residual HID deflection before the operator touches anything.
        if self._spacemouse is not None:
            t_arm0 = time.perf_counter()
            while rclpy.ok() and not self._spacemouse.is_centered():
                rclpy.spin_once(self, timeout_sec=0.0)
                if time.perf_counter() - t_arm0 > 10.0:
                    self.get_logger().warn(
                        'SpaceMouse not centered after 10 s — arming anyway; '
                        'release the cap if motion feels wrong.'
                    )
                    break
                time.sleep(0.02)
            self._spacemouse.arm()
            self.get_logger().info('SpaceMouse armed — control active.')

        self.get_logger().info('Starting control loop.')

        try:
            while rclpy.ok() and not self._keyboard.stop_requested:
                t0 = time.perf_counter()
                rclpy.spin_once(self, timeout_sec=0.0)

                rec_status = 'REC' if self._is_recording else ('PAUSED' if self._enable_recording else '')

                stop, vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd = \
                    self._keyboard.step(
                        self._spacemouse, self._target, self._gripper,
                        extra_keydown_callbacks=recorder_callbacks,
                        recording_status=rec_status,
                    )

                if stop:
                    break

                prev_x, prev_y, prev_z, prev_roll, prev_pitch, prev_yaw = self._target
                x = _clamp(prev_x + vx * dt, *cfg.x_limits)
                y = _clamp(prev_y + vy * dt, *cfg.y_limits)
                z = _clamp(prev_z + vz * dt, *cfg.z_limits)

                if self._dof == 3:
                    roll, pitch, yaw = self._locked_rpy
                else:
                    roll = _clamp(_wrap_roll_deg(prev_roll + v_roll * dt), *cfg.roll_limits)
                    pitch = _clamp(_wrap_deg(prev_pitch + v_pitch * dt), *cfg.pitch_limits)
                    yaw = _clamp(_wrap_deg(prev_yaw + v_yaw * dt), *cfg.yaw_limits)

                # Anti-爆冲: absolute per-tick step caps from yaml.
                x = _clamp_delta(prev_x, x, cfg.max_step_x_mm)
                y = _clamp_delta(prev_y, y, cfg.max_step_y_mm)
                z = _clamp_delta(prev_z, z, cfg.max_step_z_mm)
                if self._dof == 6:
                    roll = _clamp_delta_angle_deg(prev_roll, roll, cfg.max_step_roll_deg)
                    pitch = _clamp_delta_angle_deg(prev_pitch, pitch, cfg.max_step_pitch_deg)
                    yaw = _clamp_delta_angle_deg(prev_yaw, yaw, cfg.max_step_yaw_deg)
                    roll = _clamp(_wrap_roll_deg(roll), *cfg.roll_limits)
                    pitch = _clamp(_wrap_deg(pitch), *cfg.pitch_limits)
                    yaw = _clamp(_wrap_deg(yaw), *cfg.yaw_limits)
                # Re-apply workspace envelope after delta clamp.
                x = _clamp(x, *cfg.x_limits)
                y = _clamp(y, *cfg.y_limits)
                z = _clamp(z, *cfg.z_limits)

                self._target = [x, y, z, roll, pitch, yaw]

                if gripper_cmd is not None:
                    self._gripper = gripper_cmd

                msg = Float64MultiArray()
                msg.data = [
                    x, y, z,
                    math.radians(roll),
                    math.radians(pitch),
                    math.radians(yaw),
                    self._gripper,
                ]
                self._pub.publish(msg)

                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        finally:
            if self._spacemouse:
                self._spacemouse.stop()
            self._keyboard.stop()
            self.get_logger().info('Teleop stopped.')


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
