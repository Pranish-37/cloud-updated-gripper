"""
teleop_node — ROS 2 keyboard + SpaceMouse teleoperation node.

Control mapping:
  W/S           → EE +x / -x  (translation)
  A/D           → EE +y / -y
  Space/Ctrl    → EE +z / -z
  J/L I/K U/O   → roll/pitch/yaw (keyboard-only mode)
  SpaceMouse    → roll/pitch/yaw (when use_spacemouse:=true)
  Q             → gripper open
  E             → gripper close
  Esc           → stop  [+ finalize dataset when enable_recording:=true]

Recording control (only active when enable_recording:=true):
  R             → toggle recording on/off within current episode
  N             → save episode, start next
  D             → discard episode, restart same index

Subscribes (relative, namespace-scoped):
  robot/ee_pose       geometry_msgs/PoseStamped   — to seed initial target pose
  robot/gripper_pos   std_msgs/Float64            — to seed initial gripper value

Publishes (relative, namespace-scoped):
  policy/action_cmd   std_msgs/Float64MultiArray  (7-vec: x,y,z mm · roll,pitch,yaw rad · gripper)

Calls (when enable_recording:=true):
  recorder/toggle_recording   std_srvs/Trigger
  recorder/save_episode       std_srvs/Trigger
  recorder/discard_episode    std_srvs/Trigger
  recorder/finalize_dataset   std_srvs/Trigger

Parameters:
  control_hz          (float, default 50.0)
  translation_speed   (float, default 50.0   mm/s)
  rotation_speed      (float, default 10.0   deg/s, keyboard only)
  use_spacemouse      (bool,  default True)
  spacemouse_rotation_speed (float, default 30.0)
  spacemouse_deadzone       (float, default 0.05)
  invert_x / invert_y       (bool,  default True)
  invert_roll/pitch/yaw     (bool,  default False)
  roll/pitch/yaw_scale      (float, default 1.0)
  enable_recording    (bool,  default False) — enable R/N/D/ESC recorder key bindings

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
    return max(lo, min(hi, v)) # clamp the value between the lower and upper limits


def _wrap_deg(v: float) -> float:
    """Wrap angle to [-180, 180] degrees.""" # round the angle to the range [-180, 180] degrees
    return ((v + 180.0) % 360.0) - 180.0


class TeleopNode(Node):
    def __init__(self): # TeleopNode 
        super().__init__('teleop_node')

        # --- declare parameters ---
        # declare the parameters for the TeleopNode
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('translation_speed', 50.0)
        self.declare_parameter('rotation_speed', 10.0)
        self.declare_parameter('use_spacemouse', True)
        self.declare_parameter('spacemouse_rotation_speed', 30.0)
        self.declare_parameter('spacemouse_deadzone', 0.05)
        self.declare_parameter('invert_x', True)
        self.declare_parameter('invert_y', True)
        self.declare_parameter('invert_roll', False)
        self.declare_parameter('invert_pitch', False)
        self.declare_parameter('invert_yaw', False)
        self.declare_parameter('roll_scale', 1.0)
        self.declare_parameter('pitch_scale', 1.0)
        self.declare_parameter('yaw_scale', 1.0)

        # get the parameter values from the parameter server
        def _f(name): return self.get_parameter(name).get_parameter_value().double_value # float
        def _b(name): return self.get_parameter(name).get_parameter_value().bool_value # boolean

        self._hz = _f('control_hz') # control frequency
        use_sm   = _b('use_spacemouse') # use SpaceMouse

        cfg = TeleopConfig( # create the TeleopConfig object
            translation_speed         = _f('translation_speed'),
            rotation_speed            = _f('rotation_speed'),
            control_hz                = self._hz,
            invert_x                  = _b('invert_x'),
            invert_y                  = _b('invert_y'),
            spacemouse_rotation_speed = _f('spacemouse_rotation_speed'),
            spacemouse_deadzone       = _f('spacemouse_deadzone'),
            invert_roll               = _b('invert_roll'),
            invert_pitch              = _b('invert_pitch'),
            invert_yaw                = _b('invert_yaw'),
            roll_scale                = _f('roll_scale'),
            pitch_scale               = _f('pitch_scale'),
            yaw_scale                 = _f('yaw_scale'),
        )

        self.declare_parameter('enable_recording', False)
        self._enable_recording = _b('enable_recording')

        self._cfg = cfg
        self._keyboard = KeyboardInput(cfg)
        # create the SpaceMouseInput object if use_spacemouse is True, otherwise None
        self._spacemouse: SpaceMouseInput | None = SpaceMouseInput(cfg) if use_sm else None

        # target pose [x,y,z mm, roll,pitch,yaw deg] — seeded independently from robot topics
        self._target = [300.0, 0.0, 300.0, 180.0, 0.0, 0.0]
        self._gripper = cfg.gripper_close_val
        self._target_seeded = False   # True once ee_pose received
        self._gripper_seeded = False  # True once gripper_pos received

        # recording state (mirrors recorder_node state for HUD display)
        self._is_recording = False

        # --- subscribers to seed initial pose ---
        self.create_subscription(PoseStamped, 'robot/ee_pose', self._on_ee_pose, 1)
        self.create_subscription(Float64, 'robot/gripper_pos', self._on_gripper, 1)

        # --- publisher ---
        self._pub = self.create_publisher(Float64MultiArray, 'policy/action_cmd', 1)

        # --- recorder service clients (only created when enable_recording is True) ---
        self._rec_toggle_client   = None
        self._rec_save_client     = None
        self._rec_discard_client  = None
        self._rec_finalize_client = None

        if self._enable_recording:
            self._rec_toggle_client   = self.create_client(Trigger, 'recorder/toggle_recording')
            self._rec_save_client     = self.create_client(Trigger, 'recorder/save_episode')
            self._rec_discard_client  = self.create_client(Trigger, 'recorder/discard_episode')
            self._rec_finalize_client = self.create_client(Trigger, 'recorder/finalize_dataset')
            self.get_logger().info(
                'Recording mode enabled — R=toggle  N=save  D=discard  ESC=finalize+quit'
            )

        self.get_logger().info(
            f'TeleopNode ready — {self._hz} Hz, '
            f'SpaceMouse={"on" if use_sm else "off"}, '
            f'recording={"on" if self._enable_recording else "off"}. '
            f'Open pygame window and start controlling.'
        )

    # ------------------------------------------------------------------
    # Pose seeding from robot state
    # ------------------------------------------------------------------

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        if self._target_seeded:
            return
        # Recover euler angles from quaternion (ZYX) — inverse of xarm_driver's encoding
        q = msg.pose.orientation
        roll  = math.atan2(2*(q.w*q.x + q.y*q.z), 1 - 2*(q.x*q.x + q.y*q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2*(q.w*q.y - q.z*q.x))))
        yaw   = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        p = msg.pose.position
        self._target = [
            p.x, p.y, p.z,
            math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
        ]
        self._target_seeded = True
        self.get_logger().info(
            f'Target seeded: '
            f'xyz=({p.x:.1f},{p.y:.1f},{p.z:.1f}) mm  '
            f'rpy=({math.degrees(roll):.1f},{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}) deg'
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
        """Fire-and-forget async service call with a log callback."""
        if client is None or not client.service_is_ready():
            self.get_logger().warn(f'Recorder service not ready: {name}')
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self._on_recorder_response(f, name)
        )

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
        self._call_recorder(self._rec_save_client, 'save_episode')

    def _discard_episode(self) -> None:
        self._is_recording = False
        self._call_recorder(self._rec_discard_client, 'discard_episode')

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
                self.get_logger().info('SpaceMouse opened.')
            except Exception as e:
                self.get_logger().warn(f'SpaceMouse unavailable: {e} — falling back to keyboard rotation.')
                self._spacemouse = None

        dt = 1.0 / self._hz
        cfg = self._cfg

        # Build recorder keydown callbacks (only populated when recording is enabled)
        recorder_callbacks: dict = {}
        if self._enable_recording:
            recorder_callbacks = {
                pygame.K_r:      self._toggle_recording,
                pygame.K_n:      self._save_episode,
                pygame.K_d:      self._discard_episode,
                pygame.K_ESCAPE: self._finalize_dataset,
            }

        # Wait until both robot topics have been received before starting control.
        # This guarantees _target reflects the real EE position, not the default value.
        self.get_logger().info('Waiting for robot state seed (ee_pose + gripper_pos)...')
        while rclpy.ok() and not (self._target_seeded and self._gripper_seeded):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Robot state seeded — starting control loop.')

        try:
            while rclpy.ok() and not self._keyboard.stop_requested:
                t0 = time.perf_counter()

                # spin once to keep ROS callbacks alive
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

                # accumulate target pose
                x, y, z, roll, pitch, yaw = self._target
                x     = _clamp(x     + vx      * dt, *cfg.x_limits)
                y     = _clamp(y     + vy      * dt, *cfg.y_limits)
                z     = _clamp(z     + vz      * dt, *cfg.z_limits)
                roll  = _wrap_deg(roll  + v_roll  * dt)
                pitch = _wrap_deg(pitch + v_pitch * dt)
                yaw   = _wrap_deg(yaw   + v_yaw   * dt)
                self._target = [x, y, z, roll, pitch, yaw]

                if gripper_cmd is not None:
                    self._gripper = gripper_cmd

                # publish 7-vector: x,y,z mm · roll,pitch,yaw rad · gripper
                msg = Float64MultiArray()
                msg.data = [
                    x, y, z,
                    math.radians(roll),
                    math.radians(pitch),
                    math.radians(yaw),
                    self._gripper,
                ]
                self._pub.publish(msg)

                # rate limiting
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
