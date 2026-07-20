"""
obs_aggregator_node — assembles LeRobot-compatible observation frames.

Subscribes (relative names, scoped by namespace):
  robot/joint_states      sensor_msgs/JointState       (from xarm_driver)
  robot/ee_pose           geometry_msgs/PoseStamped    (from xarm_driver)
  robot/gripper_pos       std_msgs/Float64             (from xarm_driver)
  camera/color/image      sensor_msgs/Image RGB8 or BGR8 (from realsense2_camera)

Publishes (relative name, scoped by namespace):
  observation/frame       cloudgripper_msgs/ObsFrame

Parameters:
  publish_rate                    (float, default 5.0)   — ObsFrame Hz
  dof_mode                        (int,   default 3)     — 3 or 6
  state_stamp_mismatch_warn_ms    (float, default 5.0)   — warn if |t_ee - t_joint|
  state_buffer_sec                (float, default 0.5)   — robot-state history length
  state_image_match_warn_ms       (float, default 50.0)  — warn if best |t_image-t_state|

Frame assembly (3-DoF):
  state  float32 (11,)  = [joint_0..6 (rad), ee_x, ee_y, ee_z (mm), gripper (0-850)]
  image  sensor_msgs/Image rgb8  (H x W x 3)

Frame assembly (6-DoF):
  state  float32 (14,)  = [joint_0..6 (rad), ee_x, ee_y, ee_z (mm),
                            roll_norm (rad, [0,2π]), pitch (rad), yaw (rad), gripper]
     roll is remapped from [-π, π] → [0, 2π] to avoid ±π discontinuity at the
     gimbal-lock singularity (arm pointing straight down, default RPY [π,0,0]).

Timing:
  Robot states are kept in a short ring buffer. At assemble time the sample whose
  ee_pose stamp is closest to image.header.stamp is selected (stamp-matched).
  timestamp_delta_ms = (t_image - t_ee_pose) * 1000 for that matched sample.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64

from cloudgripper_msgs.msg import ObsFrame


def _stamp_to_sec(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _ensure_rgb8(msg: Image) -> Image:
    """Return an rgb8 Image, converting from bgr8 only when needed."""
    enc = msg.encoding.lower()
    if enc == 'rgb8':
        return msg
    if enc in ('bgr8', 'bgr'):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        rgb = arr[:, :, ::-1].copy()
        out = Image()
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.encoding = 'rgb8'
        out.is_bigendian = False
        out.step = msg.width * 3
        out.data = rgb.tobytes()
        return out
    raise ValueError(f'Unsupported image encoding: {msg.encoding!r} (expected rgb8 or bgr8)')


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


@dataclass(frozen=True)
class _StateSample:
    """One synchronized robot snapshot keyed by ee_pose stamp."""
    t_ee: float
    t_joint: float
    joints: tuple[float, ...]   # (7,)
    ee: tuple[float, ...]       # (6,) x,y,z,roll,pitch,yaw
    gripper: float


class ObsAggregatorNode(Node):
    def __init__(self):
        super().__init__('obs_aggregator_node')

        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('dof_mode', 3)
        self.declare_parameter('state_stamp_mismatch_warn_ms', 5.0)
        self.declare_parameter('state_buffer_sec', 0.5)
        self.declare_parameter('state_image_match_warn_ms', 50.0)

        rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self._dof = self.get_parameter('dof_mode').get_parameter_value().integer_value
        self._state_stamp_warn_ms = (
            self.get_parameter('state_stamp_mismatch_warn_ms')
            .get_parameter_value()
            .double_value
        )
        self._buffer_sec = (
            self.get_parameter('state_buffer_sec').get_parameter_value().double_value
        )
        self._match_warn_ms = (
            self.get_parameter('state_image_match_warn_ms')
            .get_parameter_value()
            .double_value
        )

        if self._dof not in (3, 6):
            raise ValueError(f'dof_mode must be 3 or 6, got {self._dof}')
        if self._buffer_sec <= 0.0:
            raise ValueError(f'state_buffer_sec must be > 0, got {self._buffer_sec}')

        # --- partial caches used to assemble buffer samples ---
        self._joints: list[float] | None = None
        self._joints_t: float | None = None
        self._gripper: float | None = None
        self._image: Image | None = None

        # Ring buffer of complete robot snapshots (ee-stamp keyed)
        self._state_buf: deque[_StateSample] = deque()

        # --- subscribers (relative → namespace-scoped) ---
        self.create_subscription(JointState,   'robot/joint_states',     self._on_joints,  1)
        self.create_subscription(PoseStamped,  'robot/ee_pose',          self._on_ee_pose, 1)
        self.create_subscription(Float64,      'robot/gripper_pos',      self._on_gripper, 1)
        self.create_subscription(Image,        'camera/color/image',     self._on_image,   1)

        # --- publisher ---
        self._pub = self.create_publisher(ObsFrame, 'observation/frame', 1)

        # --- assembly timer ---
        self._timer = self.create_timer(1.0 / rate, self._assemble_and_publish)

        self.get_logger().info(
            f'ObsAggregator ready — {self._dof}-DoF, {rate} Hz, '
            f'state_buffer_sec={self._buffer_sec}, '
            f'state_stamp_mismatch_warn_ms={self._state_stamp_warn_ms}, '
            f'state_image_match_warn_ms={self._match_warn_ms}.'
        )

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        self._joints = list(msg.position[:7])
        self._joints_t = _stamp_to_sec(msg.header.stamp)

    def _on_gripper(self, msg: Float64) -> None:
        self._gripper = msg.data

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        # Push a buffer sample when we have joints + gripper; ee stamp is the key.
        if self._joints is None or self._joints_t is None or self._gripper is None:
            return

        t_ee = _stamp_to_sec(msg.header.stamp)
        mismatch_ms = abs(t_ee - self._joints_t) * 1000.0
        if mismatch_ms > self._state_stamp_warn_ms:
            self.get_logger().warn(
                f'joint/ee stamp mismatch {mismatch_ms:.2f} ms '
                f'(threshold {self._state_stamp_warn_ms} ms) — sample not buffered',
                throttle_duration_sec=2.0,
            )
            return

        q = msg.pose.orientation
        roll, pitch, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
        p = msg.pose.position
        sample = _StateSample(
            t_ee=t_ee,
            t_joint=self._joints_t,
            joints=tuple(self._joints),
            ee=(p.x, p.y, p.z, roll, pitch, yaw),
            gripper=float(self._gripper),
        )
        self._push_sample(sample)

    def _on_image(self, msg: Image) -> None:
        self._image = msg

    def _push_sample(self, sample: _StateSample) -> None:
        self._state_buf.append(sample)
        cutoff = sample.t_ee - self._buffer_sec
        while self._state_buf and self._state_buf[0].t_ee < cutoff:
            self._state_buf.popleft()

    def _pick_closest_state(self, t_image: float) -> _StateSample | None:
        if not self._state_buf:
            return None
        return min(self._state_buf, key=lambda s: abs(s.t_ee - t_image))

    # ------------------------------------------------------------------
    # Frame assembly
    # ------------------------------------------------------------------

    def _assemble_and_publish(self) -> None:
        if self._image is None or not self._state_buf:
            missing = []
            if self._image is None:
                missing.append('image')
            if not self._state_buf:
                missing.append('robot_state_buffer')
            self.get_logger().warn(f'Waiting for: {missing}', throttle_duration_sec=5.0)
            return

        t_image = _stamp_to_sec(self._image.header.stamp)
        sample = self._pick_closest_state(t_image)
        if sample is None:
            return

        delta_ms = (t_image - sample.t_ee) * 1000.0
        if abs(delta_ms) > self._match_warn_ms:
            self.get_logger().warn(
                f'stamp-matched |t_image - t_ee| = {abs(delta_ms):.2f} ms '
                f'(threshold {self._match_warn_ms} ms); '
                f'buffer_len={len(self._state_buf)}',
                throttle_duration_sec=2.0,
            )

        joints = list(sample.joints)
        ee = list(sample.ee)
        gripper = sample.gripper

        if self._dof == 3:
            state = joints + ee[:3] + [gripper]
        else:
            roll_norm = (ee[3] + 2 * math.pi) % (2 * math.pi)
            state = joints + ee[:3] + [roll_norm, ee[4], ee[5], gripper]

        now = self.get_clock().now().to_msg()
        rgb_image = _ensure_rgb8(self._image)

        frame = ObsFrame()
        frame.header.stamp = now
        frame.header.frame_id = ''
        frame.state = [float(v) for v in state]
        frame.image = rgb_image
        frame.dof_mode = f'{self._dof}dof'
        frame.timestamp_delta_ms = float(delta_ms)

        self._pub.publish(frame)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self.get_logger().info('ObsAggregator shutting down.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObsAggregatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
