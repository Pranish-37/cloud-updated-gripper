"""
obs_aggregator_node — assembles LeRobot-compatible observation frames at 5 Hz.

Subscribes (relative names, scoped by namespace):
  robot/joint_states      sensor_msgs/JointState       (from xarm_driver, 15 Hz)
  robot/ee_pose           geometry_msgs/PoseStamped    (from xarm_driver, 15 Hz)
  robot/gripper_pos       std_msgs/Float64             (from xarm_driver, 15 Hz)
  camera/color/image      sensor_msgs/Image RGB8 or BGR8 (from realsense2_camera)

Publishes (relative name, scoped by namespace):
  observation/frame       cloudgripper_msgs/ObsFrame   (5 Hz)

Parameters:
  publish_rate  (float, default 5.0)   — Hz
  dof_mode      (int,   default 3)     — 3 or 6

Frame assembly (3-DoF):
  state  float32 (11,)  = [joint_0..6 (rad), ee_x, ee_y, ee_z (mm), gripper (0-850)]
  image  sensor_msgs/Image rgb8  (H x W x 3)

Frame assembly (6-DoF):
  state  float32 (14,)  = [joint_0..6 (rad), ee_x, ee_y, ee_z (mm),
                            roll_norm (rad, [0,2π]), pitch (rad), yaw (rad), gripper]
  ⚠️  roll is remapped from [-π, π] → [0, 2π] to avoid ±π discontinuity at the
     gimbal-lock singularity (arm pointing straight down, default RPY [π,0,0]).
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64

from cloudgripper_msgs.msg import ObsFrame


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


class ObsAggregatorNode(Node):
    def __init__(self):
        super().__init__('obs_aggregator_node')

        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('dof_mode', 3)

        rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self._dof = self.get_parameter('dof_mode').get_parameter_value().integer_value

        if self._dof not in (3, 6):
            raise ValueError(f'dof_mode must be 3 or 6, got {self._dof}')

        # --- latest-value cache (None until first message received) ---
        self._joints: list[float] | None = None    # (7,) radians
        self._ee_pose: list[float] | None = None   # (6,) x,y,z mm · roll,pitch,yaw rad
        self._gripper: float | None = None         # 0–850 raw
        self._image: Image | None = None           # RGB8 or BGR8 sensor_msgs/Image

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
            f'ObsAggregator ready — {self._dof}-DoF mode, {rate} Hz.'
        )

    # ------------------------------------------------------------------
    # Subscriber callbacks — just cache the latest value
    # ------------------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        self._joints = list(msg.position[:7])

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        # PoseStamped carries position (mm) + orientation (quaternion).
        # We need back the 6-vector [x,y,z,roll,pitch,yaw].
        # The xarm_driver encoded roll,pitch,yaw into the quaternion using ZYX convention;
        # we recover it here via the same convention.
        q = msg.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w

        # ZYX quaternion → euler
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        p = msg.pose.position
        self._ee_pose = [p.x, p.y, p.z, roll, pitch, yaw]

    def _on_gripper(self, msg: Float64) -> None:
        self._gripper = msg.data

    def _on_image(self, msg: Image) -> None:
        self._image = msg

    # ------------------------------------------------------------------
    # Frame assembly
    # ------------------------------------------------------------------

    def _assemble_and_publish(self) -> None:
        if any(v is None for v in (self._joints, self._ee_pose, self._gripper, self._image)):
            missing = [n for n, v in zip(
                ['joints', 'ee_pose', 'gripper', 'image'],
                [self._joints, self._ee_pose, self._gripper, self._image],
            ) if v is None]
            self.get_logger().warn(f'Waiting for: {missing}', throttle_duration_sec=5.0)
            return

        now = self.get_clock().now().to_msg()

        # --- build state vector ---
        joints  = self._joints           # (7,)
        ee      = self._ee_pose          # [x,y,z,roll,pitch,yaw]
        gripper = self._gripper

        if self._dof == 3:
            # clip to [x, y, z] — roll discontinuity not an issue
            state = joints + ee[:3] + [gripper]          # length 11

        else:  # 6-DoF
            # ⚠️  Remap roll from [-π, π] → [0, 2π] to prevent ±π discontinuity
            roll_norm = (ee[3] + 2 * math.pi) % (2 * math.pi)
            state = joints + ee[:3] + [roll_norm, ee[4], ee[5], gripper]  # length 14

        # --- ensure rgb8 (swap channels only if camera publishes bgr8) ---
        rgb_image = _ensure_rgb8(self._image)
        rgb_image.header.stamp = now

        # --- publish ObsFrame ---
        frame = ObsFrame()
        frame.header.stamp = now
        frame.header.frame_id = ''
        frame.state = [float(v) for v in state]
        frame.image = rgb_image
        frame.dof_mode = f'{self._dof}dof'

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
