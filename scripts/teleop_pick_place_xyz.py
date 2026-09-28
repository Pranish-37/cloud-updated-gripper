#!/usr/bin/env python3
"""
Pick-and-place sequence using the exact pose values taken from the teleop HUD.

Sequence:
  1) read current start pose from robot/ee_pose
  2) move to pick pose (second teleop position) and close gripper to 0
  3) move to top-in-air pose and keep gripper closed
  4) move to drop pose and open gripper to 850
  5) return to the original start pose

This matches the values reported by the teleop window:
  start:  (x=300.0, y=0.0, z=100.0, roll=180°, pitch=0°, yaw=0°, gripper=850)
  pick:   (x=448.6, y=152.5, z=15.0, roll=180°, pitch=0°, yaw=0°, gripper=0)
  air:    (x=487.6, y=45.5, z=179.6, roll=180°, pitch=0°, yaw=0°, gripper=0)
  drop:   (x=487.8, y=-134.5, z=12.9, roll=180°, pitch=0°, yaw=0°, gripper=850)

This script is intentionally simple and uses the robot's Cartesian pose interface,
which is what the teleop HUD is reporting.
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from cloudgripper_msgs.srv import ExecuteAction

# Teleop values in mm and radians.
START = [300.0, 0.0, 100.0, math.pi, 0.0, 0.0]
PICK = [448.6, 152.5, 15.0, math.pi, 0.0, 0.0]
AIR_TOP = [487.6, 45.5, 179.6, math.pi, 0.0, 0.0]
DROP = [487.8, -134.5, 12.9, math.pi, 0.0, 0.0]

GRIPPER_OPEN = 850.0
GRIPPER_CLOSE = 0.0


class TeleopPoseSequence(Node):
    def __init__(self):
        super().__init__('teleop_pose_sequence')
        self._current_pose = None
        self._ee_sub = self.create_subscription(PoseStamped, 'robot/ee_pose', self._on_ee_pose, 10)
        self._exec_client = self.create_client(ExecuteAction, 'robot/execute_action')

    def _on_ee_pose(self, msg: PoseStamped):
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        roll, pitch, yaw = self._quaternion_to_rpy(qx, qy, qz, qw)
        self._current_pose = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(roll),
            float(pitch),
            float(yaw),
        ]

    @staticmethod
    def _quaternion_to_rpy(x: float, y: float, z: float, w: float):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def wait_for_pose(self, timeout=10.0):
        deadline = time.time() + timeout
        while rclpy.ok() and self._current_pose is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() > deadline:
                raise RuntimeError('Timed out waiting for robot/ee_pose.')
        if self._current_pose is None:
            raise RuntimeError('Current pose unavailable.')
        return self._current_pose

    def wait_for_service(self, timeout=10.0):
        deadline = time.time() + timeout
        while rclpy.ok() and not self._exec_client.service_is_ready():
            if time.time() > deadline:
                raise RuntimeError(
                    'robot/execute_action service not available. Start xarm_driver_node first.'
                )
            time.sleep(0.05)

    def move_to(self, pose, gripper):
        req = ExecuteAction.Request()
        req.command = [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
            float(gripper),
        ]
        self.get_logger().info(
            f"Move -> x={req.command[0]:.2f} y={req.command[1]:.2f} z={req.command[2]:.2f} "
            f"roll={req.command[3]:.4f} pitch={req.command[4]:.4f} yaw={req.command[5]:.4f} "
            f"gripper={req.command[6]:.1f}"
        )
        future = self._exec_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        result = future.result()
        if result is None:
            raise RuntimeError('No result from robot/execute_action.')
        if not result.success:
            raise RuntimeError(f'Execution failed: {result.message}')

    @staticmethod
    def _poses_match(a, b, tol=1e-3):
        return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(6))

    def move_slowly(self, target_pose, target_gripper, steps=60, pause=0.45):
        """Move smoothly while keeping the target orientation fixed.

        The xArm motion is sensitive to orientation changes during a translational
        move. We therefore interpolate only x/y/z and keep roll/pitch/yaw fixed at
        the target pose for the entire segment.
        """
        if self._current_pose is None:
            return self.move_to(target_pose, target_gripper)

        start_pose = list(self._current_pose)
        if self._poses_match(start_pose, target_pose):
            self.get_logger().info('Skipping no-op move to current pose.')
            return

        current_pose = list(start_pose)
        for i in range(1, steps + 1):
            alpha = i / steps
            pose = [
                start_pose[0] + (target_pose[0] - start_pose[0]) * alpha,
                start_pose[1] + (target_pose[1] - start_pose[1]) * alpha,
                start_pose[2] + (target_pose[2] - start_pose[2]) * alpha,
                float(target_pose[3]),
                float(target_pose[4]),
                float(target_pose[5]),
            ]
            if self._poses_match(current_pose, pose):
                continue
            self.move_to(pose, target_gripper)
            current_pose = list(pose)
            time.sleep(pause)
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        self.wait_for_service()
        start_pose = self.wait_for_pose()
        self.get_logger().info(f'Captured start pose: {start_pose}')

        # 1) Start pose: open gripper
        self.move_slowly(start_pose, GRIPPER_OPEN, steps=5, pause=0.08)

        # 2) Move to pick point with gripper still open, then close only at the pick pose
        self.move_slowly(PICK, GRIPPER_OPEN, steps=5, pause=0.08)
        self.move_to(PICK, GRIPPER_CLOSE)

        # 3) Air top pose: keep gripper closed
        self.move_slowly(AIR_TOP, GRIPPER_CLOSE, steps=5, pause=0.08)

        # 4) Drop pose: keep gripper closed until the target, then open there
        self.move_slowly(DROP, GRIPPER_CLOSE, steps=5, pause=0.08)
        self.move_to(DROP, GRIPPER_OPEN)

        # 5) Return to original start pose with gripper open
        self.move_slowly(start_pose, GRIPPER_OPEN, steps=5, pause=0.08)

        self.get_logger().info('Sequence finished.')


def main(args=None):
    rclpy.init(args=args)
    node = TeleopPoseSequence()
    try:
        node.run()
    except Exception as exc:
        node.get_logger().error(f'Sequence failed: {exc}')
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
