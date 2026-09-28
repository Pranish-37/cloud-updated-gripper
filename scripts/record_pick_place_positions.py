#!/usr/bin/env python3
"""
Record the robot joint poses during teleop.

Usage:
  1) Start the ROS stack and teleop node.
  2) Run this script in the same ROS environment.
  3) Move the robot to the desired pose with teleop.
  4) Press:
       s = save current pose as START
       1 = save current pose as PICK
       2 = save current pose as AIR_TRANSFER
       3 = save current pose as DROP
       q = quit and save file

The values are saved to:
  scripts/pick_place_positions.json

The saved file format is JSON with the joint poses in degrees:
  {
    "start": [j1, j2, ..., j7],
    "pick": [...],
    "air": [...],
    "drop": [...],
    "gripper_close": 450,
    "gripper_open": 590
  }
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_PATH = REPO_ROOT / 'scripts' / 'pick_place_positions.json'


class JointRecorder(Node):
    def __init__(self):
        super().__init__('joint_position_recorder')
        self._latest_joints = None
        self._sub = self.create_subscription(JointState, 'robot/joint_states', self._on_joints, 10)
        self._saved = {
            'start': None,
            'pick': None,
            'air': None,
            'drop': None,
            'gripper_close': 450,
            'gripper_open': 590,
        }

    def _on_joints(self, msg: JointState) -> None:
        if len(msg.position) < 7:
            return
        self._latest_joints = [float(v) for v in msg.position[:7]]

    def _angles_deg(self):
        if self._latest_joints is None:
            return None
        return [float((v * 180.0 / 3.141592653589793)) for v in self._latest_joints]

    def prompt(self):
        while rclpy.ok():
            try:
                key = input(
                    '\nChoose one: s=start, 1=pick, 2=air, 3=drop, q=quit and save\n> '
                ).strip().lower()
            except EOFError:
                print('\nNo input detected; quitting.')
                break

            if key == 'q':
                self._save_file()
                print(f'Saved file: {SAVE_PATH}')
                return

            if self._latest_joints is None:
                print('No joint state received yet; move the robot first.')
                continue

            current_deg = self._angles_deg()
            print(f'Current joint angles (deg): {current_deg}')

            if key == 's':
                self._saved['start'] = current_deg
                print('Saved START pose.')
            elif key == '1':
                self._saved['pick'] = current_deg
                print('Saved PICK pose.')
            elif key == '2':
                self._saved['air'] = current_deg
                print('Saved AIR pose.')
            elif key == '3':
                self._saved['drop'] = current_deg
                print('Saved DROP pose.')
            else:
                print('Unknown key. Use s, 1, 2, 3, q.')

    def _save_file(self):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self._saved, f, indent=2)


def main(args=None):
    rclpy.init(args=args)
    node = JointRecorder()
    try:
        print('Recording joint states...')
        print('Move the robot with teleop, then press the key to save.')
        print('Keys: s=start, 1=pick, 2=air, 3=drop, q=quit+')
        node.prompt()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
