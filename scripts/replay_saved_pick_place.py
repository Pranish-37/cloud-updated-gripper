#!/usr/bin/env python3
"""
Replay the saved pick/place joint angles from a JSON file.

This script reads the file saved by record_pick_place_positions.py and moves the arm
through the sequence:
  start -> pick -> air -> drop -> start

It uses the xArm SDK directly and moves slowly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from xarm.wrapper import XArmAPI

REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_PATH = REPO_ROOT / 'scripts' / 'pick_place_positions.json'

ROBOT_IP = '192.168.0.244'
SLOW_SPEED = 10.0
SLOW_MVACC = 50.0


def deg_to_rad(values):
    return [math.radians(float(v)) for v in values]


def load_positions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'No saved positions found at {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key in ('start', 'pick', 'air', 'drop'):
        if data.get(key) is None:
            raise ValueError(f'Missing saved joint state: {key}')

    return data


def connect_arm(ip: str = ROBOT_IP):
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_gripper_mode(0)
    arm.set_gripper_enable(True)
    arm.set_mode(0)
    arm.clean_error()
    arm.set_state(state=0)
    arm.set_gripper_position(pos=0, wait=True)
    return arm


def move_to_joint_angles(arm: XArmAPI, angles_deg, speed: float = SLOW_SPEED, wait: bool = True):
    """xArm SDK requires a single joint ID + single angle value per call."""
    joint_rad = deg_to_rad(angles_deg)
    for servo_id, angle_rad in enumerate(joint_rad, start=1):
        code = arm.set_servo_angle(
            servo_id=servo_id,
            angle=angle_rad,
            speed=speed,
            mvacc=SLOW_MVACC,
            wait=False,
            is_radian=True,
        )
        if code != 0:
            raise RuntimeError(f'Joint {servo_id} move failed with code={code}.')
    if wait:
        arm.wait_move()
    return 0


def set_gripper(arm: XArmAPI, pos: float, wait: bool = True):
    code = arm.set_gripper_position(pos=pos, wait=wait)
    if code != 0:
        raise RuntimeError(f'Gripper command failed with code={code}.')


def main():
    data = load_positions(SAVE_PATH)
    arm = connect_arm()
    try:
        print('Loaded recorded positions from:', SAVE_PATH)
        print('start:', data['start'])
        print('pick:', data['pick'])
        print('air:', data['air'])
        print('drop:', data['drop'])

        print('Step 1: move to start pose')
        move_to_joint_angles(arm, data['start'])

        print('Step 2: move to pick pose and close gripper')
        move_to_joint_angles(arm, data['pick'])
        set_gripper(arm, data.get('gripper_close', 450), wait=True)

        print('Step 3: move to air transfer pose with gripper still closed')
        move_to_joint_angles(arm, data['air'])

        print('Step 4: move to drop pose and open gripper')
        move_to_joint_angles(arm, data['drop'])
        set_gripper(arm, data.get('gripper_open', 590), wait=True)

        print('Step 5: return to start pose')
        move_to_joint_angles(arm, data['start'])
        print('Sequence finished.')
    finally:
        arm.disconnect()


if __name__ == '__main__':
    main()
