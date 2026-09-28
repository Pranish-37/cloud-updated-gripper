#!/usr/bin/env python3
"""
Slow joint-space pick/place script for the xArm.

This version matches your requested motion pattern:
  1) read the current joint pose when the script starts
  2) move to state 1 (pick position) and close gripper to 450
  3) move to state 2 (air / transfer position) while gripper stays closed
  4) move to state 3 (drop position) and open gripper to 590
  5) return to the original start pose

The robot is commanded at a slow speed to keep the motion stable.

Notes:
  - All angles below are in degrees.
  - The xArm SDK expects joint values internally in radians, so the script converts
    them before sending commands.
"""

from __future__ import annotations

import math
import time

from xarm.wrapper import XArmAPI

ROBOT_IP = '192.168.0.244'
SLOW_SPEED = 10.0
SLOW_MVACC = 50.0
GRIPPER_CLOSE = 450.0
GRIPPER_OPEN = 590.0

# Your joint states in degrees
STATE_1 = [9.6, 5.6, 9.5, 46.1, -3.9, 33.3, 22.8]
STATE_2 = [3.7, 2.3, 4.7, 71.9, 2.9, 62.7, 4.6]
STATE_3 = [-15.2, 31.5, -3.9, 86.6, 2.5, 52.0, -22.3]


def deg_to_rad(values):
    return [math.radians(float(v)) for v in values]


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


def get_current_joint_angles(arm: XArmAPI):
    resp = arm.get_servo_angle(is_radian=True)
    if resp[0] != 0:
        raise RuntimeError(f'Failed to read current joint angles: code={resp[0]}')
    return [float(v) for v in resp[1]]


def move_to_joint_angles(arm: XArmAPI, angles_deg, speed: float = SLOW_SPEED, wait: bool = True):
    """xArm SDK requires one joint ID + one angle value per call.
    Passing a full 7-joint list to set_servo_angle() causes the TypeError:
    '<=' not supported between instances of 'int' and 'list'.
    """
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
    arm = connect_arm()
    try:
        start_angles_rad = get_current_joint_angles(arm)
        start_angles_deg = [math.degrees(v) for v in start_angles_rad]
        print('Captured start joint angles (rad):', start_angles_rad)
        print('Captured start joint angles (deg):', start_angles_deg)

        print('Step 1: move to pick position')
        move_to_joint_angles(arm, STATE_1)
        set_gripper(arm, GRIPPER_CLOSE)
        time.sleep(0.5)

        print('Step 2: move to air transfer position, gripper still closed')
        move_to_joint_angles(arm, STATE_2)
        time.sleep(0.5)

        print('Step 3: move to drop position and open gripper')
        move_to_joint_angles(arm, STATE_3)
        set_gripper(arm, GRIPPER_OPEN)
        time.sleep(0.5)

        print('Step 4: return to the live start position')
        move_to_joint_angles(arm, start_angles_deg)
        time.sleep(0.5)

        print('Sequence complete.')
    finally:
        arm.disconnect()


if __name__ == '__main__':
    main()
