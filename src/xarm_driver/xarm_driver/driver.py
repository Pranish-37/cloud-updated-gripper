"""
XArmDriver — low-level wrapper around xarm-python-sdk.
Ported from hardware/xarm_driver.py; no ROS dependency.
"""

import sys
import time

from xarm.wrapper import XArmAPI


class XArmDriver:
    def __init__(self, ip: str):
        self.ip = ip
        self.arm: XArmAPI | None = None
        self.xarm_gripper_min = -10
        self.xarm_gripper_max = 850
        self.xarm_gripper_pos = 0
        self.teaching_mode = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            self.arm = XArmAPI(self.ip)
            self.arm.motion_enable(enable=True)
            self.arm.set_gripper_mode(0) # 0: position control, 1: velocity control
            self.arm.set_gripper_enable(True) # True: enable gripper, False: disable gripper
            self.arm.set_mode(0)   # position control
            self.arm.clean_error() 
        except Exception as exc:
            print(f"[XArmDriver] Failed to initialize arm: {exc}")
            sys.exit(1)

        self.arm.set_state(state=0)  # sport state
        self.arm.set_gripper_enable(True)  # re-enable after state reset
        self.arm.set_gripper_position(pos=self.xarm_gripper_pos, wait=True)

    def disconnect(self) -> None:
        if self.arm is None:
            return
        try:
            self.arm.set_tgpio_digital(0, 0)
            self.arm.set_tgpio_digital(1, 0)
            time.sleep(0.5)
        except Exception:
            pass
        try:
            self.arm.disconnect()
        finally:
            self.arm = None

    @property
    def is_connected(self) -> bool:
        return self.arm is not None and self.arm.connected

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_robot_state(self, is_radian: bool = True) -> dict | None:
        """Return {'joints': (7,), 'ee_pose': (6,), 'gripper': float} or None on error."""
        pos = self.arm.get_position(is_radian=is_radian)
        if pos[0] != 0:
            print(f"[XArmDriver] get_position failed: code={pos[0]}")
            return None

        joint = self.arm.get_servo_angle(is_radian=is_radian)
        if joint[0] != 0:
            print(f"[XArmDriver] get_servo_angle failed: code={joint[0]}")
            return None

        return {
            "joints": joint[1],          # float list (7,)  radians
            "ee_pose": pos[1],           # float list (6,)  x,y,z mm + roll,pitch,yaw rad
            "gripper": self.xarm_gripper_pos,  # 0–850 raw
        }

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def move_to(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
        speed: float = 100.0,
        mvacc: float = 200.0,
        wait: bool = True,
        is_radian: bool = True,
    ) -> bool:
        if self.teaching_mode:
            raise RuntimeError("Cannot move arm in teaching mode.")
        code = self.arm.set_position(
            x, y, z, roll, pitch, yaw,
            speed=speed, mvacc=mvacc,
            wait=wait, is_radian=is_radian,
        )
        return code == 0

    def set_servo_cartesian(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
        is_radian: bool = True,
    ) -> bool:
        """Servo mode (xArm mode 1) — replaces current target immediately.
        Non-blocking; must be called at high rate (≥ 50 Hz) for smooth motion.
        Arm must be in mode 1 before calling (use set_arm_mode(1)).
        """
        if self.teaching_mode:
            raise RuntimeError("Cannot move arm in teaching mode.")
        code = self.arm.set_servo_cartesian(
            [x, y, z, roll, pitch, yaw],
            is_radian=is_radian,
        )
        return code == 0

    def set_arm_mode(self, mode: int) -> None:
        """Switch xArm control mode (0 = position, 1 = servo/real-time).
        Sequence: pause → set mode → resume.
        """
        self.arm.set_state(3)   # pause
        self.arm.set_mode(mode)
        self.arm.set_state(0)   # resume # return True if the move is successful, False otherwise

    def set_xarm_gripper(self, pos: float, wait: bool = True) -> None:
        if self.teaching_mode:
            raise RuntimeError("Cannot set gripper in teaching mode.")
        pos = max(self.xarm_gripper_min, min(pos, self.xarm_gripper_max))
        self.arm.set_gripper_position(pos=pos, wait=wait)
        self.xarm_gripper_pos = pos
