import json
import os
import time
from typing import Iterable, List, Tuple

import cv2
from xarm.wrapper import XArmAPI

# ================= HARDWARE CONFIGURATION =================
ROBOT_IP = "192.168.0.244"   # Replace with your xArm7 IP
CAM_REALSENSE_IDX = 4         # /dev/video4
CAM_GRIPPER_IDX = 10         # /dev/video10
RECORD_FPS = 30
DATASET_DIR = "/workspace/episodes"
PROMPT = (
    "Pick the white cube, place it in the tray, then pick the green cube and stack it on top of the white cube."
)
# =========================================================


def open_camera(device_idx: int, is_realsense: bool = False):
    """Open a camera using V4L2 and warm it up to clear stale frames."""
    cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device index {device_idx}.")

    if is_realsense:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up: clear stale/unexposed frames before real recording starts.
    for _ in range(10):
        cap.read()

    return cap


def settle_cameras(cap_rs, cap_gripper, settle_seconds: float = 2.0):
    """Let auto-exposure and auto-white-balance settle before the first real capture."""
    print("[INFO] Allowing cameras to settle for auto-exposure / white-balance convergence...")
    start = time.perf_counter()
    while time.perf_counter() - start < settle_seconds:
        cap_rs.read()
        cap_gripper.read()
        time.sleep(0.05)
    print("[INFO] Cameras settled.")


def _extract_value(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return [value]


def get_next_episode_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("episode_")]
    indices = []
    for name in existing:
        suffix = name.replace("episode_", "")
        if suffix.isdigit():
            indices.append(int(suffix))

    next_index = max(indices + [-1]) + 1
    episode_dir = os.path.join(base_dir, f"episode_{next_index:04d}")
    os.makedirs(episode_dir, exist_ok=False)
    os.makedirs(os.path.join(episode_dir, "cam_realsense"), exist_ok=True)
    os.makedirs(os.path.join(episode_dir, "cam_gripper"), exist_ok=True)
    return episode_dir


def read_robot_state(arm: XArmAPI):
    """Collect robot state needed for imitation learning: joints, EE pose, gripper."""
    try:
        code_j, joint_angles = arm.get_servo_angle()
    except Exception:
        code_j, joint_angles = -1, []

    try:
        code_p, ee_pose = arm.get_position()
    except Exception:
        code_p, ee_pose = -1, []

    try:
        code_g, gripper_pos = arm.get_gripper_position()
    except Exception:
        code_g, gripper_pos = -1, 0.0

    if code_j != 0:
        raise RuntimeError(f"Failed to read joint angles (code={code_j}).")
    if code_p != 0:
        raise RuntimeError(f"Failed to read end-effector pose (code={code_p}).")
    if code_g != 0:
        raise RuntimeError(f"Failed to read gripper position (code={code_g}).")

    return {
        "joint_positions": _extract_value(joint_angles),
        "ee_position": _extract_value(ee_pose),
        "gripper_position": float(gripper_pos),
    }


def capture_step(arm: XArmAPI, cap_rs, cap_grip, step: int, episode_dir: str):
    """Capture one synchronized sample from both cameras and robot state."""
    ret_rs, frame_rs = cap_rs.read()
    ret_gripper, frame_gripper = cap_grip.read()

    if not ret_rs or not ret_gripper:
        raise RuntimeError("Failed to read one or both camera frames.")

    robot_state = read_robot_state(arm)
    image_name_rs = f"frame_{step:05d}.jpg"
    image_name_gripper = f"frame_{step:05d}.jpg"

    cv2.imwrite(os.path.join(episode_dir, "cam_realsense", image_name_rs), frame_rs)
    cv2.imwrite(os.path.join(episode_dir, "cam_gripper", image_name_gripper), frame_gripper)

    step_record = {
        "step": step,
        "timestamp": time.time(),
        "joint_positions": robot_state["joint_positions"],
        "ee_position": robot_state["ee_position"],
        "gripper_position": robot_state["gripper_position"],
        "image_realsense": f"cam_realsense/{image_name_rs}",
        "image_gripper": f"cam_gripper/{image_name_gripper}",
    }

    return step_record


def record_episode():
    print("[INFO] Initializing dataset directory...")
    episode_dir = get_next_episode_dir(DATASET_DIR)

    print(f"[INFO] Using episode directory: {episode_dir}")

    print("[INFO] Connecting to xArm7...")
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    try:
        print("[INFO] Opening cameras...")
        cap_rs = open_camera(CAM_REALSENSE_IDX, is_realsense=True)
        cap_gripper = open_camera(CAM_GRIPPER_IDX, is_realsense=False)

        settle_cameras(cap_rs, cap_gripper, settle_seconds=2.0)

        print("[INFO] Camera warm-up and settling complete. Press ENTER to start recording.")
        input()

        print("\n[RECORDING] 30 Hz capture active. Press Ctrl+C to stop.\n")
        trajectory = []
        step = 0

        while True:
            loop_start = time.perf_counter()
            record = capture_step(arm, cap_rs, cap_gripper, step, episode_dir)
            trajectory.append(record)

            if step % 30 == 0:
                print(f"[INFO] Recorded step {step} ({step / RECORD_FPS:.1f}s elapsed)")

            step += 1
            elapsed = time.perf_counter() - loop_start
            sleep_time = 1.0 / RECORD_FPS - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[INFO] Stopping recording after {len(trajectory)} steps.")

        metadata = {
            "prompt": PROMPT,
            "episode_id": int(os.path.basename(episode_dir).split("_")[-1]),
            "fps": RECORD_FPS,
            "total_steps": len(trajectory),
            "trajectory": trajectory,
        }

        output_path = os.path.join(episode_dir, "trajectory.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Saved trajectory metadata to: {output_path}")

    finally:
        try:
            if "cap_rs" in locals():
                cap_rs.release()
            if "cap_gripper" in locals():
                cap_gripper.release()
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    record_episode()
