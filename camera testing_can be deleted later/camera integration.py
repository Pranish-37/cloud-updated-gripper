import cv2
import time

def open_camera(device_idx, is_realsense=False):
    cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
    
    if is_realsense:
        # Force MJPEG mode and resolution for RealSense RGB
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Error: Could not open /dev/video{device_idx}")
        return None

    # Warm-up loop: throw away initial 10 frames for exposure / buffer flush
    for _ in range(10):
        cap.read()
        time.sleep(0.02)

    return cap

print("Initializing cameras...")
cap_rs = open_camera(4, is_realsense=True)
cap_grip = open_camera(10, is_realsense=False)

if cap_rs and cap_grip:
    ret_rs, frame_rs = cap_rs.read()
    ret_grip, frame_grip = cap_grip.read()

    if ret_rs:
        cv2.imwrite("realsense_rgb_stable.jpg", frame_rs)
        print("RealSense RGB frame saved successfully!")
    
    if ret_grip:
        cv2.imwrite("gripper_feed_stable.jpg", frame_grip)
        print("Gripper frame saved successfully!")

    cap_rs.release()
    cap_grip.release()