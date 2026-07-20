# ObsFrame — one assembled observation frame published by obs_aggregator.
#
# state  : float32 array, shape (11,) for 3-DoF or (14,) for 6-DoF
#            3-DoF: [joint_0..6 (rad), ee_x, ee_y, ee_z (mm), gripper (0-850)]
#            6-DoF: [joint_0..6 (rad), ee_x, ee_y, ee_z (mm), roll_norm, pitch, yaw (rad), gripper]
#
# image  : sensor_msgs/Image, encoding rgb8, shape (H, W, 3) uint8
#
# dof_mode: "3dof" or "6dof"
#
# timestamp_delta_ms : signed (t_image - t_state) in milliseconds, where
#                      t_state is ee_pose.header.stamp and t_image is the
#                      camera Image.header.stamp (original sensor stamps).

std_msgs/Header header
float32[] state
sensor_msgs/Image image
string dof_mode
float32 timestamp_delta_ms
