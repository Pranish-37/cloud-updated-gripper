"""
xarm_driver_node — ROS 2 node that wraps XArmDriver.

Publishes (15 Hz) — relative topic names, affected by ROS 2 namespace:
  robot/joint_states   sensor_msgs/JointState
  robot/ee_pose        geometry_msgs/PoseStamped   (position mm, orientation as quaternion)
  robot/gripper_pos    std_msgs/Float64             (0-850 raw)

Subscribes — relative topic name:
  policy/action_cmd    std_msgs/Float64MultiArray   (7-vector: x,y,z mm · roll,pitch,yaw rad · gripper)

Parameters:
  robot_ip        (str,   default '192.168.0.244')
  robot_name      (str,   default 'robot')          — used as TF frame_id prefix
  publish_rate    (float, default 15.0  Hz)
  gripper_deadband(float, default 20.0  raw units)

Multi-robot usage — launch each instance with its own namespace, e.g.:
  ros2 run xarm_driver xarm_driver_node --ros-args \
      -r __ns:=/robot_1 -p robot_ip:=192.168.0.244 -p robot_name:=robot_1
  ros2 run xarm_driver xarm_driver_node --ros-args \
      -r __ns:=/robot_2 -p robot_ip:=192.168.0.245 -p robot_name:=robot_2
Topics will be scoped automatically: /robot_1/robot/joint_states, /robot_2/robot/joint_states, etc.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from xarm_driver.driver import XArmDriver

# Workspace limits (mm)
_WS = {
    'x': (200.0, 700.0),
    'y': (-400.0, 400.0),
    'z': (6.0, 600.0),
}

JOINT_NAMES = [f'joint_{i}' for i in range(7)]


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """ZYX convention → (x, y, z, w)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class XArmDriverNode(Node):
    def __init__(self):
        super().__init__('xarm_driver_node')

        self.declare_parameter('robot_ip', '192.168.0.244')
        self.declare_parameter('robot_name', 'robot')
        self.declare_parameter('publish_rate', 15.0)
        self.declare_parameter('gripper_deadband', 20.0)

        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self._robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self._deadband = self.get_parameter('gripper_deadband').get_parameter_value().double_value

        # --- driver ---
        self._driver = XArmDriver(robot_ip)
        self._driver.connect()
        self.get_logger().info(f'Connected to xArm at {robot_ip}') # log the connection status

        # --- publishers (relative names → scoped by namespace for multi-robot) ---
        self._pub_joints = self.create_publisher(JointState, 'robot/joint_states', 1) # use relative dir, scoped by namespace for multi-robot
        self._pub_ee = self.create_publisher(PoseStamped, 'robot/ee_pose', 1) # use relative dir, scoped by namespace for multi-robot
        self._pub_gripper = self.create_publisher(Float64, 'robot/gripper_pos', 1) # use relative dir, scoped by namespace for multi-robot

        # --- subscriber (relative name → scoped by namespace) ---
        self._sub_action = self.create_subscription(
            Float64MultiArray,
            'policy/action_cmd',
            self._on_action_cmd,
            1,
        )

        # --- state publish timer ---
        self._timer = self.create_timer(1.0 / rate, self._publish_state) # create a timer for the state publishing, it will be called at the publish rate

    # ------------------------------------------------------------------
    # State publishing
    # ------------------------------------------------------------------

    def _publish_state(self) -> None:
        state = self._driver.get_robot_state(is_radian=True)
        if state is None:
            self.get_logger().warn('Failed to read robot state; skipping publish.')
            return

        now = self.get_clock().now().to_msg() # get the current time
        joints = state['joints']   # (7,) radians
        ee = state['ee_pose']      # (6,) x,y,z mm · roll,pitch,yaw rad
        gripper = state['gripper'] # 0–850

        # JointState
        js = JointState()
        js.header.stamp = now
        js.header.frame_id = f'{self._robot_name}/base_link'
        js.name = JOINT_NAMES
        js.position = [float(j) for j in joints[:7]] # convert the joint angles to float
        self._pub_joints.publish(js)

        # PoseStamped — position in mm, orientation as quaternion
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = f'{self._robot_name}/base_link'
        ps.pose.position.x = float(ee[0])
        ps.pose.position.y = float(ee[1])
        ps.pose.position.z = float(ee[2])
        qx, qy, qz, qw = _euler_to_quaternion(float(ee[3]), float(ee[4]), float(ee[5]))
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self._pub_ee.publish(ps) # publish the end effector pose

        # Gripper
        gp = Float64()
        gp.data = float(gripper)
        self._pub_gripper.publish(gp) # publish the gripper position

    # ------------------------------------------------------------------
    # Action command callback
    # ------------------------------------------------------------------

    def _on_action_cmd(self, msg: Float64MultiArray) -> None: # callback function for the action command, it will be called when the action command is received
        data = list(msg.data)
        if len(data) != 7: # check if the action command is a 7-vector: x,y,z mm  roll,pitch,yaw rad  gripper
            self.get_logger().error(
                f'/policy/action_cmd expects 7-vector, got {len(data)} — ignoring.' # log the error
            )
            return

        x, y, z, roll, pitch, yaw, gripper = data # unpack the action command

        # Workspace limit check
        if not (_WS['x'][0] <= x <= _WS['x'][1]): # check if the x coordinate is within the workspace limits
            self.get_logger().warn(f'x={x:.1f} mm out of workspace [{_WS["x"]}] — cmd rejected.')
            return
        if not (_WS['y'][0] <= y <= _WS['y'][1]):
            self.get_logger().warn(f'y={y:.1f} mm out of workspace [{_WS["y"]}] — cmd rejected.')
            return
        if not (_WS['z'][0] <= z <= _WS['z'][1]):
            self.get_logger().warn(f'z={z:.1f} mm out of workspace [{_WS["z"]}] — cmd rejected.')
            return

        # Cartesian move
        ok = self._driver.move_to(x, y, z, roll, pitch, yaw, wait=True, is_radian=True)
        if not ok:
            self.get_logger().warn('move_to returned failure code.')

        # Gripper with dead-band
        if abs(gripper - self._driver.xarm_gripper_pos) > self._deadband:
            self._driver.set_xarm_gripper(gripper, wait=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self.get_logger().info('Shutting down — disconnecting from xArm.')
        self._driver.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args) # initialize the ROS client library
    node = XArmDriverNode()
    try:
        rclpy.spin(node) # spin the node, it will run until the node is stopped
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
