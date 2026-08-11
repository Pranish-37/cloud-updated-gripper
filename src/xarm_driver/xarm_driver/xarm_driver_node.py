"""
xarm_driver_node — ROS 2 node that wraps XArmDriver.

Publishes (robot_publish_rate Hz, default 50) — relative topic names, affected by ROS 2 namespace:
  robot/joint_states   sensor_msgs/JointState
  robot/ee_pose        geometry_msgs/PoseStamped   (position mm, orientation as quaternion)
  robot/gripper_pos    std_msgs/Float64             (0-850 raw)

Subscribes — relative topic name:
  policy/action_cmd    std_msgs/Float64MultiArray   (7-vector: x,y,z mm · roll,pitch,yaw rad · gripper)

Parameters:
  robot_ip / robot_name / robot_publish_rate / gripper_deadband / servo_mode / robot_home_position
  x/y/z_limits — workspace envelope [min,max] mm (from config/safety.yaml)
  max_step_x/y/z_mm / max_step_roll/pitch/yaw_deg
      — absolute per-command step caps on /policy/action_cmd only (safety hard gate)

Services:
  robot/set_servo_mode     std_srvs/SetBool
  robot/move_home          std_srvs/Trigger
  robot/execute_action     cloudgripper_msgs/ExecuteAction
      — blocking move+gripper (same as old real_e2e send_action wait=True)

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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from cloudgripper_msgs.srv import ExecuteAction
from xarm_driver.driver import XArmDriver

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


def _clamp_delta(prev: float, new: float, max_delta: float) -> float:
    d = new - prev
    if abs(d) <= max_delta:
        return new
    return prev + math.copysign(max_delta, d)


def _clamp_delta_angle_rad(prev: float, new: float, max_delta: float) -> float:
    """Limit angular step using shortest-path delta (radians)."""
    d = (new - prev + math.pi) % (2.0 * math.pi) - math.pi
    if abs(d) <= max_delta:
        return prev + d
    return prev + math.copysign(max_delta, d)


class XArmDriverNode(Node):
    def __init__(self):
        super().__init__('xarm_driver_node')

        self.declare_parameter('robot_ip', '192.168.0.244')
        self.declare_parameter('robot_name', 'robot')
        # Prefer robot_publish_rate so /** yaml publish_rate (obs_aggregator) does not collide.
        self.declare_parameter('robot_publish_rate', 50.0)
        self.declare_parameter('gripper_deadband', 20.0)
        self.declare_parameter('servo_mode', False)
        self.declare_parameter('robot_home_position', [330.0, 0.0, 160.0, 3.14159, 0.0, 0.0, 850.0])
        # Workspace envelope (mm) — shared with teleop via config/safety.yaml
        self.declare_parameter('x_limits', [200.0, 700.0])
        self.declare_parameter('y_limits', [-400.0, 400.0])
        self.declare_parameter('z_limits', [6.0, 600.0])
        # Absolute per-command step caps for /policy/action_cmd (mm / deg).
        self.declare_parameter('max_step_x_mm', 5.0)
        self.declare_parameter('max_step_y_mm', 5.0)
        self.declare_parameter('max_step_z_mm', 5.0)
        self.declare_parameter('max_step_roll_deg', 2.0)
        self.declare_parameter('max_step_pitch_deg', 2.0)
        self.declare_parameter('max_step_yaw_deg', 2.0)

        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self._robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        rate = self.get_parameter('robot_publish_rate').get_parameter_value().double_value
        self._deadband = self.get_parameter('gripper_deadband').get_parameter_value().double_value
        self._servo_mode = self.get_parameter('servo_mode').get_parameter_value().bool_value
        self._home_position = list(
            self.get_parameter('robot_home_position').get_parameter_value().double_array_value
        )  # [x, y, z, roll, pitch, yaw (rad), gripper]

        def _f(name: str) -> float:
            return self.get_parameter(name).get_parameter_value().double_value

        def _lim(name: str) -> tuple[float, float]:
            v = list(self.get_parameter(name).get_parameter_value().double_array_value)
            if len(v) != 2:
                raise ValueError(f'{name} must be [min, max], got {v}')
            lo, hi = float(v[0]), float(v[1])
            if lo > hi:
                raise ValueError(f'{name}: min ({lo}) > max ({hi})')
            return lo, hi

        self._x_limits = _lim('x_limits')
        self._y_limits = _lim('y_limits')
        self._z_limits = _lim('z_limits')

        self._max_step_x = _f('max_step_x_mm')
        self._max_step_y = _f('max_step_y_mm')
        self._max_step_z = _f('max_step_z_mm')
        self._max_step_roll = math.radians(_f('max_step_roll_deg'))
        self._max_step_pitch = math.radians(_f('max_step_pitch_deg'))
        self._max_step_yaw = math.radians(_f('max_step_yaw_deg'))

        # Last pose actually sent on /policy/action_cmd; None until first seed.
        self._last_cmd_pose: list[float] | None = None

        # True while move_home is executing — blocks _on_action_cmd to avoid conflicts
        self._resetting = False

        # --- driver ---
        self._driver = XArmDriver(robot_ip)
        self._driver.connect()

        # Initial reset to home — runs in position mode before servo mode activates,
        # so the state publisher starts only after the robot is at the home pose.
        self._move_to_home_blocking()

        if self._servo_mode:
            self._driver.set_arm_mode(1)
            self.get_logger().info('Servo mode (mode 1) active — real-time streaming.')
        self.get_logger().info(
            f'Connected to xArm at {robot_ip} — state publish {rate} Hz'
        )
        self.get_logger().info(
            f'Workspace — xyz mm: x={self._x_limits}, y={self._y_limits}, '
            f'z={self._z_limits}'
        )
        self.get_logger().info(
            f'action_cmd max step — xyz mm: '
            f'({self._max_step_x}, {self._max_step_y}, {self._max_step_z}); '
            f'rpy deg: ({math.degrees(self._max_step_roll)}, '
            f'{math.degrees(self._max_step_pitch)}, '
            f'{math.degrees(self._max_step_yaw)})'
        )

        # Callback groups so the state-publish timer can fire while a blocking
        # execute_action / move_home service call is in progress.
        # _state_cbg  : dedicated thread for the 50 Hz timer → never starved
        # _robot_cbg  : single thread for all robot-mutating calls (mutual exclusion)
        self._state_cbg = MutuallyExclusiveCallbackGroup()
        self._robot_cbg = MutuallyExclusiveCallbackGroup()

        # --- publishers (relative names → scoped by namespace for multi-robot) ---
        self._pub_joints = self.create_publisher(JointState, 'robot/joint_states', 1)
        self._pub_ee = self.create_publisher(PoseStamped, 'robot/ee_pose', 1)
        self._pub_gripper = self.create_publisher(Float64, 'robot/gripper_pos', 1)

        # --- subscriber (relative name → scoped by namespace) ---
        self._sub_action = self.create_subscription(
            Float64MultiArray,
            'policy/action_cmd',
            self._on_action_cmd,
            1,
            callback_group=self._robot_cbg,
        )

        # --- servo mode switch service ---
        self._mode_srv = self.create_service(
            SetBool,
            'robot/set_servo_mode',
            self._on_set_servo_mode,
            callback_group=self._robot_cbg,
        )

        # --- move-to-home service ---
        self._home_srv = self.create_service(
            Trigger,
            'robot/move_home',
            self._on_move_home,
            callback_group=self._robot_cbg,
        )

        # --- blocking execute (policy eval) ---
        self._exec_srv = self.create_service(
            ExecuteAction,
            'robot/execute_action',
            self._on_execute_action,
            callback_group=self._robot_cbg,
        )

        # --- state publish timer (dedicated group — runs even during blocking moves) ---
        self._timer = self.create_timer(1.0 / rate, self._publish_state,
                                        callback_group=self._state_cbg)

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

    def _seed_last_cmd_from_feedback(self) -> bool:
        """Seed last commanded pose from current EE feedback. Returns False on failure."""
        state = self._driver.get_robot_state(is_radian=True)
        if state is None:
            self.get_logger().warn(
                'action_cmd: cannot seed step limiter from feedback — ignoring command.'
            )
            return False
        ee = state['ee_pose']
        self._last_cmd_pose = [float(ee[i]) for i in range(6)]
        self.get_logger().info(
            f'action_cmd step limiter seeded from feedback: '
            f'xyz=({self._last_cmd_pose[0]:.1f},{self._last_cmd_pose[1]:.1f},'
            f'{self._last_cmd_pose[2]:.1f}) mm'
        )
        return True

    def _clamp_action_cmd_pose(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Clamp incoming action_cmd pose vs last commanded pose (absolute step caps)."""
        assert self._last_cmd_pose is not None
        px, py, pz, pr, pp, pyaw = self._last_cmd_pose
        return (
            _clamp_delta(px, x, self._max_step_x),
            _clamp_delta(py, y, self._max_step_y),
            _clamp_delta(pz, z, self._max_step_z),
            _clamp_delta_angle_rad(pr, roll, self._max_step_roll),
            _clamp_delta_angle_rad(pp, pitch, self._max_step_pitch),
            _clamp_delta_angle_rad(pyaw, yaw, self._max_step_yaw),
        )

    def _apply_cartesian_gripper(
        self, x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float, gripper: float,
        *, wait: bool,
    ) -> tuple[bool, str]:
        """Shared move path for topic callback and ExecuteAction service."""
        if not (self._x_limits[0] <= x <= self._x_limits[1]):
            return False, f'x={x:.1f} mm out of workspace [{self._x_limits}]'
        if not (self._y_limits[0] <= y <= self._y_limits[1]):
            return False, f'y={y:.1f} mm out of workspace [{self._y_limits}]'
        if not (self._z_limits[0] <= z <= self._z_limits[1]):
            return False, f'z={z:.1f} mm out of workspace [{self._z_limits}]'

        if self._servo_mode:
            ok = self._driver.set_servo_cartesian(x, y, z, roll, pitch, yaw, is_radian=True)
        else:
            ok = self._driver.move_to(
                x, y, z, roll, pitch, yaw, wait=wait, is_radian=True,
            )
        if not ok:
            return False, 'move command returned failure code'

        # In servo/teleop streaming, never block on the gripper — a blocking
        # set_gripper_position stalls this callback while teleop keeps integrating
        # SpaceMouse into a far-away target, then the next servo cmd 爆冲.
        gripper_wait = False if self._servo_mode else wait
        if abs(gripper - self._driver.xarm_gripper_pos) > self._deadband:
            self._driver.set_xarm_gripper(gripper, wait=gripper_wait)
        return True, 'ok'

    def _on_action_cmd(self, msg: Float64MultiArray) -> None:
        if self._resetting:
            return
        data = list(msg.data)
        if len(data) != 7:
            self.get_logger().error(
                f'/policy/action_cmd expects 7-vector, got {len(data)} — ignoring.'
            )
            return

        x, y, z, roll, pitch, yaw, gripper = (float(v) for v in data)

        # First streaming command: seed limiter from live feedback, then clamp.
        if self._last_cmd_pose is None:
            if not self._seed_last_cmd_from_feedback():
                return

        x, y, z, roll, pitch, yaw = self._clamp_action_cmd_pose(
            x, y, z, roll, pitch, yaw,
        )

        # Streaming topic: never wait — teleop/servo must stay real-time.
        ok, msg_txt = self._apply_cartesian_gripper(
            x, y, z, roll, pitch, yaw, gripper, wait=False,
        )
        if not ok:
            self.get_logger().warn(msg_txt)
            return

        # Only advance the command chain after a successful send.
        self._last_cmd_pose = [x, y, z, roll, pitch, yaw]

    def _on_execute_action(
        self,
        request: ExecuteAction.Request,
        response: ExecuteAction.Response,
    ) -> ExecuteAction.Response:
        """Blocking policy step — same semantics as old send_action(wait=True)."""
        if self._resetting:
            response.success = False
            response.message = 'Busy resetting / moving home.'
            return response

        data = list(request.command)
        if len(data) != 7:
            response.success = False
            response.message = f'command must be length 7, got {len(data)}'
            return response

        x, y, z, roll, pitch, yaw, gripper = (float(v) for v in data)
        ok, msg_txt = self._apply_cartesian_gripper(
            x, y, z, roll, pitch, yaw, gripper, wait=True,
        )
        response.success = ok
        response.message = msg_txt
        if not ok:
            self.get_logger().warn(f'execute_action: {msg_txt}')
        return response

    # ------------------------------------------------------------------
    # Move-home (shared logic used at startup and via service)
    # ------------------------------------------------------------------

    def _move_to_home_blocking(self) -> None:
        """Move arm + gripper to home position in position mode (blocking)."""
        hp = self._home_position
        if len(hp) != 7:
            self.get_logger().error(
                f'robot_home_position must have 7 elements, got {len(hp)} — skipping init move.'
            )
            return
        self.get_logger().info(
            f'Moving to home position: xyz=({hp[0]},{hp[1]},{hp[2]}) mm  '
            f'rpy=({hp[3]:.4f},{hp[4]:.4f},{hp[5]:.4f}) rad  gripper={hp[6]}'
        )
        self._driver.move_to(
            hp[0], hp[1], hp[2], hp[3], hp[4], hp[5],
            speed=100.0, mvacc=200.0,
            wait=True, is_radian=True,
        )
        self._driver.set_xarm_gripper(hp[6], wait=True)
        self.get_logger().info('Home position reached.')

    def _on_move_home(self, _req: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Block action_cmd, move arm+gripper to home position, then restore mode."""
        self._resetting = True
        try:
            self._driver.set_arm_mode(0)
            self._move_to_home_blocking()
            if self._servo_mode:
                self._driver.set_arm_mode(1)
            hp = self._home_position
            response.success = True
            response.message = (
                f'Reset to home: xyz=({hp[0]},{hp[1]},{hp[2]}) mm  '
                f'rpy=({hp[3]:.4f},{hp[4]:.4f},{hp[5]:.4f}) rad  gripper={hp[6]}'
            )
            self.get_logger().info(f'[move_home] {response.message}')
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'[move_home] failed: {exc}')
        finally:
            # Home moved outside the action_cmd chain — force re-seed from feedback.
            self._last_cmd_pose = None
            self._resetting = False
        return response

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_set_servo_mode(self, request: SetBool.Request, response: SetBool.Response):
        """Switch between servo mode (True) and position mode (False) at runtime."""
        target_mode = 1 if request.data else 0
        try:
            self._driver.set_arm_mode(target_mode)
            self._servo_mode = request.data
            response.success = True
            response.message = f'Switched to {"servo" if request.data else "position"} mode.'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f'Mode switch failed: {e}')
        return response

    def destroy_node(self) -> None:
        self.get_logger().info('Shutting down — disconnecting from xArm.')
        if self._servo_mode: # switch to position mode before disconnecting
            try:
                self._driver.set_arm_mode(0)
            except Exception: # log the error
                pass
        self._driver.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmDriverNode()
    # MultiThreadedExecutor lets _publish_state (state_cbg) fire concurrently
    # with blocking execute_action / move_home calls (robot_cbg).
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
