"""
ae_awb_controller_node — AE/AWB convergence and lock controller for realsense2_camera.

Mirrors the LeRobot RealSenseCameraConfig convergence + snapshot logic:
  1. AE and AWB are enabled at startup (realsense2_camera starts with them on).
  2. Call /camera/lock_exposure (std_srvs/Trigger) after the camera has converged
     (typically a few seconds after startup).
  3. The service reads the current exposure and gain values, then disables
     auto-exposure and auto-white-balance on the realsense2_camera node via its
     SetParameters service — freezing the values at the converged snapshot.

Parameters:
  realsense_node_name  (str, default 'camera')
      The node name used for realsense2_camera. The SetParameters service is
      called at '<realsense_node_name>/set_parameters'.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters
from std_srvs.srv import Trigger


class AeAwbControllerNode(Node):
    def __init__(self):
        super().__init__('ae_awb_controller')

        self.declare_parameter('realsense_node_name', 'camera')
        rs_name = self.get_parameter('realsense_node_name').get_parameter_value().string_value

        # Fully-qualified service paths on the realsense2_camera node
        self._set_params_srv = f'{rs_name}/set_parameters'
        self._get_params_srv = f'{rs_name}/get_parameters'

        # Clients for realsense2_camera parameter services
        self._set_client = self.create_client(SetParameters, self._set_params_srv)
        self._get_client = self.create_client(GetParameters, self._get_params_srv)

        # Expose lock service (relative → scoped by namespace)
        self._lock_srv = self.create_service(
            Trigger,
            'camera/lock_exposure',
            self._on_lock_exposure,
        )

        self.get_logger().info(
            f'AE/AWB controller ready. '
            f'Call "camera/lock_exposure" after convergence to freeze exposure.'
        )

    # ------------------------------------------------------------------
    # Lock service
    # ------------------------------------------------------------------

    def _on_lock_exposure(self, request: Trigger.Request, response: Trigger.Response):
        """Read current exposure/gain, then disable AE and AWB."""
        if not self._set_client.wait_for_service(timeout_sec=3.0):
            response.success = False
            response.message = f'SetParameters service not available: {self._set_params_srv}'
            self.get_logger().error(response.message)
            return response

        if not self._get_client.wait_for_service(timeout_sec=3.0):
            response.success = False
            response.message = f'GetParameters service not available: {self._get_params_srv}'
            self.get_logger().error(response.message)
            return response

        # --- Step 1: snapshot current exposure and gain ---
        get_req = GetParameters.Request()
        get_req.names = ['color_exposure', 'color_gain']
        get_future = self._get_client.call_async(get_req)
        rclpy.spin_until_future_complete(self, get_future, timeout_sec=5.0)

        if get_future.result() is None:
            response.success = False
            response.message = 'Failed to read exposure/gain from realsense2_camera.'
            self.get_logger().error(response.message)
            return response

        values = get_future.result().values
        exposure = values[0].integer_value if values[0].type == ParameterType.PARAMETER_INTEGER else -1
        gain = values[1].integer_value if values[1].type == ParameterType.PARAMETER_INTEGER else -1
        self.get_logger().info(f'Snapshotting — exposure={exposure}, gain={gain}')

        # --- Step 2: disable AE and AWB ---
        def _bool_param(name: str, value: bool) -> Parameter:
            p = Parameter()
            p.name = name
            p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
            return p

        set_req = SetParameters.Request()
        set_req.parameters = [
            _bool_param('color_auto_exposure', False),
            _bool_param('color_auto_white_balance', False),
        ]
        set_future = self._set_client.call_async(set_req)
        rclpy.spin_until_future_complete(self, set_future, timeout_sec=5.0)

        if set_future.result() is None:
            response.success = False
            response.message = 'SetParameters call timed out.'
            self.get_logger().error(response.message)
            return response

        results = set_future.result().results
        failed = [r for r in results if not r.successful]
        if failed:
            reasons = '; '.join(r.reason for r in failed)
            response.success = False
            response.message = f'Failed to lock some parameters: {reasons}'
            self.get_logger().error(response.message)
            return response

        response.success = True
        response.message = f'AE/AWB locked. exposure={exposure}, gain={gain}.'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AeAwbControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
