"""
inference_node — passive LeRobot policy wrapper for CloudGripper V2.

Services:
  inference/predict_action_chunk   cloudgripper_msgs/PredictActionChunk
  inference/reset                  std_srvs/Trigger

Parameters:
  policy_path          (string)  — path to pretrained_model/ (config.json + weights)
  dataset_meta_path    (string)  — dataset root or meta/ containing info.json
  device               (string)  — cuda or cpu (fallback to cpu with warning)
  chunk_size           (int)     — must match policy config (error on mismatch)
  n_action_steps       (int)     — compared to policy config (warning on mismatch)
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from cloudgripper_msgs.srv import PredictActionChunk

from inference_node.policy_runner import PolicyRunner


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('policy_path', '')
        self.declare_parameter('dataset_meta_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('chunk_size', 16)
        self.declare_parameter('n_action_steps', 8)

        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        dataset_meta_path = self.get_parameter('dataset_meta_path').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        chunk_size = self.get_parameter('chunk_size').get_parameter_value().integer_value
        n_action_steps = self.get_parameter('n_action_steps').get_parameter_value().integer_value

        if not policy_path:
            raise RuntimeError("Parameter 'policy_path' must be set.")
        if not dataset_meta_path:
            raise RuntimeError("Parameter 'dataset_meta_path' must be set.")

        self.get_logger().info(f'Loading policy from {policy_path}')
        self._runner = PolicyRunner(
            policy_path=policy_path,
            dataset_meta_path=dataset_meta_path,
            device=device,
            yaml_chunk_size=chunk_size,
            yaml_n_action_steps=n_action_steps,
        )

        self.create_service(
            PredictActionChunk,
            'inference/predict_action_chunk',
            self._on_predict,
        )
        self.create_service(
            Trigger,
            'inference/reset',
            self._on_reset,
        )

        self.get_logger().info(
            f'InferenceNode ready — chunk_size={self._runner.chunk_size}, '
            f'action_dim={self._runner.action_dim}.'
        )

    def _on_predict(
        self,
        request: PredictActionChunk.Request,
        response: PredictActionChunk.Response,
    ) -> PredictActionChunk.Response:
        try:
            chunk = self._runner.predict_chunk(request.frame)
            response.success = True
            response.message = 'ok'
            response.chunk_size = chunk.shape[0]
            response.action_dim = chunk.shape[1]
            response.actions = chunk.reshape(-1).tolist()
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.actions = []
            response.chunk_size = 0
            response.action_dim = 0
            self.get_logger().error(f'predict_action_chunk failed: {exc}')
        return response

    def _on_reset(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        try:
            self._runner.reset()
            response.success = True
            response.message = 'Policy and processors reset.'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'inference/reset failed: {exc}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
