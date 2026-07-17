"""LeRobot policy loading and chunk inference for CloudGripper V2."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from cloudgripper_msgs.msg import ObsFrame

logger = logging.getLogger(__name__)


def resolve_dataset_root(dataset_meta_path: str) -> Path:
    """Return dataset root directory containing meta/info.json."""
    path = Path(dataset_meta_path).expanduser().resolve()
    if (path / 'meta' / 'info.json').exists():
        return path
    if (path / 'info.json').exists():
        return path.parent
    raise FileNotFoundError(
        f"Could not find meta/info.json under {path} "
        f"(expected dataset root or meta/ directory)."
    )


def load_dataset_metadata(dataset_meta_path: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = resolve_dataset_root(dataset_meta_path)
    info_path = root / 'meta' / 'info.json'
    with open(info_path) as f:
        info = json.load(f)
    repo_id = info.get('repo_id', 'local/dataset')
    return LeRobotDatasetMetadata(repo_id=repo_id, root=str(root))


def resolve_device(requested: str) -> str:
    device = requested.strip().lower()
    if device not in ('cuda', 'cpu'):
        raise ValueError(f"device must be 'cuda' or 'cpu', got {requested!r}")
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA requested but unavailable — falling back to CPU.')
        return 'cpu'
    return device


def policy_output_chunk_size(policy_cfg) -> int:
    """Policy one-shot action-chunk length (CloudGripper ``chunk_size``).

    - ACT-style: ``config.chunk_size``
    - Diffusion: ``config.horizon`` (full sampled trajectory; not ``n_action_steps``)
    """
    if hasattr(policy_cfg, 'chunk_size'):
        return int(policy_cfg.chunk_size)
    if hasattr(policy_cfg, 'horizon'):
        return int(policy_cfg.horizon)
    raise AttributeError(
        f'Policy config {type(policy_cfg).__name__} has neither chunk_size nor horizon'
    )


def validate_policy_params(
    yaml_chunk_size: int,
    yaml_n_action_steps: int,
    policy_cfg,
) -> None:
    """Validate YAML against policy config.

    ``chunk_size``: policy one-shot output length (must match).
    ``n_action_steps``: how many of those steps the controller executes (warn only).
    """
    policy_chunk = policy_output_chunk_size(policy_cfg)
    policy_n_steps = int(getattr(policy_cfg, 'n_action_steps', yaml_n_action_steps))

    if yaml_chunk_size != policy_chunk:
        raise RuntimeError(
            f'chunk_size mismatch: evaluation.yaml={yaml_chunk_size}, '
            f'policy config={policy_chunk}'
        )
    if yaml_n_action_steps != policy_n_steps:
        logger.warning(
            'n_action_steps mismatch: evaluation.yaml=%s, policy config=%s '
            '(using evaluation.yaml value for control loop).',
            yaml_n_action_steps,
            policy_n_steps,
        )


def obs_frame_to_values(frame: ObsFrame, ds_features: dict) -> dict:
    """Convert ObsFrame into the values dict expected by build_dataset_frame."""
    state = np.asarray(frame.state, dtype=np.float32)
    image = np.frombuffer(frame.image.data, dtype=np.uint8).reshape(
        frame.image.height, frame.image.width, 3,
    )

    state_names = ds_features['observation.state']['names']
    if len(state) != len(state_names):
        raise ValueError(
            f'observation.state length {len(state)} != '
            f'dataset meta {len(state_names)}'
        )

    img_shape = ds_features['observation.images.front']['shape']
    if (frame.image.height, frame.image.width) != tuple(img_shape[:2]):
        raise ValueError(
            f'image shape ({frame.image.height}, {frame.image.width}) != '
            f'dataset meta ({img_shape[0]}, {img_shape[1]})'
        )

    values = {name: float(state[i]) for i, name in enumerate(state_names)}
    values['front'] = image
    return values


class PolicyRunner:
    """Loads a LeRobot policy and runs passive chunk inference."""

    def __init__(
        self,
        policy_path: str,
        dataset_meta_path: str,
        device: str,
        yaml_chunk_size: int,
        yaml_n_action_steps: int,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self._build_dataset_frame = build_dataset_frame

        self._ds_meta = load_dataset_metadata(dataset_meta_path)
        self._ds_features = self._ds_meta.features

        resolved_device = resolve_device(device)
        policy_dir = Path(policy_path).expanduser().resolve()
        if not (policy_dir / 'config.json').exists():
            raise FileNotFoundError(f'Missing config.json under {policy_dir}')

        self._cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
        self._cfg.pretrained_path = str(policy_dir)
        self._cfg.device = resolved_device

        validate_policy_params(yaml_chunk_size, yaml_n_action_steps, self._cfg)

        self._chunk_size = yaml_chunk_size
        self._n_action_steps = yaml_n_action_steps
        self._action_dim = int(self._ds_features['action']['shape'][0])

        self._policy = make_policy(self._cfg, ds_meta=self._ds_meta)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._cfg,
            pretrained_path=str(policy_dir),
            dataset_stats=self._ds_meta.stats,
        )
        self._policy.eval()
        logger.info(
            'Policy loaded — type=%s device=%s chunk_size=%d '
            'n_action_steps=%d action_dim=%d',
            self._cfg.type,
            resolved_device,
            self._chunk_size,
            self._n_action_steps,
            self._action_dim,
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> None:
        if hasattr(self._policy, 'reset'):
            self._policy.reset()
        for proc in (self._preprocessor, self._postprocessor):
            if hasattr(proc, 'reset'):
                proc.reset()

    def _prepare_policy_batch(self, batch: dict) -> dict:
        """Populate temporal obs queues when required (e.g. DiffusionPolicy).

        Mirrors the queue-filling half of DiffusionPolicy.select_action before
        predict_action_chunk stacks from self._queues.
        """
        from lerobot.policies.utils import populate_queues
        from lerobot.utils.constants import ACTION, OBS_IMAGES

        infer_batch = dict(batch)
        if ACTION in infer_batch:
            infer_batch.pop(ACTION)

        policy = self._policy
        queues = getattr(policy, '_queues', None)
        if queues is None:
            return infer_batch

        cfg = policy.config
        if getattr(cfg, 'image_features', None):
            infer_batch[OBS_IMAGES] = torch.stack(
                [infer_batch[key] for key in cfg.image_features], dim=-4,
            )
        policy._queues = populate_queues(queues, infer_batch)
        return infer_batch

    def _predict_raw_chunk(self, batch: dict) -> torch.Tensor:
        """Run policy and return a (B, chunk_size, action_dim) tensor.

        LeRobot DiffusionPolicy.predict_action_chunk truncates to n_action_steps
        for its internal action queue. CloudGripper treats chunk_size as the full
        one-shot output (Diffusion: horizon), so we sample the full trajectory.
        """
        policy = self._policy
        cfg = policy.config

        is_diffusion = (
            getattr(cfg, 'type', None) == 'diffusion'
            or (hasattr(cfg, 'horizon') and not hasattr(cfg, 'chunk_size'))
        )
        if not is_diffusion:
            return policy.predict_action_chunk(batch)

        from lerobot.utils.constants import OBS_STATE

        queues = policy._queues
        stacked = {
            k: torch.stack(list(queues[k]), dim=1)
            for k in batch if k in queues
        }
        model = policy.diffusion
        batch_size = stacked[OBS_STATE].shape[0]
        global_cond = model._prepare_global_conditioning(stacked)
        # (B, horizon, action_dim) — full policy chunk, not n_action_steps slice
        actions = model.conditional_sample(batch_size, global_cond=global_cond)
        if actions.shape[1] != self._chunk_size:
            raise RuntimeError(
                f'Diffusion horizon output length {actions.shape[1]} != '
                f'chunk_size {self._chunk_size}'
            )
        return actions

    def predict_chunk_from_values(self, values: dict) -> np.ndarray:
        """Return denormalized action chunk from build_dataset_frame values.

        values must match obs_frame_to_values() output:
          state feature names -> float, plus camera key 'front' -> HxWx3 uint8.

        Returns shape (chunk_size, action_dim). The controller should execute
        only the first n_action_steps of this chunk.
        """
        from lerobot.policies.utils import prepare_observation_for_inference

        dataset_frame = self._build_dataset_frame(
            self._ds_features, values, prefix='observation',
        )
        # Official LeRobot inference path: HWC uint8 → CHW float32 [0, 1],
        # add batch dim, move to device — must happen before NormalizerProcessor
        # (which adapts stats dtype to the tensor dtype and overflows on uint8).
        device = torch.device(self._cfg.device)
        batch = prepare_observation_for_inference(dataset_frame, device)
        batch = self._preprocessor(batch)
        batch = self._prepare_policy_batch(batch)

        with torch.inference_mode():
            action_chunk = self._predict_raw_chunk(batch)

        if not isinstance(action_chunk, torch.Tensor):
            raise TypeError(f'Expected Tensor action chunk, got {type(action_chunk)}')

        denorm_steps: list[np.ndarray] = []
        for step_idx in range(action_chunk.shape[1]):
            step = action_chunk[:, step_idx, :]
            denorm = self._postprocessor(step)
            if isinstance(denorm, torch.Tensor):
                denorm_steps.append(denorm.squeeze(0).detach().cpu().numpy())
            else:
                from lerobot.utils.constants import ACTION
                denorm_steps.append(denorm[ACTION].squeeze(0).detach().cpu().numpy())

        chunk = np.stack(denorm_steps, axis=0).astype(np.float32)
        if chunk.shape != (self._chunk_size, self._action_dim):
            raise RuntimeError(
                f'Unexpected chunk shape {chunk.shape}, '
                f'expected ({self._chunk_size}, {self._action_dim})'
            )
        return chunk

    def predict_chunk(self, frame: ObsFrame) -> np.ndarray:
        """Return denormalized action chunk, shape (chunk_size, action_dim)."""
        values = obs_frame_to_values(frame, self._ds_features)
        return self.predict_chunk_from_values(values)
