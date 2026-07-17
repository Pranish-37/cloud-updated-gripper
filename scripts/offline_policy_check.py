#!/usr/bin/env python3
"""Offline single-frame policy check — same inference path as evaluation.

Loads one (episode, frame) from a LeRobot dataset, runs PolicyRunner
(build_dataset_frame → preprocessor → predict_action_chunk → postprocessor),
and prints the predicted action chunk vs the dataset ground-truth action.

Usage (inside container / after workspace_env):
  python3 scripts/offline_policy_check.py \
      --policy /workspace/policy/030000/pretrained_model \
      --dataset /workspace/data/rgb_stacking \
      --episode 0 --frame 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_eval_defaults() -> dict:
    cfg_path = _repo_root() / 'config' / 'evaluation.yaml'
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}
    return (raw.get('/**') or {}).get('ros__parameters') or {}


def _tensor_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _image_to_hwc_uint8(image) -> np.ndarray:
    """Convert LeRobot dataset image (CHW float[0,1] or HWC) to HWC uint8."""
    arr = _tensor_to_numpy(image)
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D image, got shape {arr.shape}')

    # CHW float → HWC
    if arr.shape[0] in (1, 3) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    if arr.shape[-1] != 3:
        raise ValueError(f'Expected RGB image, got shape {arr.shape}')
    return np.ascontiguousarray(arr)


def _dataset_item_to_values(item: dict, ds_features: dict) -> dict:
    """Mirror obs_frame_to_values() output from a LeRobotDataset __getitem__ item."""
    state = _tensor_to_numpy(item['observation.state']).astype(np.float32).reshape(-1)
    image = _image_to_hwc_uint8(item['observation.images.front'])

    state_names = ds_features['observation.state']['names']
    if len(state) != len(state_names):
        raise ValueError(
            f'observation.state length {len(state)} != dataset meta {len(state_names)}'
        )

    img_shape = ds_features['observation.images.front']['shape']
    if image.shape[:2] != tuple(img_shape[:2]):
        raise ValueError(
            f'image shape {image.shape[:2]} != dataset meta {tuple(img_shape[:2])}'
        )

    values = {name: float(state[i]) for i, name in enumerate(state_names)}
    values['front'] = image
    return values


def _fmt_vec(vec: np.ndarray, names: list[str] | None = None, precision: int = 4) -> str:
    if names is not None and len(names) == len(vec):
        parts = [f'{n}={float(v):.{precision}f}' for n, v in zip(names, vec)]
        return '[' + ', '.join(parts) + ']'
    return np.array2string(vec, precision=precision, suppress_small=True)


def parse_args() -> argparse.Namespace:
    defaults = _load_eval_defaults()
    p = argparse.ArgumentParser(
        description='Offline single-frame policy inference check (evaluation data path).',
    )
    p.add_argument(
        '--policy', type=str,
        default=defaults.get('policy_path', ''),
        help='Path to pretrained_model/ (config.json + weights)',
    )
    p.add_argument(
        '--dataset', type=str,
        default=defaults.get('dataset_meta_path', ''),
        help='Dataset root containing meta/info.json',
    )
    p.add_argument('--episode', type=int, required=True, help='Episode index (0-based)')
    p.add_argument('--frame', type=int, required=True, help='Frame index within episode (0-based)')
    p.add_argument(
        '--device', type=str, default=defaults.get('device', 'cuda'),
        choices=('cuda', 'cpu'),
    )
    p.add_argument(
        '--chunk-size', type=int, default=int(defaults.get('chunk_size', 16)),
    )
    p.add_argument(
        '--n-action-steps', type=int, default=int(defaults.get('n_action_steps', 8)),
    )
    p.add_argument(
        '--save-chunk', type=str, default='',
        help='Optional path to save predicted chunk as .npy',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.policy:
        print('[ERROR] --policy is required (or set policy_path in config/evaluation.yaml)', file=sys.stderr)
        return 1
    if not args.dataset:
        print('[ERROR] --dataset is required (or set dataset_meta_path in config/evaluation.yaml)', file=sys.stderr)
        return 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from inference_node.policy_runner import PolicyRunner, resolve_dataset_root

    root = resolve_dataset_root(args.dataset)
    info_path = root / 'meta' / 'info.json'
    with open(info_path) as f:
        info = json.load(f)
    repo_id = info.get('repo_id', 'local/dataset')

    total_episodes = int(info.get('total_episodes', 0))
    if args.episode < 0 or (total_episodes and args.episode >= total_episodes):
        print(
            f'[ERROR] episode {args.episode} out of range '
            f'[0, {total_episodes - 1}]',
            file=sys.stderr,
        )
        return 1

    print(f'Loading dataset  : {root}  (episode={args.episode})')
    # Use pyav — torchcodec often fails in this Docker image (FFmpeg .so mismatch).
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(root),
        episodes=[args.episode],
        video_backend='pyav',
    )
    ep_len = len(dataset)
    if args.frame < 0 or args.frame >= ep_len:
        print(
            f'[ERROR] frame {args.frame} out of range for episode {args.episode} '
            f'[0, {ep_len - 1}] (len={ep_len})',
            file=sys.stderr,
        )
        return 1

    item = dataset[args.frame]
    gt_action = _tensor_to_numpy(item['action']).astype(np.float32).reshape(-1)
    state = _tensor_to_numpy(item['observation.state']).astype(np.float32).reshape(-1)
    ds_features = dataset.meta.features
    action_names = ds_features['action'].get('names')
    state_names = ds_features['observation.state'].get('names')

    print(f'Loading policy   : {args.policy}')
    runner = PolicyRunner(
        policy_path=args.policy,
        dataset_meta_path=str(root),
        device=args.device,
        yaml_chunk_size=args.chunk_size,
        yaml_n_action_steps=args.n_action_steps,
    )
    runner.reset()

    values = _dataset_item_to_values(item, ds_features)
    chunk = runner.predict_chunk_from_values(values)

    print('')
    print('═' * 64)
    print(f'  episode={args.episode}  frame={args.frame}  '
          f'chunk_size={chunk.shape[0]}  action_dim={chunk.shape[1]}')
    print('═' * 64)
    print(f'  observation.state : {_fmt_vec(state, state_names)}')
    print(f'  GT action        : {_fmt_vec(gt_action, action_names)}')
    print(f'  pred[0] (step 0) : {_fmt_vec(chunk[0], action_names)}')
    delta = chunk[0] - gt_action
    print(f'  pred[0] - GT     : {_fmt_vec(delta, action_names)}')
    print(f'  ||pred[0]-GT||_2 : {float(np.linalg.norm(delta)):.4f}')
    print('')
    print('  Full predicted action chunk:')
    header = '  step | ' + '  '.join(f'{n:>10s}' for n in (action_names or [f'a{i}' for i in range(chunk.shape[1])]))
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for i, step in enumerate(chunk):
        cols = '  '.join(f'{float(v):10.4f}' for v in step)
        print(f'  {i:4d} | {cols}')
    print('═' * 64)

    if args.save_chunk:
        out = Path(args.save_chunk).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, chunk)
        print(f'Saved chunk → {out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
