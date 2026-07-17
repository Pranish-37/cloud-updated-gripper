"""
Real-robot end-to-end evaluation (XArm 3-DoF + gripper) with a LeRobot policy.

Offline sanity check: ``--offline-test`` (or ``offline_test.enabled=true``) loads one dataset
frame, runs ``predict_action_chunk`` when available (ACT, π₀, …) else ``predict_action``,
and prints per-timestep actions plus the ``send_action``-style 7-vector at t=0.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import types
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any
import time
import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EVAL_DIR = Path(__file__).resolve().parent
CONFIG_DIR = _EVAL_DIR / "conf"
DEFAULT_CONFIG_NAME = "real_e2e_3DoF"

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep

try:
    import pyrealsense2 as _rs
except ImportError:
    _rs = None

from hardware.xarm_lerobot import XArmLeRobot, XArmRobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation

from data_collection.episode_reset import reset_cube_and_arm, default_initial_grid
from evaluation.eval_log import (
    EvalLogger,
    build_frame_record_leobot_format,
    features_to_json_doc,
    observation_for_json_log,
)

def _register_now_resolver() -> None:
    if OmegaConf.has_resolver("now"):
        return

    def _now(pattern: str) -> str:
        return datetime.datetime.now().strftime(pattern)

    OmegaConf.register_new_resolver("now", _now)


def _apply_runtime_env(cfg: Any) -> None:
    """Set os.environ from cfg.runtime_env (optional)."""
    env = cfg.get("runtime_env")
    if env is None:
        return

    def _set_dir(cfg_key: str, env_key: str) -> None:
        if not env.get(cfg_key):
            return
        p = Path(str(env[cfg_key])).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        os.environ[env_key] = str(p)

    if env.get("hf_home"):
        hf_home = Path(str(env.hf_home)).expanduser()
        os.environ["HF_HOME"] = str(hf_home)
        hf_home.mkdir(parents=True, exist_ok=True)
    if env.get("hf_hub_cache"):
        hub = Path(str(env.hf_hub_cache)).expanduser()
        os.environ["HF_HUB_CACHE"] = str(hub)
        hub.mkdir(parents=True, exist_ok=True)
    _set_dir("hf_datasets_cache", "HF_DATASETS_CACHE")
    _set_dir("hf_assets_cache", "HF_ASSETS_CACHE")


def _resolve_under_repo(path_str: str | Path) -> Path:
    """Resolve path; if relative, interpret from repository root (not cwd)."""
    p = Path(str(path_str)).expanduser()
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _print_torch_gpu_inventory(requested_device: str) -> None:
    print("--- PyTorch GPU check ---", flush=True)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"torch.cuda.device_count(): {n}", flush=True)
        for i in range(n):
            pr = torch.cuda.get_device_properties(i)
            gib = pr.total_memory / (1024**3)
            print(
                f"  cuda:{i}  name={pr.name}  total VRAM={gib:.2f} GiB  "
                f"compute_capability={pr.major}.{pr.minor}",
                flush=True,
            )
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(
                f"  cuda:0 free (driver report): {free_b / (1024**3):.2f} GiB "
                f"of {total_b / (1024**3):.2f} GiB",
                flush=True,
            )
        except Exception:
            pass
    else:
        print(
            "No CUDA GPU visible to PyTorch (CPU-only torch build, missing driver, or no NVIDIA GPU).",
            flush=True,
        )
    print(f"Requested device string: {requested_device!r}", flush=True)
    print("---", flush=True)


def _first_step_of_type(pipe: Any, cls: type) -> Any | None:
    for step in pipe.steps:
        if isinstance(step, cls):
            return step
    return None


def _print_norm_unnorm_summary(pre: Any, post: Any) -> None:
    """report the normalization and unnormalization summary from the checkpoint processors."""
    norm = _first_step_of_type(pre, NormalizerProcessorStep)
    unnorm = _first_step_of_type(post, UnnormalizerProcessorStep)
    print("\n--- Normalize / unnormalize (from checkpoint processors) ---", flush=True)
    if norm is None:
        print("  Pre: no NormalizerProcessorStep found.", flush=True)
    else:
        _one_line_norm_block(norm, forward=True)
    if unnorm is None:
        print("  Post: no UnnormalizerProcessorStep found.", flush=True)
    else:
        _one_line_norm_block(unnorm, forward=False)
    print("---", flush=True)


def _one_line_norm_block(step: Any, *, forward: bool) -> None:
    label = "Pre (normalize to network)" if forward else "Post (unnormalize from network)"
    nm = getattr(step, "norm_map", {}) or {}
    modes = ", ".join(f"{ft!s}→{mode!s}" for ft, mode in nm.items())
    feats = getattr(step, "features", {}) or {}
    keys = ", ".join(
        f"{name} {tuple(getattr(pf, 'shape', ()))}" for name, pf in feats.items()
    )
    eps = getattr(step, "eps", None)
    # MEAN_STD is the common ACT path; other modes use the same codepath in LeRobot with different stats.
    if forward:
        formula = "z = (x - mean) / (std + eps)"
    else:
        formula = "x = z * std + mean   (inverse of MEAN_STD)"
    print(f"  {label}", flush=True)
    print(f"    {formula}", flush=True)
    if nm and not all("MEAN_STD" in str(mode) for mode in nm.values()):
        print(
            "    (Other modes in norm_map use different formulas; see LeRobot normalize_processor.)",
            flush=True,
        )
    print(f"    By feature type: {modes}", flush=True)
    print(f"    Tensor keys: {keys}", flush=True)
    if eps is not None:
        print(f"    eps: {eps}", flush=True)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_hwc_uint8_image(arr: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB for ``predict_action`` ``prepare_observation_for_inference`` (/255, then CHW).
    Accepts CHW float in ~[0,1] or uint8, or HWC uint8/float.
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape={a.shape}")
    # CHW: channel dim is first and last dim is spatial (e.g. 3 x H x W)
    if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 2, 3, 4):
        x = a.astype(np.float32)
        if x.max() <= 1.5:
            x = np.clip(x * 255.0, 0.0, 255.0)
        hwc = np.transpose(x, (1, 2, 0))
        return np.round(hwc).astype(np.uint8)
    # HWC
    if a.shape[-1] == 3:
        x = a.astype(np.float32)
        if x.max() <= 1.5:
            x = np.clip(x * 255.0, 0.0, 255.0)
        return np.clip(np.round(x), 0, 255).astype(np.uint8)
    raise ValueError(f"Cannot interpret image shape {a.shape} as RGB")


def _raw_obs_to_dataset_values(raw: dict[str, Any], ds_features: dict[str, dict]) -> dict[str, Any]:
    """Expand observation.state into per-name scalars for build_dataset_frame (same as real_e2e_eval_6DoF)."""
    values: dict[str, Any] = {}
    st_key = "observation.state"
    if st_key in ds_features and st_key in raw:
        names = ds_features[st_key]["names"]
        arr = np.asarray(raw[st_key], dtype=np.float32).reshape(-1)
        if len(arr) != len(names):
            raise ValueError(
                f"observation.state length {len(arr)} != feature names count {len(names)}"
            )
        for i, n in enumerate(names):
            values[n] = float(arr[i]) # put the value of the observation.state into the values dictionary.
    for k, v in raw.items():
        if k.startswith("observation.images."):
            cam = k.removeprefix("observation.images.")
            values[cam] = v
    return values


def _merged_action_names(ds_features: dict[str, dict]) -> list[str]:
    """get the action names from the dataset features."""
    if ACTION in ds_features:
        return list(ds_features[ACTION]["names"])
    subkeys = sorted(k for k in ds_features if k.startswith("action.") and k != ACTION)
    out: list[str] = []
    for k in subkeys:
        out.extend(ds_features[k]["names"])
    return out


def _apply_camera_sensor_options(robot: Any, sensor_opts: Any) -> None:
    """Apply fixed WB / exposure settings to the RealSense color sensor after connect().

    sensor_opts is the OmegaConf node under robot.front_camera.sensor_options.
    Does nothing if pyrealsense2 is not importable or sensor_opts is None/empty.
    """
    if _rs is None or sensor_opts is None:
        return
    try:
        cam = robot.cameras.get("front")
        if cam is None or not hasattr(cam, "rs_profile") or cam.rs_profile is None:
            print("[camera] Warning: could not find connected RealSense to apply sensor options.", flush=True)
            return
        device = cam.rs_profile.get_device()
        color_sensor = None
        for s in device.query_sensors():
            if _rs.stream.color in [p.stream_type() for p in s.get_stream_profiles()]:
                color_sensor = s
                break
        if color_sensor is None:
            print("[camera] Warning: could not find color sensor.", flush=True)
            return

        import time as _time

        opts = dict(sensor_opts) if hasattr(sensor_opts, "items") else {}
        applied: dict[str, Any] = {}

        want_awb = bool(opts.get("auto_white_balance", True))
        want_ae  = bool(opts.get("auto_exposure",       True))

        if want_awb and not want_ae:
            # Special case: AWB=ON + AE=OFF.
            # AWB needs AE to be ON first so it can see a properly-exposed image
            # and converge to the correct white point.  Sequence:
            #   1. Start with both AWB=ON and AE=ON and let them converge together.
            #   2. Snapshot the stabilised exposure/gain.
            #   3. Disable AE and lock the snapshotted values.
            print("[camera] AWB+AE both ON — converging ...", flush=True)
            color_sensor.set_option(_rs.option.enable_auto_white_balance, 1.0)
            color_sensor.set_option(_rs.option.enable_auto_exposure,       1.0)
            _time.sleep(5.0) # wait for the AE and AWB to converge.

            # Snapshot current AE output (override with YAML values if explicitly set).
            snap_exp  = float(opts["exposure"]) if "exposure" in opts \
                        else color_sensor.get_option(_rs.option.exposure)
            snap_gain = float(opts["gain"])     if "gain"     in opts \
                        else color_sensor.get_option(_rs.option.gain)

            # Lock AE; double-write to prevent the camera from resetting gain.
            color_sensor.set_option(_rs.option.enable_auto_exposure, 0.0)
            color_sensor.set_option(_rs.option.exposure, snap_exp)
            color_sensor.set_option(_rs.option.gain,     snap_gain)
            
            color_sensor.set_option(_rs.option.exposure, snap_exp)
            color_sensor.set_option(_rs.option.gain,     snap_gain)
            _time.sleep(5.0)

            applied["auto_white_balance"] = True
            applied["auto_exposure"]      = False
            applied["exposure"]           = snap_exp
            applied["gain"]               = snap_gain

        else:
            print("[camera] AWB+AE both ON — converging ...", flush=True)
            color_sensor.set_option(_rs.option.enable_auto_white_balance, 1.0)
            color_sensor.set_option(_rs.option.enable_auto_exposure,       1.0)
            _time.sleep(5.0) # wait for the AE and AWB to converge.
            # Standard case: apply AE setting first, then WB.
            if "auto_exposure" in opts:
                color_sensor.set_option(_rs.option.enable_auto_exposure, int(want_ae))
                applied["auto_exposure"] = want_ae
            if not want_ae:
                if "exposure" in opts:
                    color_sensor.set_option(_rs.option.exposure, float(opts["exposure"]))
                if "gain" in opts:
                    color_sensor.set_option(_rs.option.gain, float(opts["gain"]))
                _time.sleep(0.05)
                if "exposure" in opts: # double-write to prevent the camera from resetting gain.
                    color_sensor.set_option(_rs.option.exposure, float(opts["exposure"]))
                    applied["exposure"] = float(opts["exposure"])
                if "gain" in opts: # double-write to prevent the camera from resetting gain.
                    color_sensor.set_option(_rs.option.gain, float(opts["gain"]))
                    applied["gain"] = float(opts["gain"])

            if "auto_white_balance" in opts:
                color_sensor.set_option(_rs.option.enable_auto_white_balance, int(want_awb))
                applied["auto_white_balance"] = want_awb
            if not want_awb and "white_balance" in opts:
                color_sensor.set_option(_rs.option.white_balance, float(opts["white_balance"]))
                applied["white_balance"] = float(opts["white_balance"])

            if want_awb:
                print("[camera] AWB settling (3s) ...", flush=True)
                _time.sleep(3.0)

        print(f"[camera] Sensor options applied: {applied}", flush=True)
        flush_n = 20
        print(f"[camera] Flushing {flush_n} buffered frames ...", flush=True)
        for _ in range(flush_n):
            cam.read()
        print("[camera] Camera ready.", flush=True)
    except Exception as exc:
        print(f"[camera] Warning: failed to apply sensor options: {exc}", flush=True)


def _policy_to_xarm_send_action_3dof(
    robot_action: dict[str, float], pad_rpy: list[float] | tuple[float, ...]
) -> dict[str, np.ndarray]:
    """7-float vector for XArmLeRobot.send_action: x,y,z,roll,pitch,yaw,gripper."""
    if len(pad_rpy) != 3:
        raise ValueError("pad_rpy must have 3 values (roll, pitch, yaw radians)")
    for k in ("ee_x", "ee_y", "ee_z", "gripper"):
        if k not in robot_action:
            raise KeyError(f"Missing action key {k!r} in policy output {sorted(robot_action)}")
    r, p, y = (float(pad_rpy[0]), float(pad_rpy[1]), float(pad_rpy[2])) # pad the roll, pitch, yaw to the action vector for 3DoF.
    vec = np.array(
        [
            robot_action["ee_x"],
            robot_action["ee_y"],
            robot_action["ee_z"],
            r,
            p,
            y,
            robot_action["gripper"],
        ],
        dtype=np.float32,
    )
    return {"action": vec}


def _predict_action_chunk_batched(
    policy: Any,
    policy_cfg: PreTrainedConfig,
    policy_type: str,
    obs_proc: dict[str, Any],
) -> torch.Tensor:
    """
    For DiffusionPolicy, ``predict_action_chunk`` only stacks from internal deques; observations must
    be pushed with ``populate_queues`` like ``select_action`` does, or queues stay empty
    (``stack expects a non-empty TensorList``). ACT/π₀ can forward the preprocessed batch directly.
    """
    if str(policy_type).lower() != "diffusion":
        return policy.predict_action_chunk(obs_proc)

    from lerobot.utils.constants import ACTION as AKEY, OBS_IMAGES
    from lerobot.policies.utils import populate_queues

    batch = dict(obs_proc)
    if AKEY in batch:
        batch.pop(AKEY)
    feats = getattr(policy_cfg, "image_features", None)
    if feats:
        klist = list(feats.keys()) if isinstance(feats, dict) else list(feats)
        if klist:
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in klist], dim=-4)
    policy._queues = populate_queues(policy._queues, batch)
    return policy.predict_action_chunk(batch)


def _offline_test_config(cfg: Any) -> dict[str, Any]:
    """Read ``offline_test.*`` from the config."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "frame_index": 0,
        "print_action_chunk": True,
        "print_chunk_max_steps": 64,
    }
    node = cfg.get("offline_test")
    if node is None:
        d: dict[str, Any] = {}
    else:
        d = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
    return {
        "enabled": bool(d.get("enabled", defaults["enabled"])),
        "frame_index": int(d.get("frame_index", defaults["frame_index"])),
        "print_action_chunk": bool(d.get("print_action_chunk", defaults["print_action_chunk"])),
        "print_chunk_max_steps": int(d.get("print_chunk_max_steps", defaults["print_chunk_max_steps"])),
    }


def _run_offline_one_frame(
    *,
    cfg: Any,
    dataset_root: Path,
    ds_meta: LeRobotDatasetMetadata,
    policy: Any,
    policy_cfg: PreTrainedConfig,
    policy_type: str,
    preprocessor: Any,
    postprocessor: Any,
    frame_index: int,
    pad_rpy: list[float],
    print_action_chunk: bool,
    print_chunk_max_steps: int,
) -> None:
    print("\n--- Offline one-frame inference ---", flush=True)
    repo_id = str(cfg.dataset.repo_id)
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    n = len(dataset)
    if n == 0:
        print("Error: dataset is empty.", file=sys.stderr)
        raise SystemExit(3)
    idx = frame_index
    frame = dataset[idx]
    ds_features = ds_meta.features
    # Only print ee_x, ee_y, ee_z, gripper from observation.state (names live under observation.state).
    obs_state = frame.get("observation.state")
    if obs_state is not None:
        obs_arr = _to_numpy(obs_state).astype(np.float32).reshape(-1)
        st_key = "observation.state"
        names = (
            list(ds_features[st_key]["names"])
            if st_key in ds_features and ds_features[st_key].get("names")
            else []
        )

        def _idx(nm: str, default: int) -> int:
            return names.index(nm) if nm in names else default

        # 3DoF xArm dataset: 11-dim state → ee_* / gripper at indices 7..10.
        ee_x_idx = _idx("ee_x", 7)
        ee_y_idx = _idx("ee_y", 8)
        ee_z_idx = _idx("ee_z", 9)
        gripper_idx = _idx("gripper", 10)
        print(f"Dataset frame at index {idx}:")
        print(f"  ee_x: {obs_arr[ee_x_idx]}")
        print(f"  ee_y: {obs_arr[ee_y_idx]}")
        print(f"  ee_z: {obs_arr[ee_z_idx]}")
        print(f"  gripper: {obs_arr[gripper_idx]}")
    else:
        print(f"Dataset frame at index {idx}: Could not find observation.state", flush=True)

    raw_obs: dict[str, Any] = {}
    if "observation.state" in frame:
        raw_obs["observation.state"] = _to_numpy(frame["observation.state"]).astype(np.float32).reshape(-1)
    for key in frame:
        if key.startswith("observation.images."):
            raw_obs[key] = _to_hwc_uint8_image(_to_numpy(frame[key]))

    values = _raw_obs_to_dataset_values(raw_obs, ds_features)
    observation_frame = build_dataset_frame(ds_features, values, prefix=OBS_STR)

    device = get_safe_torch_device(str(cfg.get("device", policy_cfg.device)))
    task = str(cfg.get("task", ""))
    robot_type = str(cfg.get("robot_type", "xarm"))

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    action_names = _merged_action_names(ds_features)
    use_amp = bool(policy_cfg.use_amp)
    amp_ctx = ( # automatic mixed precision context for the policy forward pass.
        torch.autocast(device_type=device.type, enabled=device.type == "cuda" and use_amp)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )

    obs_for_prep = deepcopy(observation_frame)
    # extend the dataset frame to the batch size of the policy to shape (batch_size, *obs_frame.shape).
    batch_prepared = prepare_observation_for_inference(obs_for_prep, device, task, robot_type)

    with torch.inference_mode(), amp_ctx:
        obs_proc = preprocessor(batch_prepared) # obs_proc is the observation processed by the preprocessor.

    # ACT, π₀, π₀-fast, etc. implement ``predict_action_chunk``; do not hardcode type == "act".
    can_chunk = print_action_chunk and hasattr(policy, "predict_action_chunk")
    chunk: torch.Tensor | None = None
    if can_chunk:
        with torch.inference_mode(), amp_ctx:
            chunk = _predict_action_chunk_batched(
                policy, policy_cfg, policy_type, obs_proc
            )
        if chunk is not None and chunk.ndim == 2:
            # extend the chunk to the batch size of the policy to shape (batch_size, chunk.shape[1], chunk.shape[2]).
            chunk = chunk.unsqueeze(0)

    if chunk is not None:
        _b, t_horizon, _dim_a = chunk.shape
        # chunk_size / n_action_steps come only from the loaded policy checkpoint (not from this script).
        cs = getattr(policy_cfg, "chunk_size", None) # get chunk size from the policy config.
        nas = getattr(policy_cfg, "n_action_steps", None) # get number of action steps from the policy config.

        print(
            f"policy_chunk (raw forward output) shape={tuple(chunk.shape)} chunk_size={cs!r} n_action_steps={nas!r}",
            flush=True,
        )
        if cs is not None and int(cs) != int(t_horizon):
            print(
                f"WARNING: horizon {t_horizon} != chunk_size {cs}",
                flush=True,
            )
        n_print = max(1, min(t_horizon, print_chunk_max_steps))
        robot_action: dict[str, float] | None = None
        with torch.inference_mode(), amp_ctx:
            for t in range(n_print):
                post_out = postprocessor(chunk[:, t, :])
                ra = make_robot_action(post_out, {ACTION: {"names": action_names}})
                if t == 0:
                    robot_action = ra
                print(f"  t={t} {ra}", flush=True)
        assert robot_action is not None
    else:
        if print_action_chunk and not hasattr(policy, "predict_action_chunk"):
            print(
                f"Skipping chunk print (policy_type={policy_type!r}; no predict_action_chunk).",
                flush=True,
            )
        action_tensor = predict_action(
            observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        robot_action = make_robot_action(action_tensor, {ACTION: {"names": action_names}})

    xarm_cmd = _policy_to_xarm_send_action_3dof(robot_action, pad_rpy)

    print(f"t=0_policy {robot_action}", flush=True)
    print(f"send_action {xarm_cmd['action'].tolist()}", flush=True)
    print("--- End offline one-frame ---\n", flush=True)


def main() -> None:
    # read the config from yaml file
    parser = argparse.ArgumentParser(description="YAML for policy evaluation")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help=f"Stem or filename under config dir (default: {DEFAULT_CONFIG_NAME})",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing YAML files (default: <this_dir>/conf)",
    )
    parser.add_argument(
        "--offline-test",
        action="store_true",
        help="Run one-frame offline inference from the dataset and print XArm send_action; then exit.",
    )
    args, rest = parser.parse_known_args()
    # get the config path from the command line arguments.
    config_name = args.config_name
    config_dir = args.config_dir or CONFIG_DIR
    if not config_dir.is_dir():
        print(f"Error: config directory not found: {config_dir}", file=sys.stderr)
        raise SystemExit(1)
    if config_name.endswith((".yaml", ".yml")):
        config_name = config_name[: -len(Path(config_name).suffix)]
    yaml_path = config_dir / f"{config_name}.yaml"
    if not yaml_path.is_file():
        print(f"Error: config not found: {yaml_path}", file=sys.stderr)
        raise SystemExit(1)
    # set time
    _register_now_resolver()
    # load the config from the yaml file.
    cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(cfg)
    if rest:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(rest))
        OmegaConf.resolve(cfg)

    if args.offline_test:
        if cfg.get("offline_test") is None:
            cfg.offline_test = OmegaConf.create(
                {
                    "enabled": True,
                    "frame_index": 0,
                    "print_action_chunk": True,
                    "print_chunk_max_steps": 64,
                }
            )
        else:
            cfg.offline_test.enabled = True

    _apply_runtime_env(cfg)

    # get the policy path from the config.
    policy_path = cfg.get("policy_path")
    if not policy_path or not str(policy_path).strip():
        print("Error: set `policy_path` in the config or via override.", file=sys.stderr)
        raise SystemExit(2)
    policy_path = _resolve_under_repo(str(policy_path).strip())
    # check if the policy path is a valid LeRobot format.
    if not (policy_path / "config.json").is_file():
        print(f"Error: no config.json under {policy_path}", file=sys.stderr)
        raise SystemExit(2)

    dataset_root = _resolve_under_repo(cfg.dataset.root)
    if not (dataset_root / "meta" / "info.json").is_file():
        print(f"Error: LeRobot dataset metadata not found at {dataset_root}", file=sys.stderr)
        raise SystemExit(2)

    register_third_party_plugins()

    ds_meta = LeRobotDatasetMetadata(repo_id=str(cfg.dataset.repo_id), root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = str(policy_path)
    if cfg.get("n_action_steps") is not None:
        # override the n_action_steps in the evaluation config into the policy config.
        policy_cfg.n_action_steps = int(cfg.n_action_steps)
        print(f"Eval config override: n_action_steps {policy_cfg.n_action_steps!r}.", flush=True)
    n_action_steps_cfg = getattr(policy_cfg, "n_action_steps", None)
    if n_action_steps_cfg is None:
        print("Error: n_action_steps is not set in the policy config.", file=sys.stderr)
        raise SystemExit(2)
    chunk_size_cfg = getattr(policy_cfg, "chunk_size", None)
    if cfg.get("device") is not None:
        policy_cfg.device = str(cfg.device)
    else:
        print("Warning: no `device` in config; using policy default.", flush=True)

    policy_cfg.use_amp = bool(cfg.get("policy_use_amp", False))

    policy_type = str(getattr(policy_cfg, "type", "") or "")
    pd = cfg.get("policy_dtype")
    if pd is not None and str(pd).strip():
        if policy_type in ("pi0", "pi05", "pi0_fast") and hasattr(policy_cfg, "dtype"):
            policy_cfg.dtype = str(pd).strip()
            print(f"Overriding policy dtype for inference: {policy_cfg.dtype}", flush=True)
        else:
            print(
                f"Note: policy_dtype ignored for policy type {policy_type!r}.",
                flush=True,
            )

    _print_torch_gpu_inventory(str(cfg.get("device", policy_cfg.device)))

    print(f"Loading policy {policy_type!r} on {policy_cfg.device}.", flush=True)

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    print(f"make_policy OK: {type(policy).__name__}", flush=True)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(policy_path),
        dataset_stats=ds_meta.stats,
    )
    _print_norm_unnorm_summary(preprocessor, postprocessor)

    pad_rpy = [float(x) for x in cfg.pad_rpy]

    # run the offline test.
    ot = _offline_test_config(cfg)
    if ot["enabled"]:
        _run_offline_one_frame(
            cfg=cfg,
            dataset_root=dataset_root,
            ds_meta=ds_meta,
            policy=policy,
            policy_cfg=policy_cfg,
            policy_type=policy_type,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            frame_index=ot["frame_index"],
            pad_rpy=pad_rpy,
            print_action_chunk=ot["print_action_chunk"],
            print_chunk_max_steps=ot["print_chunk_max_steps"],
        )
    else:
        print("Offline test not enabled. Running real world evaluation.", flush=True)

        # Build initial-position table from eval_grid config (random OOD or deterministic grid).
        _eg = cfg.get("eval_grid") or {}
        _eg_mode = str(_eg.get("mode", "random")).lower()
        _n_total = int(cfg.get("n_episodes", 20))
        if _eg_mode == "random":
            _seed = int(_eg.get("seed", 42))
            _x_lo, _x_hi = float(_eg.get("x_range", [355, 505])[0]), float(_eg.get("x_range", [355, 505])[1])
            _y_lo, _y_hi = float(_eg.get("y_range", [-275, -125])[0]), float(_eg.get("y_range", [-275, -125])[1])
            _z = float(_eg.get("z", 9.0))
            _rng = np.random.default_rng(_seed)
            _xs = _rng.uniform(_x_lo, _x_hi, _n_total)
            _ys = _rng.uniform(_y_lo, _y_hi, _n_total)
            initial_positions = {i: {"x": float(_xs[i]), "y": float(_ys[i]), "z": _z} for i in range(_n_total)}
            print(
                f"Eval grid (random seed={_seed}): x∈[{_x_lo},{_x_hi}] y∈[{_y_lo},{_y_hi}] z={_z} "
                f"n={_n_total}",
                flush=True,
            )
        else:
            initial_positions = default_initial_grid()
            print("Eval grid (default deterministic grid).", flush=True)

        if not hasattr(policy, "predict_action_chunk"):
            print(
                "Error: real-world loop needs ``predict_action_chunk`` (ACT, pi0, …). "
                f"Policy type {policy_type!r} has no such method.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        start_episode = cfg.get("start_episode", 0)
        n_episodes = cfg.get("n_episodes", 20)
        max_steps_per_episode = cfg.get("max_steps_per_episode", 100)
        device = get_safe_torch_device(str(cfg.get("device", policy_cfg.device)))
        task = str(cfg.get("task", ""))
        robot_type = str(cfg.get("robot_type", "xarm"))
        ds_features = ds_meta.features
        action_names = _merged_action_names(ds_features)
        n_action_steps = int(n_action_steps_cfg)
        use_amp = bool(policy_cfg.use_amp)
        amp_ctx = (
            torch.autocast(device_type=device.type, enabled=device.type == "cuda" and use_amp)
            if device.type == "cuda" and use_amp
            else nullcontext()
        )
        fps = float(cfg.get("fps", 0) or 0)
        dt = 1.0 / fps if fps > 0 else 0.0

        # Nested keys: cfg.get("robot.front_camera.x") does not work on OmegaConf; use select().
        cam_serial = OmegaConf.select(cfg, "robot.front_camera.serial_number_or_name")
        if cam_serial is None or str(cam_serial).strip() == "":
            print(
                "Error: set ``robot.front_camera.serial_number_or_name`` in the YAML "
                "(required for RealSense).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        cam_fps = OmegaConf.select(cfg, "robot.front_camera.fps")
        if cam_fps is None:
            cam_fps = OmegaConf.select(cfg, "fps", default=5)
        cam_config = RealSenseCameraConfig(
            serial_number_or_name=str(cam_serial),
            fps=int(cam_fps),
            width=int(OmegaConf.select(cfg, "robot.front_camera.width", default=640)),
            height=int(OmegaConf.select(cfg, "robot.front_camera.height", default=480)),
            color_mode=ColorMode.RGB,
            use_depth=False,
            rotation=Cv2Rotation.NO_ROTATION,
        )

        robot_ip = OmegaConf.select(cfg, "robot.ip", default="192.168.0.244")
        robot_config = XArmRobotConfig(
            ip=str(robot_ip),
            cameras={"front": cam_config},
            use_front_camera=bool(OmegaConf.select(cfg, "robot.use_front_camera", default=True)),
        )
        robot = XArmLeRobot(robot_config)
        print("Connecting to robot...", flush=True)
        robot.connect()

        _sensor_opts = OmegaConf.select(cfg, "robot.front_camera.sensor_options")
        _apply_camera_sensor_options(robot, _sensor_opts)

        print(
            f"Chunk control ({policy_type}): re-infer every {n_action_steps} executed steps; "
            f"chunk_size={chunk_size_cfg!r} (full horizon from model).",
            flush=True,
        )

        _log_cfg = cfg.get("logging") or {}
        print_action_chunk_debug = bool(_log_cfg.get("print_action_chunk_debug", True))
        save_frames = bool(_log_cfg.get("save_frames", False))
        save_frames_every_n = max(1, int(_log_cfg.get("save_frames_every_n", 1)))
        log_actions_jsonl = bool(_log_cfg.get("log_actions_jsonl", False))
        episode_dataset_json = bool(_log_cfg.get("episode_dataset_json", True))

        out_dir_cfg = OmegaConf.select(cfg, "output_dir")
        eval_logger: EvalLogger | None = None
        features_doc = features_to_json_doc(ds_meta.features)
        task_index_eval = int(cfg.get("task_index", 0))
        if save_frames or log_actions_jsonl or episode_dataset_json:
            root = _resolve_under_repo(
                str(out_dir_cfg or "outputs/real_e2e_eval_3DoF")
            )
            eval_logger = EvalLogger(root)
            eval_logger.log({
                "event": "run_start",
                "task": task,
                "policy_path": str(policy_path),
                "save_frames": save_frames,
                "log_actions_jsonl": log_actions_jsonl,
                "episode_dataset_json": episode_dataset_json,
                "initial_positions": {str(k): v for k, v in initial_positions.items()},
            })

        for episode in range(start_episode, n_episodes):
            try:
                user_input = input(
                    f"Episode {episode} of {n_episodes}. Press Enter to continue or 'q' to quit."
                )
            except KeyboardInterrupt:
                print("\nCtrl+C: skipping this episode prompt → next episode.", flush=True)
                continue
            if user_input == "q":
                break
            # Log this episode's initial cube position before reset.
            _ep_pos = initial_positions[episode % len(initial_positions)]
            print(
                f"Episode {episode}: cube initial position "
                f"x={_ep_pos['x']:.1f} y={_ep_pos['y']:.1f} z={_ep_pos['z']:.1f}",
                flush=True,
            )
            if eval_logger is not None:
                eval_logger.log({
                    "event": "episode_start",
                    "episode": episode,
                    "cube_init_pos": _ep_pos,
                })
            # reset the object and robot arm
            reset_cube_and_arm(robot, initial_positions, episode)
            time.sleep(1)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            if eval_logger is not None and log_actions_jsonl:
                ap = eval_logger.episode_dir(episode) / "actions.jsonl"
                if ap.is_file():
                    ap.unlink()
            if eval_logger is not None and episode_dataset_json:
                jp = eval_logger.episode_dir(episode) / "episode_dataset.json"
                if jp.is_file():
                    jp.unlink()
                eval_logger.begin_episode(episode, control_dt=dt)

            chunk: torch.Tensor | None = None
            chunk_cursor = 0
            steps_per_chunk = n_action_steps

            try:
                for step in range(max_steps_per_episode):
                    try:
                        _t_step_start = time.perf_counter()
                        _t_infer_ms: float = 0.0

                        if chunk is None or chunk_cursor >= steps_per_chunk:
                            _t0 = time.perf_counter()
                            raw_obs = robot.get_observation()
                            _t_obs_infer_ms = (time.perf_counter() - _t0) * 1e3

                            values = _raw_obs_to_dataset_values(
                                raw_obs, ds_features
                            )
                            observation_frame = build_dataset_frame(
                                ds_features, values, prefix=OBS_STR
                            )
                            obs_for_prep = deepcopy(observation_frame)
                            batch_prepared = prepare_observation_for_inference(
                                obs_for_prep, device, task, robot_type
                            )
                            _t0 = time.perf_counter()
                            with torch.inference_mode(), amp_ctx:
                                obs_proc = preprocessor(batch_prepared)
                                chunk = _predict_action_chunk_batched(
                                    policy, policy_cfg, policy_type, obs_proc
                                )
                            _t_infer_ms = (time.perf_counter() - _t0) * 1e3

                            if chunk is None:
                                print(
                                    "Error: predict_action_chunk returned None.",
                                    file=sys.stderr,
                                )
                                raise SystemExit(2)
                            if chunk.ndim == 2:
                                chunk = chunk.unsqueeze(0)
                            _b, t_horizon, _a_dim = chunk.shape
                            if chunk_size_cfg is not None and int(
                                chunk_size_cfg
                            ) != int(t_horizon):
                                print(
                                    f"WARNING: chunk time dim {t_horizon} != policy chunk_size {chunk_size_cfg}",
                                    flush=True,
                                )
                            steps_per_chunk = min(n_action_steps, t_horizon)
                            if n_action_steps > t_horizon:
                                print(
                                    f"WARNING: n_action_steps ({n_action_steps}) > chunk horizon ({t_horizon}); "
                                    f"executing {steps_per_chunk} steps before re-inference.",
                                    flush=True,
                                )
                            chunk_cursor = 0

                            print(
                                f"  [timing] re-infer: obs={_t_obs_infer_ms:.1f}ms"
                                f"  infer={_t_infer_ms:.1f}ms",
                                flush=True,
                            )

                            if print_action_chunk_debug:
                                print(
                                    f"\n--- chunk debug (episode={episode} env_step={step}) "
                                    f"T={t_horizon} postprocessed real-scale ---",
                                    flush=True,
                                )
                                with torch.inference_mode(), amp_ctx:
                                    for t_dbg in range(t_horizon):
                                        po = postprocessor(
                                            chunk[:, t_dbg, :]
                                        )
                                        ra_dbg = make_robot_action(
                                            po,
                                            {ACTION: {"names": action_names}},
                                        )
                                        _ = _policy_to_xarm_send_action_3dof(
                                            ra_dbg, pad_rpy
                                        )
                                        print(
                                            f"  t={t_dbg}  {ra_dbg}  ",
                                            flush=True,
                                        )
                                print("--- end chunk debug ---\n", flush=True)
                                # user_input = input("Press any key to continue")

                        t_exec = chunk_cursor
                        if t_exec >= chunk.shape[1]:
                            chunk = None
                            continue

                        _t0 = time.perf_counter()
                        with torch.inference_mode(), amp_ctx:
                            post_out = postprocessor(chunk[:, t_exec, :])
                        robot_action = make_robot_action(
                            post_out, {ACTION: {"names": action_names}}
                        )
                        _t_post_ms = (time.perf_counter() - _t0) * 1e3

                        _t0 = time.perf_counter()
                        raw_obs_now = robot.get_observation()
                        _t_obs_log_ms = (time.perf_counter() - _t0) * 1e3

                        img_paths: dict[str, str] = {}
                        _t0 = time.perf_counter()
                        if eval_logger is not None and (
                            (
                                save_frames
                                and step % save_frames_every_n == 0
                            )
                            or episode_dataset_json
                        ):
                            img_paths = eval_logger.save_observation_images(
                                episode, step, raw_obs_now
                            )
                        _t_save_img_ms = (time.perf_counter() - _t0) * 1e3

                        # user_input = input(
                        #     f"Action index: {step}, Action to execute: {robot_action}, press any key to continue"
                        # )
                        xarm_action = _policy_to_xarm_send_action_3dof(
                            robot_action, pad_rpy
                        )
                        _t0 = time.perf_counter()
                        robot.send_action(xarm_action, wait=True)
                        _t_send_ms = (time.perf_counter() - _t0) * 1e3

                        _t_step_total_ms = (time.perf_counter() - _t_step_start) * 1e3
                        print(
                            f"Action index: {step} (chunk_t={t_exec})"
                            f"  action={robot_action}"
                            f"  | post={_t_post_ms:.1f}ms"
                            f"  obs_log={_t_obs_log_ms:.1f}ms"
                            f"  save_img={_t_save_img_ms:.1f}ms"
                            f"  send_action={_t_send_ms:.1f}ms"
                            + (f"  infer={_t_infer_ms:.1f}ms" if _t_infer_ms > 0 else "")
                            + f"  step_total={_t_step_total_ms:.1f}ms",
                            flush=True,
                        )
                        if eval_logger is not None and log_actions_jsonl:
                            eval_logger.append_action_jsonl(
                                episode,
                                {
                                    "episode": episode,
                                    "step": step,
                                    "chunk_timestep": t_exec,
                                    "state_observation": observation_for_json_log(
                                        raw_obs_now
                                    ),
                                    "policy_action": {
                                        k: (
                                            float(v.detach().cpu().item())
                                            if isinstance(v, torch.Tensor)
                                            else float(v)
                                        )
                                        for k, v in robot_action.items()
                                    },
                                    "xarm7": np.asarray(
                                        xarm_action["action"],
                                        dtype=np.float32,
                                    ).tolist(),
                                },
                            )
                        if eval_logger is not None and episode_dataset_json:
                            act_vec = [
                                float(
                                    robot_action[n].detach().cpu().item()
                                )
                                if isinstance(
                                    robot_action[n], torch.Tensor
                                )
                                else float(robot_action[n])
                                for n in action_names
                            ]
                            obs_vec = np.asarray(
                                raw_obs_now["observation.state"],
                                dtype=np.float32,
                            ).reshape(-1)
                            rec = build_frame_record_leobot_format(
                                frame_index=step,
                                episode_index=episode,
                                index=eval_logger.alloc_global_index(),
                                task_index=task_index_eval,
                                timestamp=eval_logger.step_timestamp(
                                    step
                                ),
                                observation_state=obs_vec.tolist(),
                                action=act_vec,
                                image_path_front=img_paths.get(
                                    "observation.images.front"
                                ),
                            )
                            eval_logger.append_dataset_frame(episode, rec)
                        chunk_cursor += 1

                        if dt > 0:
                            precise_sleep(dt)
                    except KeyboardInterrupt:
                        print(
                            "\nCtrl+C: stopping this episode; moving to next episode.",
                            flush=True,
                        )
                        # save the current frame when Ctrl+C is pressed.
                        if eval_logger is not None and episode_dataset_json:
                            try:
                                act_vec = [
                                    float(
                                        robot_action[n].detach().cpu().item()
                                    )
                                    if isinstance(
                                        robot_action[n], torch.Tensor
                                    )
                                    else float(robot_action[n])
                                    for n in action_names
                                ]
                                obs_vec = np.asarray(
                                    raw_obs_now["observation.state"],
                                    dtype=np.float32,
                                ).reshape(-1)
                                rec = build_frame_record_leobot_format(
                                    frame_index=step,
                                    episode_index=episode,
                                    index=eval_logger.alloc_global_index(),
                                    task_index=task_index_eval,
                                    timestamp=eval_logger.step_timestamp(
                                        step
                                    ),
                                    observation_state=obs_vec.tolist(),
                                    action=act_vec,
                                    image_path_front=img_paths.get(
                                        "observation.images.front"
                                    ),
                                )
                                eval_logger.append_dataset_frame(episode, rec)
                            except Exception as e:
                                print(f"Warning: Failed to save observation on Ctrl+C: {e}", flush=True)
                        break
                 
            finally:
                if eval_logger is not None and episode_dataset_json:
                    eval_logger.finalize_episode_dataset_json(
                        episode, features_doc
                    )

if __name__ == "__main__":
    main()
