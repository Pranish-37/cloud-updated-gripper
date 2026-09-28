import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 11
ACTION_DIM = 4
LATENT_DIM = 32
CHUNK_SIZE = 24

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def resolve_dataset_path():
    env_path = os.environ.get("DATASET_PATH")
    if env_path:
        return env_path
    for p in [
        "/workspace/episodes",
        "/home/jin/RPL/CloudGripper_V2_ROS/episodes",
        "./episodes",
    ]:
        if os.path.exists(p):
            return p
    return "./episodes"


def resolve_checkpoint_path(custom_path=None):
    if custom_path:
        return custom_path
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path:
        return env_path
    for p in [
        "/workspace/checkpoints/best_act_cube_stacking.pth",
        "/home/jin/RPL/CloudGripper_V2_ROS/checkpoints/best_act_cube_stacking.pth",
        "./checkpoints/best_act_cube_stacking.pth",
    ]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No checkpoint found. Set CHECKPOINT_PATH or pass --checkpoint."
    )


def normalize_image_for_torch(img):
    """Convert an image into HWC uint8 RGB layout expected by the ACT policy."""
    arr = img
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] not in (1, 3):
            arr = arr[..., :3]

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    return np.ascontiguousarray(arr)


class CVAEEncoder(nn.Module):
    def __init__(self, state_dim=11, action_dim=4, chunk_size=24, latent_dim=32):
        super().__init__()
        input_dim = state_dim + (action_dim * chunk_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, state, actions):
        actions_flat = actions.reshape(actions.size(0), -1)
        x = torch.cat([state, actions_flat], dim=1)
        feat = self.encoder(x)
        return self.fc_mu(feat), self.fc_logvar(feat)


class ACTPolicy(nn.Module):
    def __init__(self, state_dim=11, action_dim=4, chunk_size=24, latent_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.cvae_encoder = CVAEEncoder(state_dim, action_dim, chunk_size, latent_dim)
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.state_proj = nn.Linear(state_dim, 128)
        self.latent_proj = nn.Linear(latent_dim, 64)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.query_embed = nn.Parameter(torch.randn(chunk_size, 256))

    def forward(self, image, state):
        B = image.size(0)
        z = torch.zeros(B, self.latent_dim, device=image.device)

        img_feats = self.backbone(image)
        state_feats = self.state_proj(state)
        latent_feats = self.latent_proj(z)

        memory = torch.cat([img_feats[:, :64], state_feats, latent_feats], dim=1).unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        pred_actions = self.action_head(decoded)
        return pred_actions


class ACTPolicyTester:
    def __init__(self, checkpoint_path, stats_path, device=DEVICE, camera_index=0):
        self.device = torch.device(device)
        self.camera_index = camera_index

        dataset_root = os.path.dirname(os.path.dirname(stats_path))
        info_path = os.path.join(dataset_root, "meta", "info.json")
        if not os.path.exists(info_path):
            info_path = os.path.join(dataset_root, "info.json")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        info = {}
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)

        self.state_mean = torch.as_tensor(stats["observation.state"]["mean"], dtype=torch.float32)
        self.state_std = torch.as_tensor(stats["observation.state"]["std"], dtype=torch.float32) + 1e-8
        self.action_mean = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32)
        self.action_std = torch.as_tensor(stats["action"]["std"], dtype=torch.float32) + 1e-8

        feature_meta = info.get("features", {})
        if not feature_meta:
            feature_meta = stats.get("features", {})

        if "observation.images.front" in feature_meta:
            self.expected_image_shape = tuple(feature_meta["observation.images.front"]["shape"])
        else:
            self.expected_image_shape = (480, 640, 3)

        if "observation.state" in feature_meta:
            self.expected_state_dim = int(feature_meta["observation.state"]["shape"][0])
        else:
            self.expected_state_dim = STATE_DIM

        if "action" in feature_meta:
            self.expected_action_dim = int(feature_meta["action"]["shape"][0])
        else:
            self.expected_action_dim = ACTION_DIM

        self.model = ACTPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.cam = cv2.VideoCapture(camera_index)
        if not self.cam.isOpened():
            print(f"Warning: camera index {camera_index} could not be opened; using a synthetic frame instead.")
            self.cam = None

    def validate_frame(self, frame):
        frame = normalize_image_for_torch(frame)
        if frame.shape[:2] != (self.expected_image_shape[0], self.expected_image_shape[1]):
            raise ValueError(
                f"Camera image shape mismatch: got {frame.shape[:2]}, expected {self.expected_image_shape[:2]}."
            )
        if frame.shape[2] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got shape {frame.shape}.")
        return frame

    def validate_state(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != STATE_DIM:
            raise ValueError(f"State dims mismatch: got {state.shape[0]}, expected {STATE_DIM}.")
        return state

    def read_camera_frame(self):
        if self.cam is not None:
            ok, frame = self.cam.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return rgb
            print("Camera read failed; using synthetic fallback frame.")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def read_joint_state(self):
        raw = os.environ.get("JOINT_STATE")
        if raw:
            try:
                arr = np.asarray(json.loads(raw), dtype=np.float32)
                return self.validate_state(arr)
            except Exception:
                pass
        return np.zeros(STATE_DIM, dtype=np.float32)

    def preprocess(self, raw_frame_rgb, raw_joint_state):
        frame = self.validate_frame(raw_frame_rgb)
        state = self.validate_state(raw_joint_state)

        image_tensor = IMAGE_TRANSFORMS(frame).unsqueeze(0).to(self.device)
        norm_state = (torch.as_tensor(state, dtype=torch.float32).to(self.device) - self.state_mean.to(self.device)) / self.state_std.to(self.device)
        state_tensor = norm_state.unsqueeze(0)
        return image_tensor, state_tensor

    def postprocess_action(self, norm_action):
        norm_action_tensor = torch.as_tensor(norm_action, dtype=torch.float32, device=self.device)
        action = norm_action_tensor * self.action_std.to(self.device) + self.action_mean.to(self.device)
        return action.cpu().numpy()

    @torch.no_grad()
    def step(self, raw_frame_rgb=None, raw_joint_state=None):
        if raw_frame_rgb is None:
            raw_frame_rgb = self.read_camera_frame()
        if raw_joint_state is None:
            raw_joint_state = self.read_joint_state()

        image_tensor, state_tensor = self.preprocess(raw_frame_rgb, raw_joint_state)
        pred_actions_norm = self.model(image_tensor, state_tensor).squeeze(0)
        pred_actions = self.postprocess_action(pred_actions_norm)
        return pred_actions


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACT policy inference on a live camera frame or a synthetic fallback.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the trained ACT checkpoint (.pth).")
    parser.add_argument("--dataset", type=str, default=resolve_dataset_path(), help="Dataset root containing meta/stats.json.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index to test live inference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    parser.add_argument("--once", action="store_true", help="Run one inference pass and exit.")
    parser.add_argument("--fps", type=float, default=10.0, help="Inference loop rate in Hz when not --once.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset
    stats_path = os.path.join(dataset_root, "meta", "stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"stats.json not found at {stats_path}. Check the dataset path and camera layout.")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    tester = ACTPolicyTester(checkpoint_path, stats_path, device=args.device, camera_index=args.camera_index)

    print("\n=== ACT Policy Test ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset:    {dataset_root}")
    print(f"camera:     index {args.camera_index} | device={args.device}")
    print(f"image layout expected: {tester.expected_image_shape}")
    print(f"state shape expected: {STATE_DIM}")
    print(f"action dim expected:  {ACTION_DIM}\n")

    loop_count = 0
    try:
        while True:
            action_chunk = tester.step()
            first_action = action_chunk[0]
            print(f"step {loop_count:03d} | action[0] = {np.round(first_action, 4)} | action_chunk_shape={action_chunk.shape}")

            if args.once:
                break
            loop_count += 1
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        print("\nInference stopped by user.")


if __name__ == "__main__":
    main()
