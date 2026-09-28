import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
DATASET_PATH = os.environ.get("DATASET_PATH") or (
    "/workspace/episodes" if os.path.exists("/workspace/episodes") else "./episodes"
)
REPO_ID = os.environ.get("REPO_ID") or "local/dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
CHUNK_SIZE = 24             # Action chunk horizon (k = 24 steps into future)

# CVAE Latent Regularization Fixes
KL_WEIGHT = 0.1            # Lowered from 10.0 to prevent CVAE collapse
KL_WARMUP_EPOCHS = 10      # Gradually ramp up KL penalty over first 10 epochs

VAL_SPLIT_RATIO = 0.15      # ~4 episodes out of 25 for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def normalize_image_for_torch(img):
    """Convert LeRobot image tensors to a standard RGB numpy array [H, W, C]."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim == 3:
        # Handle channel-first tensors like [C, H, W] or [H, W, C].
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] not in (1, 3):
            # Fallback for unexpected tensor layout: prefer channel-first image.
            if arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            else:
                arr = arr[..., :3]

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


# ==============================================================================
# 2. ACT POLICY ARCHITECTURE (Action Chunking with Transformers)
# ==============================================================================
class CVAEEncoder(nn.Module):
    """Encodes (State + Action Chunk) into latent Gaussian (mu, logvar) during training."""
    def __init__(self, state_dim=11, action_dim=4, chunk_size=16, latent_dim=32):
        super().__init__()
        input_dim = state_dim + (action_dim * chunk_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, state, actions):
        # actions shape: [B, chunk_size, action_dim] -> flatten to [B, chunk_size * action_dim]
        actions_flat = actions.reshape(actions.size(0), -1)
        x = torch.cat([state, actions_flat], dim=1)
        feat = self.encoder(x)
        return self.fc_mu(feat), self.fc_logvar(feat)


class ACTPolicy(nn.Module):
    def __init__(self, state_dim=11, action_dim=4, chunk_size=16, latent_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # CVAE Encoder for training
        self.cvae_encoder = CVAEEncoder(state_dim, action_dim, chunk_size, latent_dim)

        # Vision Backbone (ResNet18)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        # Transformer Decoder for Action Sequence Prediction
        self.state_proj = nn.Linear(state_dim, 128)
        self.latent_proj = nn.Linear(latent_dim, 64)
        
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Action Trajectory Output Head
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Fixed Positional Embeddings for the k-step action horizon
        self.query_embed = nn.Parameter(torch.randn(chunk_size, 256))

    def forward(self, image, state, actions=None):
        B = image.size(0)

        # 1. Handle CVAE Latent Vector
        if self.training and actions is not None:
            mu, logvar = self.cvae_encoder(state, actions)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            mu, logvar = None, None
            z = torch.zeros(B, self.latent_dim, device=image.device)  # Deterministic prior at inference

        # 2. Extract Features
        img_feats = self.backbone(image)                         # [B, 512] -> project down
        state_feats = self.state_proj(state)                    # [B, 128]
        latent_feats = self.latent_proj(z)                      # [B, 64]

        # Fusion memory sequence
        memory = torch.cat([img_feats[:, :64], state_feats, latent_feats], dim=1).unsqueeze(1) # [B, 1, 256]

        # 3. Decode Action Chunk Sequence
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1) # [B, chunk_size, 256]
        decoded = self.transformer_decoder(tgt=queries, memory=memory) # [B, chunk_size, 256]

        # 4. Predict Action Horizon [B, chunk_size, action_dim]
        pred_actions = self.action_head(decoded)

        return pred_actions, mu, logvar


# ==============================================================================
# 3. DATA COLLATOR (Handles k-step Chunk Slicing & Normalization)
# ==============================================================================
class ACTDataCollator:
    def __init__(self, stats_path, chunk_size=16):
        self.chunk_size = chunk_size
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.state_mean = torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["observation.state"]["std"], dtype=torch.float32) + 1e-8

        self.action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32) + 1e-8

    def __call__(self, batch):
        images, states, action_chunks = [], [], []

        for sample in batch:
            # Image frame at time t
            img = normalize_image_for_torch(sample["observation.images.front"])
            images.append(IMAGE_TRANSFORMS(img))

            # State at time t (Warning-free tensor extraction)
            st = sample["observation.state"]
            if isinstance(st, torch.Tensor):
                st = st.clone().detach().to(dtype=torch.float32)
            else:
                st = torch.tensor(st, dtype=torch.float32)

            states.append((st - self.state_mean) / self.state_std)

            # Build action sequence [A_t, A_{t+1}, ..., A_{t+k-1}]
            # (Warning-free tensor extraction)
            act = sample["action"]
            if isinstance(act, torch.Tensor):
                act = act.clone().detach().to(dtype=torch.float32)
            else:
                act = torch.tensor(act, dtype=torch.float32)

            act_norm = (act - self.action_mean) / self.action_std

            # If dataset sample contains pre-chunked actions, use directly; otherwise construct window
            if act_norm.ndim == 1:
                chunk = act_norm.unsqueeze(0).repeat(self.chunk_size, 1) # Fallback single step expansion
            else:
                chunk = act_norm[:self.chunk_size]

            action_chunks.append(chunk)

        return (
            torch.stack(images),
            torch.stack(states),
            torch.stack(action_chunks)
        )


# ==============================================================================
# 4. TRAINING & EVALUATION LOOP
# ==============================================================================
def train():
    print("Loading LeRobot Dataset for ACT Policy Training...")
    dataset = LeRobotDataset(repo_id=REPO_ID, root=DATASET_PATH, video_backend='pyav')
    stats_path = os.path.join(DATASET_PATH, "meta", "stats.json")
    collator = ACTDataCollator(stats_path, chunk_size=CHUNK_SIZE)

    # Train/Validation Split by Episode Index
    num_episodes = 25
    val_episodes = set(np.random.choice(num_episodes, int(num_episodes * VAL_SPLIT_RATIO), replace=False))

    train_indices, val_indices = [], []
    for idx, sample in enumerate(dataset):
        ep_id = sample["episode_index"].item() if isinstance(sample["episode_index"], torch.Tensor) else sample["episode_index"]
        if ep_id in val_episodes:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    OPTIMIZED_BATCH_SIZE = 64 

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=OPTIMIZED_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=8,           # Use 8 CPU worker processes
        pin_memory=True,         # Locks memory for fast CPU-to-GPU transfer
        prefetch_factor=4,       # Pre-loads 4 batches per worker into RAM
        persistent_workers=True  # Keeps CPU worker processes alive between epochs
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=OPTIMIZED_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Dataset Ready. Train Samples: {len(train_indices)} | Val Samples: {len(val_indices)}")

    # Model, Optimizer, Loss Functions
    model = ACTPolicy(state_dim=11, action_dim=4, chunk_size=CHUNK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    l1_loss_fn = nn.L1Loss()

    os.makedirs("./checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_l1, train_kl = 0.0, 0.0

        # Linear KL divergence loss warmup factor (0.0 -> 1.0) over first 10 epochs
        kl_warmup = min(1.0, epoch / KL_WARMUP_EPOCHS)

        for images, states, actions in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            states = states.to(DEVICE, non_blocking=True)
            actions = actions.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred_actions, mu, logvar = model(images, states, actions)

            # L1 Reconstruction Loss over the action horizon
            l1_loss = l1_loss_fn(pred_actions, actions)

            # KL Divergence Loss for CVAE Latent Space
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)

            # Total Loss with KL Warmup applied
            total_loss = l1_loss + (KL_WEIGHT * kl_warmup * kl_loss)
            total_loss.backward()
            optimizer.step()

            train_l1 += l1_loss.item() * images.size(0)
            train_kl += kl_loss.item() * images.size(0)

        train_l1 /= len(train_indices)
        train_kl /= len(train_indices)

        # Validation Phase
        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for images, states, actions in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                states = states.to(DEVICE, non_blocking=True)
                actions = actions.to(DEVICE, non_blocking=True)

                pred_actions, _, _ = model(images, states)  # Deterministic inference mode
                val_l1 += l1_loss_fn(pred_actions, actions).item() * images.size(0)

        val_l1 /= len(val_indices)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Train L1: {train_l1:.4f} | Train KL: {train_kl:.4f} | Val L1: {val_l1:.4f}")

        # Save Best Checkpoint
        if val_l1 < best_val_loss:
            best_val_loss = val_l1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_l1_loss": val_l1,
                "chunk_size": CHUNK_SIZE
            }, "./checkpoints/best_act_cube_stacking.pth")
            print(" --> Saved best ACT model checkpoint.")

if __name__ == "__main__":
    train()
