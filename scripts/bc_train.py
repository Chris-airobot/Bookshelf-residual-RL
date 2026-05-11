"""Behaviour Cloning training script for Bookshelf-Direct-v4.

Loads expert demos from data/expert/*.pt, trains a small MLP policy
(obs 10D -> action 5D), saves the best checkpoint to data/bc/.

No Isaac dependency -- pure PyTorch, runs offline.

Usage:
    python scripts/bc_train.py
    python scripts/bc_train.py --data_dir data/expert --out_dir data/bc --epochs 200
"""

import argparse
import glob
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """Small MLP: obs (10D) -> action (5D)."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Obs dim index for the mode channel (obs[0] = mode).
# From cfg: mode_obs_scripted = 0.5 -- we skip those steps.
_SCRIPTED_MODE_VALUE = 0.5
_SCRIPTED_TOL = 0.05   # tolerance for floating-point comparison


class DemoDataset(Dataset):
    """Flat (obs, action) dataset built from a list of demo .pt files.

    SCRIPTED-mode steps are always excluded: during SCRIPTED the policy
    has no control, so those pairs carry no signal.

    Idle steps (human not pressing anything, action=[0,0,0,0,-1]) make up
    ~97% of recorded steps when record_idle=True.  Training on them teaches
    the model to always output zeros, completely overwhelming the real signal.
    Set commanded_only=True (the default) to keep only steps where the human
    actively typed a command.
    """

    def __init__(self, demo_paths: list[str], commanded_only: bool = True,
                 obs_mean: torch.Tensor | None = None,
                 obs_std: torch.Tensor | None = None):
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

        total_files = 0
        total_steps = 0
        scripted_skipped = 0
        idle_skipped = 0

        for path in demo_paths:
            demo = torch.load(path, map_location="cpu", weights_only=False)
            total_files += 1
            for step in demo["steps"]:
                obs = step["obs"].float()    # shape (10,)
                act = step["action"].float() # shape (5,)

                # Skip SCRIPTED phase: policy has no control here.
                mode_val = obs[0].item()
                if abs(mode_val - _SCRIPTED_MODE_VALUE) < _SCRIPTED_TOL:
                    scripted_skipped += 1
                    continue

                # Skip idle steps if requested: they dominate the dataset
                # (~97%) and cause the model to collapse to predicting zeros.
                if commanded_only and not step["commanded"]:
                    idle_skipped += 1
                    continue

                self.obs.append(obs)
                self.actions.append(act)
                total_steps += 1

        self.obs = torch.stack(self.obs)         # (N, 10)
        self.actions = torch.stack(self.actions)  # (N, 5)

        # Apply obs normalisation if stats are provided.
        if obs_mean is not None and obs_std is not None:
            self.obs = (self.obs - obs_mean) / obs_std

        print(
            f"  Loaded {total_files} demos, "
            f"{total_steps} steps kept "
            f"({scripted_skipped} SCRIPTED + {idle_skipped} idle skipped)."
        )

        # Count release trigger events (action[4] > 0) for diagnostic.
        release_count = (self.actions[:, 4] > 0).sum().item()
        print(
            f"  Release trigger events: {release_count} / {total_steps} "
            f"({100*release_count/max(total_steps,1):.2f}%)"
        )

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return self.obs[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Combined loss:
    - MSE on motion dims action[0:4]
    - BCE on release trigger action[4]  (target: -1 -> 0, +1 -> 1)
    """
    mse_loss = nn.functional.mse_loss(pred[:, :4], target[:, :4])

    # Convert -1/+1 to 0/1 for BCE.
    trigger_label = (target[:, 4] > 0).float()
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred[:, 4], trigger_label
    )

    total = mse_loss + bce_weight * bce_loss
    return total, {"mse": mse_loss.item(), "bce": bce_loss.item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_obs_stats(all_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-dim mean and std from ALL non-SCRIPTED steps across all demos.

    Uses all steps (including idle) so the stats represent the full obs
    distribution the PPO policy will encounter during training — not just
    the commanded steps BC trains on.
    """
    all_obs = []
    for path in all_paths:
        demo = torch.load(path, map_location="cpu", weights_only=False)
        for step in demo["steps"]:
            obs = step["obs"].float()
            mode_val = obs[0].item()
            if abs(mode_val - _SCRIPTED_MODE_VALUE) < _SCRIPTED_TOL:
                continue
            all_obs.append(obs)
    obs_tensor = torch.stack(all_obs)          # (N_total, 10)
    mean = obs_tensor.mean(0)                   # (10,)
    std = obs_tensor.std(0).clamp(min=1e-8)    # (10,)
    return mean, std


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load and split demos by file (not by step, to avoid data leakage) ---
    all_paths = sorted(glob.glob(os.path.join(args.data_dir, "demo_*.pt")))
    if not all_paths:
        raise FileNotFoundError(f"No demo_*.pt files found in {args.data_dir}")

    random.shuffle(all_paths)
    n_val = max(1, int(len(all_paths) * args.val_split))
    val_paths = all_paths[:n_val]
    train_paths = all_paths[n_val:]

    print(f"\nDataset split: {len(train_paths)} train demos, {len(val_paths)} val demos")

    # Compute normalization stats from ALL demos (train + val) so the stats
    # represent the full env obs distribution, not just the train split.
    print("Computing obs normalization stats from all demos...")
    obs_mean, obs_std = compute_obs_stats(all_paths)
    print(f"  obs_mean: {obs_mean.tolist()}")
    print(f"  obs_std:  {obs_std.tolist()}")

    print("Loading train data...")
    train_ds = DemoDataset(train_paths, commanded_only=args.commanded_only,
                           obs_mean=obs_mean, obs_std=obs_std)
    print("Loading val data...")
    val_ds = DemoDataset(val_paths, commanded_only=args.commanded_only,
                         obs_mean=obs_mean, obs_std=obs_std)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    # --- Model + optimiser ---
    model = BCPolicy(obs_dim=10, action_dim=5, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --- Training loop ---
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1

    print(f"\nTraining for {args.epochs} epochs...\n")
    header = f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'MSE':>10}  {'BCE':>10}  {'LR':>10}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = model(obs_b)
            loss, _ = compute_loss(pred, act_b, bce_weight=args.bce_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(obs_b)

        train_loss = train_loss_sum / len(train_ds)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_bce_sum = 0.0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                pred = model(obs_b)
                loss, breakdown = compute_loss(pred, act_b, bce_weight=args.bce_weight)
                n = len(obs_b)
                val_loss_sum += loss.item() * n
                val_mse_sum += breakdown["mse"] * n
                val_bce_sum += breakdown["bce"] * n

        val_loss = val_loss_sum / len(val_ds)
        val_mse = val_mse_sum / len(val_ds)
        val_bce = val_bce_sum / len(val_ds)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt_path = os.path.join(args.out_dir, "bc_policy_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "obs_dim": 10,
                    "action_dim": 5,
                    "hidden": args.hidden,
                    "model_state_dict": model.state_dict(),
                    "obs_mean": obs_mean.cpu(),
                    "obs_std": obs_std.cpu(),
                    "args": vars(args),
                },
                ckpt_path,
            )

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            marker = " *" if epoch == best_epoch else ""
            print(
                f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
                f"{val_mse:>10.6f}  {val_bce:>10.6f}  {lr:>10.2e}{marker}"
            )

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {os.path.join(args.out_dir, 'bc_policy_best.pt')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BC training for Bookshelf-v4")
    parser.add_argument("--data_dir", type=str, default="data/expert")
    parser.add_argument("--out_dir", type=str, default="data/bc")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of demos held out for validation")
    parser.add_argument("--bce_weight", type=float, default=1.0,
                        help="Weight of the BCE (release trigger) loss term")
    parser.add_argument("--commanded_only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Train only on steps where the human typed a command "
                             "(default: True). Idle steps (~97%% of data) cause the "
                             "model to collapse to predicting zeros.")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
