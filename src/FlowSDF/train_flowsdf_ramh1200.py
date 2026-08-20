#!/usr/bin/env python3
"""Train FlowSDF MoE experts on RAM-H1200 gated fragment data.

Fork of train_flowsdf_moe.py (PENGWIN) -- same training loop, loss, EMA,
checkpoint format, resume, and early-stopping logic unchanged, since none of
that is PENGWIN-specific (it only ever touches FragmentDataset-shaped
batches). The only real change is the dataset import: ramh1200_dataset
instead of dataset, which reads RAM-H1200's own gated_ramh1200_<split>_records.csv
and RAM-H1200's own per-fragment GT masks (not PENGWIN's bit-packed labels).

Before running: build_gated_csv (via `python
src/gating_mechanism/ramh1200_dataset.py --split train --threshold <px>` and
`--split val`) must exist, using a threshold derived from
ramh1200_threshold_analysis.ipynb run on the TRAIN split.

Pipeline per batch:
    1. Load embedding (256, 64, 64), binary_mask (1, 1024, 1024), gt_mask (1, 448, 448)
    2. Resize embedding and binary_mask to training size
    3. Resize gt_mask to training size and convert it to the required SDF
    4. Concatenate embedding + binary_mask as image conditioning
    5. Train FlowSDF with gt_mask SDF as the target

    6. Loss: MSE flow matching loss

Two experts (expert_small, expert_large) are trained separately, one at a time.

Usage:
    python train_flowsdf_ramh1200.py [--expert expert_small|expert_large|both] [OPTIONS]

    python train_flowsdf_ramh1200.py --expert expert_small --epochs 100 --batch-size 8
    python train_flowsdf_ramh1200.py --expert both --epochs 100 --large-subsample 8000
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # MoE-ShapeRefine-MedicalSeg
SRC_DIR      = PROJECT_ROOT / "src"
GATING_DIR   = SRC_DIR / "gating_mechanism"

for _p in (str(SRC_DIR), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdf_utils import mask_to_flowsdf  # noqa: E402
from ramh1200_dataset import build_expert_dataloaders, FragmentDataset  # noqa: E402
from models import unet_segdiff  # noqa: E402


# Reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# W&B helpers — all wandb calls isolated so code works without W&B
# ---------------------------------------------------------------------------

def wandb_init(args: argparse.Namespace, expert_id: str, resume_run_id: str | None = None) -> bool:
    """Initialize W&B run. Returns True if W&B is active."""
    if args.no_wandb:
        return False
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{expert_id}__flowsdf_ramh1200_expert",
            id=resume_run_id,
            resume="allow" if resume_run_id else None,
            config={
                "expert": expert_id,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "ema_decay": args.ema_decay,
                "sigma_min": args.sigma_min,
                "img_size": args.img_size,
                "sdf_threshold": args.sdf_threshold,
                "clip_grad": args.clip_grad,
                "img_cond_channels": args.img_cond_channels,
                "large_subsample": args.large_subsample,
                "early_stopping_patience": args.early_stopping_patience,
            },
        )
        return True
    except Exception as exc:
        print(f"WARNING: W&B init failed ({exc}). Continuing without W&B.")
        return False


def wandb_log(metrics: dict, step: int, active: bool) -> None:
    if not active:
        return
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        pass


def wandb_finish(active: bool) -> None:
    if not active:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


class MoEFlowSDFDataset(Dataset):
    """Wraps FragmentDataset for FlowSDF training.

    Converts the gated fragment data (embedding, binary_mask, gt_mask)
    into FlowSDF training format by resizing to a target image size.
    """

    def __init__(
        self,
        fragment_dataset: FragmentDataset,
        img_size: int = 128,
        sdf_threshold: float = 15.0,
    ):
        """
        Args:
            fragment_dataset: FragmentDataset instance from ramh1200_dataset
            img_size: target training resolution
            sdf_threshold: distance clipping threshold before SDF normalization
        """
        self.fragment_dataset = fragment_dataset
        self.img_size = img_size
        self.sdf_threshold = sdf_threshold

    def __len__(self):
        return len(self.fragment_dataset)

    def __getitem__(self, idx):
        batch_item = self.fragment_dataset[idx]

        # Extract data already loaded by FragmentDataset. The embedding is the
        # precomputed MedSAM image encoder output also used by the cnnNoROI pipeline.
        embedding = batch_item["embedding"]  # (256, 64, 64) torch tensor
        binary_mask = batch_item["binary_mask"]  # (1, 1024, 1024) torch tensor
        gt_mask = batch_item["gt_mask"]  # (1, 448, 448) torch tensor

        embedding_resized = F.interpolate(
            embedding.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (256, img_size, img_size)

        mask_resized = F.interpolate(
            binary_mask.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="nearest",
        ).squeeze(0)  # (1, img_size, img_size)

        gt_resized = F.interpolate(
            gt_mask.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # (1, img_size, img_size)

        # Binarize conditioning mask; convert target mask to FlowSDF SDF target.
        mask_resized = (mask_resized > 0.5).float()
        gt_binary = (gt_resized > 0.5).float()
        gt_sdf = mask_to_flowsdf(gt_binary.squeeze(0), threshold=self.sdf_threshold)

        img_cond = torch.cat([embedding_resized, mask_resized], dim=0)

        # Return in FlowSDF format:
        # - image: MedSAM image embedding + coarse mask conditioning
        # - mask: FlowSDF target from the ground-truth mask
        return {
            "image": img_cond,  # (257, img_size, img_size)
            "mask": gt_sdf,  # (img_size, img_size)
        }


def check_disk_space(path: Path, min_gb: float = 5.0) -> bool:
    """Check if path has at least min_gb free space. Returns True if OK."""
    try:
        stat = os.statvfs(str(path))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if free_gb < min_gb:
            print(f"WARNING: Only {free_gb:.1f} GB free at {path} (need {min_gb} GB)")
            return False
        return True
    except Exception as e:
        print(f"WARNING: Could not check disk space: {e}")
        return True


def save_checkpoint_safely(
    checkpoint: dict,
    final_path: Path,
    tmp_dir: str = "/tmp",
    expert_id: str = "unknown",
) -> None:
    """Save checkpoint safely by writing to temp file first, then atomically moving.

    This avoids incomplete writes to network filesystems due to buffer flushes.

    Args:
        checkpoint: dict to save
        final_path: final destination path
        tmp_dir: temporary directory (preferably local, not network)
        expert_id: for logging
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Check disk space before attempting save
    if not check_disk_space(final_path.parent, min_gb=3.0):
        print(f"ERROR: Insufficient disk space to save {final_path}")
        raise RuntimeError(f"Disk space error when saving {final_path}")

    try:
        # Create temp file in local storage (not GPFS) for faster, more reliable write
        with tempfile.NamedTemporaryFile(
            suffix=".pth", dir=tmp_dir, delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        print(f"  Saving {expert_id} checkpoint to temp: {tmp_path}")
        torch.save(checkpoint, tmp_path)

        # Verify temp file was written correctly
        temp_file_size = os.path.getsize(tmp_path)
        if temp_file_size == 0:
            raise RuntimeError(f"Temp checkpoint is empty: {tmp_path}")

        print(f"  Moving checkpoint to final location: {final_path} ({temp_file_size / 1e6:.1f} MB)")
        # Atomic move (shutil.move falls back to copy if atomic rename fails)
        shutil.move(tmp_path, str(final_path))
        print(f"  ✓ Saved: {final_path}")

    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        print(f"ERROR saving checkpoint {final_path}: {e}")
        raise


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def update_ema_variables(model, ema_model, ema_decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.copy_(ema_param.data * ema_decay + (1 - ema_decay) * param.data)


def compute_loss(v, u):
    """Flow matching loss: MSE between predicted and target velocity."""
    return ((v - u) ** 2).mean()


def run_one_epoch(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None,
    ema_decay: float,
    sigma_min: float,
    clip_grad: float | None,
    desc: str = "epoch",
) -> float:
    """One training or validation epoch. Returns mean loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            m = batch["mask"].to(device).unsqueeze(1)  # (B, 1, H, W)
            x = batch["image"].to(device)  # (B, 257, H, W) — MedSAM embedding + mask conditioning
            if x.shape[1] != model.rrdb.conv_first.in_channels:
                raise ValueError(
                    f"conditioning has {x.shape[1]} channels, but model expects "
                    f"{model.rrdb.conv_first.in_channels}"
                )
            batch_size = m.shape[0]

            # Sample random timesteps
            t = torch.rand(batch_size).float().to(device)

            # Sample noise
            eta = torch.randn_like(m).to(device)

            # Compute noisy mask
            sigma_t = 1 - (1 - sigma_min) * t
            mu_t = t[:, None, None, None] * m
            mt = mu_t + sigma_t[:, None, None, None] * eta

            # Target velocity
            u = (m - (1 - sigma_min) * mt) / (1 - (1 - sigma_min) * t[:, None, None, None])

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                # Predicted velocity (conditioned on image)
                v = model(mt, t.reshape(batch_size, -1), img_cond=x)
                # Compute loss
                loss = compute_loss(v, u)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad, norm_type="inf"
                    )
                optimizer.step()
                update_ema_variables(model, ema_model, ema_decay)

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_expert(
    expert_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    cfg_dict: dict,
) -> None:
    """Train one expert."""

    # Initialize model
    model = unet_segdiff.UNetModel(
        in_channels=cfg_dict["model"]["n_cin"],
        model_channels=cfg_dict["model"]["n_fm"],
        out_channels=cfg_dict["model"]["n_cin"],
        num_res_blocks=3,
        attention_resolutions=(16, 8),
        dropout=0,
        channel_mult=tuple(cfg_dict["model"]["mults"]),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=12,
        img_cond_channels=args.img_cond_channels,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"  Network has {count_parameters(model)} parameters")

    # Setup checkpoint dir
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_epoch = 1
    wandb_run_id = None
    patience_counter = 0

    # Resume from latest checkpoint if requested
    latest_ckpt = args.checkpoint_dir / f"{expert_id}_latest.pth"
    if args.resume and latest_ckpt.exists():
        print(f"  Resuming {expert_id} from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema_model.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        start_epoch = ckpt["epoch"] + 1
        wandb_run_id = ckpt.get("wandb_run_id")
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"  Resumed at epoch {start_epoch}/{args.epochs}, best_val_loss={best_val_loss:.4f}, patience={patience_counter}/{args.early_stopping_patience}")
    elif args.resume:
        print(f"  --resume set but no checkpoint found at {latest_ckpt}, starting fresh.")

    if start_epoch > args.epochs:
        print(f"  {expert_id} already completed {args.epochs} epochs, skipping.")
        return

    # Initialize W&B
    wb_active = wandb_init(args, expert_id, resume_run_id=wandb_run_id)

    stopped_early = False
    early_stopping_on = args.early_stopping_patience > 0
    # epoch is always defined after the loop because start_epoch <= args.epochs is guaranteed above
    epoch = start_epoch - 1

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        tag = f"[{expert_id}] {epoch}/{args.epochs}"

        train_loss = run_one_epoch(
            model, ema_model, train_loader, device, optimizer,
            args.ema_decay, args.sigma_min, args.clip_grad,
            desc=f"{tag} train",
        )

        val_loss = run_one_epoch(
            model, ema_model, val_loader, device, None,
            args.ema_decay, args.sigma_min, args.clip_grad,
            desc=f"{tag} val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Track improvement for both best-checkpoint saving and early stopping
        improved = val_loss < best_val_loss
        if early_stopping_on:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

        patience_str = f"  patience={patience_counter}/{args.early_stopping_patience}" if early_stopping_on else ""
        print(f"{tag}  train={train_loss:.4f}  val={val_loss:.4f}{patience_str}")

        # Log to W&B
        wb_metrics = {
            f"{expert_id}/train_loss": train_loss,
            f"{expert_id}/val_loss": val_loss,
        }
        if early_stopping_on:
            wb_metrics[f"{expert_id}/early_stopping_patience_counter"] = patience_counter
        wandb_log(wb_metrics, step=epoch, active=wb_active)

        # Save best checkpoint whenever val_loss strictly improves
        if improved:
            best_val_loss = val_loss
            best_ckpt_path = args.checkpoint_dir / f"{expert_id}_best.pth"
            save_checkpoint_safely(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "ema_state": ema_model.state_dict(),
                    "val_loss": val_loss,
                    "img_cond_channels": args.img_cond_channels,
                    "sdf_threshold": args.sdf_threshold,
                },
                best_ckpt_path,
                expert_id=f"{expert_id}_best",
            )

        # Save latest checkpoint every epoch for resume support
        _wb_run_id = None
        try:
            import wandb as _wandb
            if wb_active and _wandb.run is not None:
                _wb_run_id = _wandb.run.id
        except Exception:
            pass
        save_checkpoint_safely(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "patience_counter": patience_counter,
                "img_cond_channels": args.img_cond_channels,
                "sdf_threshold": args.sdf_threshold,
                "wandb_run_id": _wb_run_id,
            },
            latest_ckpt,
            expert_id=f"{expert_id}_latest",
        )

        if early_stopping_on and patience_counter >= args.early_stopping_patience:
            print(
                f"[{expert_id}] Early stopping at epoch {epoch}: "
                f"val_loss did not improve for {args.early_stopping_patience} consecutive epochs "
                f"(best val_loss={best_val_loss:.4f})"
            )
            wandb_log({f"{expert_id}/early_stopped_epoch": epoch}, step=epoch, active=wb_active)
            stopped_early = True
            break

    # Save final checkpoint at the actual last epoch reached (correct on resume + early stop)
    final = args.checkpoint_dir / f"{expert_id}_final.pth"
    save_checkpoint_safely(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema_model.state_dict(),
            "img_cond_channels": args.img_cond_channels,
            "sdf_threshold": args.sdf_threshold,
            "stopped_early": stopped_early,
        },
        final,
        expert_id=f"{expert_id}_final",
    )
    stop_reason = f"early stopping at epoch {epoch}" if stopped_early else f"completed {args.epochs} epochs"
    print(f"[{expert_id}] done ({stop_reason}). Final: {final}")

    # Save loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, train_losses, label="train loss")
        ax.plot(epochs_range, val_losses, label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{expert_id} — training curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out = args.checkpoint_dir / f"{expert_id}_loss_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Loss curve saved: {out}")
    except Exception as exc:
        print(f"  WARNING: could not save loss curve ({exc})")

    wandb_finish(wb_active)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--expert",
        choices=["expert_small", "expert_large", "both"],
        default="both",
        help="Which expert(s) to train",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--sigma-min", type=float, default=1e-5, help="Min sigma for diffusion")
    parser.add_argument("--img-size", type=int, default=128, help="Training image size")
    parser.add_argument(
        "--sdf-threshold",
        type=float,
        default=15.0,
        help="Distance clipping threshold in pixels before normalizing SDF targets to [-1, 1]",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Infinity-norm gradient clipping value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--img-cond-channels",
        type=int,
        default=257,
        help="Conditioning channels passed to FlowSDF RRDB: 256 MedSAM embedding + 1 coarse mask",
    )
    parser.add_argument(
        "--large-subsample",
        type=int,
        default=None,
        help="Subsample expert_large to this many fragments (check the actual small/large "
             "counts in gated_ramh1200_train_records.csv first -- RAM-H1200's imbalance is "
             "likely much smaller than PENGWIN's 194591:17335, may not need subsampling at all).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "flowsdf_ramh1200",
        help="Checkpoint directory",
    )
    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="moe-shaprefine",
        help="W&B project name.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging. Loss curve PNG is always saved locally.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from {expert_id}_latest.pth in checkpoint-dir if it exists.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help=(
            "Stop training if val_loss does not improve for this many consecutive epochs. "
            "Set to 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_grad is not None and args.clip_grad < 0:
        args.clip_grad = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Config for FlowSDF model
    cfg_dict = {
        "model": {
            "n_cin": 1,
            "n_fm": 128,
            "mults": [1, 1, 2, 2, 4, 4],
        },
        "learning": {
            "lr": args.lr,
            "ema_decay": args.ema_decay,
        },
    }

    # Build MoE dataloaders
    print("Building dataloaders …")
    train_loaders_dict = build_expert_dataloaders(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        large_subsample=args.large_subsample,
    )

    val_loaders_dict = build_expert_dataloaders(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Wrap base datasets with MoEFlowSDFDataset to resize and prepare for training
    train_loaders = {}
    val_loaders = {}

    for expert_id in ["expert_small", "expert_large"]:
        # Extract the FragmentDataset from the base DataLoader
        base_train_dataset = train_loaders_dict[expert_id].dataset
        base_val_dataset = val_loaders_dict[expert_id].dataset

        # Wrap with MoEFlowSDFDataset
        train_dataset = MoEFlowSDFDataset(
            base_train_dataset,
            img_size=args.img_size,
            sdf_threshold=args.sdf_threshold,
        )
        val_dataset = MoEFlowSDFDataset(
            base_val_dataset,
            img_size=args.img_size,
            sdf_threshold=args.sdf_threshold,
        )

        # Create new DataLoaders
        train_loaders[expert_id] = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        val_loaders[expert_id] = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    # Determine which experts to train
    experts = ["expert_small", "expert_large"] if args.expert == "both" else [args.expert]

    # Train each expert
    for expert_id in experts:
        print(f"\n{'='*60}\nTraining {expert_id}\n{'='*60}")
        train_expert(
            expert_id=expert_id,
            train_loader=train_loaders[expert_id],
            val_loader=val_loaders[expert_id],
            device=device,
            args=args,
            cfg_dict=cfg_dict,
        )


if __name__ == "__main__":
    main()
