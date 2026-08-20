#!/usr/bin/env python3
"""Train MoE CNN experts on RAM-H1200 gated fragment data — Alt 3 (no RoI crops).

Fork of train_moe.py (PENGWIN) -- only the dataset import changes
(ramh1200_dataset instead of dataset). Everything else (loss, optimizer,
checkpoint format) is unchanged, since it's architecture/training-procedure,
not data-source-specific.

Before running: build_gated_csv (via `python
src/gating_mechanism/ramh1200_dataset.py --split train --threshold <px>` and
`--split val`) must exist, using a threshold derived from
ramh1200_threshold_analysis.ipynb run on the TRAIN split.

Pipeline per batch:
    1. Resize binary_mask (1, 1024, 1024) → (1, 64, 64)   [nearest]
    2. Compute SDF on the 64×64 mask                       [scipy, CPU]
    3. Concatenate: embedding (256) + mask (1) + SDF (1) → (258, 64, 64)
    4. Forward through CNNExpert                           → (1, 64, 64)
    5. Upsample prediction → (1, 448, 448)                 [bilinear]
    6. Boundary-Dice-BCE loss against gt_mask (1, 448, 448)

Two experts (expert_small, expert_large) are trained separately, one at a time.

Usage:
    python train_moe_ramh1200.py [--expert expert_small|expert_large|both] [OPTIONS]

    python train_moe_ramh1200.py --expert expert_small --epochs 20 --batch-size 64
    python train_moe_ramh1200.py --expert both --epochs 20 --wandb-project moe-shaprefine
    python train_moe_ramh1200.py --no-wandb   # disable W&B, save loss curve locally only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
GATING_DIR   = SRC_DIR / "gating_mechanism"

for _p in (str(SRC_DIR), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdf_utils import sdf_channel_from_mask           # noqa: E402
from cnnNoROI.cnnMoE import CNNExpert                 # noqa: E402
from cnnNoROI.losses import boundary_dice_bce_loss    # noqa: E402
from ramh1200_dataset import build_expert_dataloaders  # noqa: E402
# --------------------------------------------------------------------------

FEAT_SIZE = 64
GT_SIZE   = 448


# ---------------------------------------------------------------------------
# W&B helpers — all wandb calls are isolated here so the rest of the code
# never imports wandb directly and works fine when --no-wandb is passed.
# ---------------------------------------------------------------------------

def wandb_init(args: argparse.Namespace, expert_id: str) -> bool:
    """Initialise a W&B run. Returns True if W&B is active."""
    if args.no_wandb:
        return False
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{expert_id}__{args.wandb_project}_ramh1200",
            config={
                "expert":          expert_id,
                "epochs":          args.epochs,
                "lr":              args.lr,
                "batch_size":      args.batch_size,
                "w_boundary":      args.w_boundary,
                "boundary_radius": args.boundary_radius,
                "dice_weight":     args.dice_weight,
                "large_subsample": args.large_subsample,
                "feat_size":       FEAT_SIZE,
                "gt_size":         GT_SIZE,
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


# ---------------------------------------------------------------------------
# Loss curve — always saved locally regardless of W&B
# ---------------------------------------------------------------------------

def save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    expert_id: str,
    checkpoint_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, label="train loss")
        ax.plot(epochs, val_losses,   label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{expert_id} — training curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out = checkpoint_dir / f"{expert_id}_loss_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Loss curve saved: {out}")
    except Exception as exc:
        print(f"  WARNING: could not save loss curve ({exc})")


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def compute_batch_sdf(mask_small: torch.Tensor) -> torch.Tensor:
    """Compute SDF for a batch of 64×64 masks on CPU, return (B, 1, H, W) float32."""
    sdfs = [
        sdf_channel_from_mask(mask_small[i, 0].cpu().numpy())
        for i in range(mask_small.size(0))
    ]
    return torch.from_numpy(np.stack(sdfs))


def run_one_epoch(
    model: CNNExpert,
    loader,
    device: torch.device,
    optimizer: AdamW | None,
    w_boundary: float,
    boundary_radius: int,
    dice_weight: float,
    desc: str,
) -> float:
    """One training or validation epoch. Returns mean loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            embedding   = batch["embedding"].to(device)      # (B, 256, 64, 64)
            binary_mask = batch["binary_mask"].to(device)    # (B, 1, 1024, 1024)
            gt_mask     = batch["gt_mask"].to(device)        # (B, 1, 448, 448)

            mask_small = F.interpolate(
                binary_mask, size=(FEAT_SIZE, FEAT_SIZE), mode="nearest"
            )                                                # (B, 1, 64, 64)

            sdf = compute_batch_sdf(mask_small).to(device)  # (B, 1, 64, 64)

            x    = torch.cat([embedding, mask_small, sdf], dim=1)   # (B, 258, 64, 64)
            pred = model(x)                                          # (B, 1, 64, 64)

            pred_up = F.interpolate(
                pred, size=(GT_SIZE, GT_SIZE),
                mode="bilinear", align_corners=False,
            )                                                # (B, 1, 448, 448)

            loss = boundary_dice_bce_loss(
                pred_up, gt_mask, w_boundary, boundary_radius, dice_weight
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_expert(
    expert_id: str,
    train_loader,
    val_loader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    model     = CNNExpert(c_in=258).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    wb_active     = wandb_init(args, expert_id)
    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, args.epochs + 1):
        tag = f"[{expert_id}] {epoch}/{args.epochs}"

        train_loss = run_one_epoch(
            model, train_loader, device, optimizer,
            args.w_boundary, args.boundary_radius, args.dice_weight,
            desc=f"{tag} train",
        )
        val_loss = run_one_epoch(
            model, val_loader, device, None,
            args.w_boundary, args.boundary_radius, args.dice_weight,
            desc=f"{tag} val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"{tag}  train={train_loss:.4f}  val={val_loss:.4f}")

        wandb_log(
            {f"{expert_id}/train_loss": train_loss, f"{expert_id}/val_loss": val_loss},
            step=epoch,
            active=wb_active,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = args.checkpoint_dir / f"{expert_id}_best.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, ckpt)
            print(f"  -> best checkpoint: {ckpt}")

    final = args.checkpoint_dir / f"{expert_id}_final.pth"
    torch.save({"epoch": args.epochs, "model_state": model.state_dict()}, final)
    print(f"[{expert_id}] done. Final: {final}")

    save_loss_curve(train_losses, val_losses, expert_id, args.checkpoint_dir)
    wandb_finish(wb_active)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--expert", choices=["expert_small", "expert_large", "both"], default="both")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--num-workers",     type=int,   default=4)
    parser.add_argument("--w-boundary",      type=float, default=3.0)
    parser.add_argument("--boundary-radius", type=int,   default=3)
    parser.add_argument("--dice-weight",     type=float, default=0.5)
    parser.add_argument("--large-subsample", type=int,   default=None,
                        help="Subsample expert_large to this many fragments (check the actual "
                             "small/large counts in gated_ramh1200_train_records.csv first).")
    parser.add_argument("--checkpoint-dir",  type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "cnnNoROI_ramh1200")
    # W&B
    parser.add_argument("--wandb-project", type=str, default="moe-shaprefine",
                        help="W&B project name.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging. Loss curve PNG is always saved locally.")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    experts = ["expert_small", "expert_large"] if args.expert == "both" else [args.expert]

    print("Building dataloaders …")
    train_loaders = build_expert_dataloaders(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        large_subsample=args.large_subsample,
    )
    val_loaders = build_expert_dataloaders(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for expert_id in experts:
        print(f"\n{'='*60}\nTraining {expert_id}\n{'='*60}")
        train_expert(
            expert_id=expert_id,
            train_loader=train_loaders[expert_id],
            val_loader=val_loaders[expert_id],
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
