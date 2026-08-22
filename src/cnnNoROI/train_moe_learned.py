#!/usr/bin/env python3
"""Train a LEARNED soft-gate MoE over two cnnNoROI experts (PLAN_C_TRAIN).

Unlike train_moe.py (which trains expert_small/expert_large independently on
a fixed area-threshold split), this script trains a small gate network + two
fresh CNNExpert instances JOINTLY on one segmentation loss. Both experts see
every fragment; the gate learns a per-fragment soft mixture weight. Small/
large specialisation is allowed to emerge rather than being imposed by a
rule. See PLAN_C_TRAIN.md for the full design rationale.

Pipeline per batch (identical up through expert input to train_moe.py):
    1. Resize binary_mask (1, 1024, 1024) → (1, 64, 64)   [nearest]
    2. Compute SDF on the 64×64 mask                       [scipy, CPU]
    3. Concatenate: embedding (256) + mask (1) + SDF (1) → (258, 64, 64)
    4. Forward through GatedMoE (gate + expert_0 + expert_1) → blended (1, 64, 64)
    5. Upsample blended prediction → (1, 448, 448)          [bilinear]
    6. L = L_seg(blend, gt) + lambda_balance * L_balance(gate_weights)

Usage:
    python train_moe_learned.py --dry-run
    python train_moe_learned.py --epochs 40 --patience 10 --batch-size 64 \
        --lambda-balance 0.1 --gate-noise 0.3 --keep-large-subsample true
"""
from __future__ import annotations

import argparse
import json
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

from sdf_utils import sdf_channel_from_mask                    # noqa: E402
from cnnNoROI.cnnMoE import GatedMoE                            # noqa: E402
from cnnNoROI.losses import boundary_dice_bce_loss, load_balance_loss  # noqa: E402
from dataset import build_moe_union_dataloader                  # noqa: E402
# --------------------------------------------------------------------------

FEAT_SIZE = 64
GT_SIZE   = 448


# ---------------------------------------------------------------------------
# W&B helpers — isolated here so the rest of the code works fine with
# --no-wandb. Logs per-epoch loss/seg/balance AND per-expert gate mass, so
# collapse (one expert's usage -> 0) is visible on a dashboard, not just in
# stdout/train_log.csv.
# ---------------------------------------------------------------------------

def wandb_init(args: argparse.Namespace) -> bool:
    if args.no_wandb:
        return False
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"moe-learned-gate__{args.wandb_project}",
            config=vars(args),
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


def compute_batch_sdf(mask_small: torch.Tensor) -> torch.Tensor:
    """Compute SDF for a batch of 64×64 masks on CPU, return (B, 1, H, W) float32."""
    sdfs = [
        sdf_channel_from_mask(mask_small[i, 0].cpu().numpy())
        for i in range(mask_small.size(0))
    ]
    return torch.from_numpy(np.stack(sdfs))


def run_one_epoch(
    model: GatedMoE,
    loader,
    device: torch.device,
    optimizer: AdamW | None,
    w_boundary: float,
    boundary_radius: int,
    dice_weight: float,
    lambda_balance: float,
    desc: str,
) -> dict:
    """One training or validation epoch. Returns mean metrics dict."""
    training = optimizer is not None
    model.train(training)

    total_loss    = 0.0
    total_seg     = 0.0
    total_balance = 0.0
    gate_mass_sum = torch.zeros(2, dtype=torch.float64)
    n_samples     = 0

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc=desc, leave=False):
            embedding   = batch["embedding"].to(device)      # (B, 256, 64, 64)
            binary_mask = batch["binary_mask"].to(device)    # (B, 1, 1024, 1024)
            gt_mask     = batch["gt_mask"].to(device)        # (B, 1, 448, 448)

            mask_small = F.interpolate(
                binary_mask, size=(FEAT_SIZE, FEAT_SIZE), mode="nearest"
            )                                                # (B, 1, 64, 64)
            sdf = compute_batch_sdf(mask_small).to(device)   # (B, 1, 64, 64)

            x = torch.cat([embedding, mask_small, sdf], dim=1)   # (B, 258, 64, 64)
            y_hat, g = model(x)                                  # (B, 1, 64, 64), (B, 2)

            pred_up = F.interpolate(
                y_hat, size=(GT_SIZE, GT_SIZE), mode="bilinear", align_corners=False,
            )                                                # (B, 1, 448, 448)

            l_seg     = boundary_dice_bce_loss(pred_up, gt_mask, w_boundary, boundary_radius, dice_weight)
            l_balance = load_balance_loss(g)
            loss      = l_seg + lambda_balance * l_balance

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = embedding.size(0)
            total_loss    += loss.item() * bs
            total_seg     += l_seg.item() * bs
            total_balance += l_balance.item() * bs
            gate_mass_sum += g.detach().sum(dim=0).double().cpu()
            n_samples     += bs

    n_samples = max(n_samples, 1)
    gate_mass_mean = (gate_mass_sum / n_samples).tolist()
    return {
        "loss":    total_loss / n_samples,
        "seg":     total_seg / n_samples,
        "balance": total_balance / n_samples,
        "gate_0":  gate_mass_mean[0],
        "gate_1":  gate_mass_mean[1],
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    large_subsample = args.large_subsample if args.keep_large_subsample else None
    if args.dry_run:
        large_subsample = min(large_subsample or 200, 200)

    print("Building dataloaders (union of expert_small + expert_large — no routing applied)…")
    train_loader = build_moe_union_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        large_subsample=large_subsample,
        csv_dir=args.data_root,
        seed=args.seed,
        record_list_out=args.out_dir / "moe_train_fragments.csv",
    )
    val_loader = build_moe_union_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        large_subsample=None,
        csv_dir=args.data_root,
        seed=args.seed,
        record_list_out=args.out_dir / "moe_val_fragments.csv",
    )

    if args.dry_run:
        from torch.utils.data import DataLoader, Subset
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(min(len(train_loader.dataset), args.batch_size * 2))),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(min(len(val_loader.dataset), args.batch_size * 2))),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

    model = GatedMoE(c_in=258, gate_hidden=64, gate_noise_eps=args.gate_noise).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    wb_active = wandb_init(args)
    epochs = 2 if args.dry_run else args.epochs
    best_val_seg = float("inf")
    epochs_no_improve = 0

    log_path = args.out_dir / "train_log.csv"
    log_path.write_text(
        "epoch,train_loss,train_seg,train_balance,train_gate0,train_gate1,"
        "val_loss,val_seg,val_balance,val_gate0,val_gate1\n"
    )

    for epoch in range(1, epochs + 1):
        tag = f"[moe-learned] {epoch}/{epochs}"

        tr = run_one_epoch(
            model, train_loader, device, optimizer,
            args.w_boundary, args.boundary_radius, args.dice_weight, args.lambda_balance,
            desc=f"{tag} train",
        )
        va = run_one_epoch(
            model, val_loader, device, None,
            args.w_boundary, args.boundary_radius, args.dice_weight, args.lambda_balance,
            desc=f"{tag} val",
        )

        print(
            f"{tag}  train: loss={tr['loss']:.4f} seg={tr['seg']:.4f} bal={tr['balance']:.4f} "
            f"gate=[{tr['gate_0']:.3f},{tr['gate_1']:.3f}]  |  "
            f"val: loss={va['loss']:.4f} seg={va['seg']:.4f} bal={va['balance']:.4f} "
            f"gate=[{va['gate_0']:.3f},{va['gate_1']:.3f}]"
        )

        # Collapse warnings — see PLAN_C_TRAIN.md §6.
        min_gate = min(va["gate_0"], va["gate_1"])
        if min_gate < 0.15:
            print(f"  WARNING: expert gate mass imbalanced (min={min_gate:.3f}) — "
                  f"possible collapse. Consider raising --lambda-balance or --gate-noise.")

        with log_path.open("a") as f:
            f.write(
                f"{epoch},{tr['loss']:.6f},{tr['seg']:.6f},{tr['balance']:.6f},"
                f"{tr['gate_0']:.6f},{tr['gate_1']:.6f},"
                f"{va['loss']:.6f},{va['seg']:.6f},{va['balance']:.6f},"
                f"{va['gate_0']:.6f},{va['gate_1']:.6f}\n"
            )

        wandb_log(
            {
                "train/loss": tr["loss"], "train/seg": tr["seg"], "train/balance": tr["balance"],
                "train/gate_mass_expert_0": tr["gate_0"], "train/gate_mass_expert_1": tr["gate_1"],
                "val/loss": va["loss"], "val/seg": va["seg"], "val/balance": va["balance"],
                "val/gate_mass_expert_0": va["gate_0"], "val/gate_mass_expert_1": va["gate_1"],
            },
            step=epoch,
            active=wb_active,
        )

        if va["seg"] < best_val_seg:
            best_val_seg = va["seg"]
            epochs_no_improve = 0
            ckpt = args.out_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "gate_state":     model.gate.state_dict(),
                "expert_0_state": model.expert_0.state_dict(),
                "expert_1_state": model.expert_1.state_dict(),
                "val_seg_loss":   va["seg"],
                "val_gate_mass":  [va["gate_0"], va["gate_1"]],
                "config":         vars(args),
            }, ckpt)
            print(f"  -> best checkpoint (val_seg={best_val_seg:.4f}): {ckpt}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[moe-learned] early stopping at epoch {epoch} "
                      f"(no val_seg improvement for {args.patience} epochs, "
                      f"best_val_seg={best_val_seg:.4f})")
                break

    wandb_finish(wb_active)

    readme = args.out_dir / "README.md"
    readme.write_text(
        "# Learned soft-gate MoE checkpoint (best.pth)\n\n"
        "Keys:\n"
        "- `gate_state`: state_dict for cnnMoE.Gate(c_in=258, hidden=64, n_experts=2)\n"
        "- `expert_0_state`, `expert_1_state`: state_dicts for cnnMoE.CNNExpert(c_in=258)\n"
        "- `val_seg_loss`: best validation segmentation loss (L_DSC/BCE blend, no balance term)\n"
        "- `val_gate_mass`: [mean gate weight expert_0, mean gate weight expert_1] on val at that epoch\n"
        "- `config`: full run config (see also config.json in this directory)\n\n"
        "Load with:\n"
        "```python\n"
        "from cnnNoROI.cnnMoE import GatedMoE\n"
        "ckpt = torch.load('best.pth', map_location='cpu')\n"
        "model = GatedMoE(c_in=258, gate_hidden=64, gate_noise_eps=0.0)\n"
        "model.gate.load_state_dict(ckpt['gate_state'])\n"
        "model.expert_0.load_state_dict(ckpt['expert_0_state'])\n"
        "model.expert_1.load_state_dict(ckpt['expert_1_state'])\n"
        "model.eval()  # disables noisy gating\n"
        "```\n\n"
        "`moe_train_fragments.csv` / `moe_val_fragments.csv` in this directory list the exact "
        "fragments (case_id, sample_name, medsam_instance_id, fragment_id, area) both experts "
        "and the gate were trained/validated on — the union of expert_small + expert_large, "
        "with expert_large stratified-subsampled per `--keep-large-subsample` for budget parity "
        "with the rule-based-gate baseline. No area-threshold routing is applied to this data.\n"
    )
    print(f"[moe-learned] done. Best val_seg={best_val_seg:.4f}. Checkpoint dir: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=GATING_DIR,
                        help="Directory containing gated_{split}_records.csv (default: gating_mechanism/).")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "cnnNoROI_moe_learned")
    parser.add_argument("--epochs",          type=int,   default=40)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--num-workers",     type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--wd",              type=float, default=1e-4)
    parser.add_argument("--w-boundary",      type=float, default=3.0)
    parser.add_argument("--boundary-radius", type=int,   default=3)
    parser.add_argument("--dice-weight",     type=float, default=0.5)
    parser.add_argument("--lambda-balance",  type=float, default=0.1,
                        help="Weight on the Shazeer load-balancing loss (raise to 0.5-1.0 if an expert collapses).")
    parser.add_argument("--gate-noise",      type=float, default=0.3,
                        help="Fixed-eps Gaussian noise added to gate logits at train time only.")
    parser.add_argument("--large-subsample", type=int,   default=17000,
                        help="Stratified subsample size for expert_large fragments before unioning "
                             "with expert_small, matching train_moe.py's default training budget.")
    parser.add_argument("--keep-large-subsample", type=lambda s: s.lower() != "false", default=True,
                        help="true (default): apply --large-subsample so the joint MoE trains on the "
                             "same fragment budget as the paper's rule-based gate. false: use all "
                             "expert_large fragments unsubsampled.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="1-2 epochs on a small subset to sanity-check the loop before a full run.")
    # W&B
    parser.add_argument("--wandb-project", type=str, default="moe-shaprefine",
                        help="W&B project name.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging. train_log.csv is always saved locally.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
