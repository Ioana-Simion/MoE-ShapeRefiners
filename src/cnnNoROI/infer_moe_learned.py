#!/usr/bin/env python3
"""Inference for the learned soft-gate MoE (PLAN_C_EVAL §1).

Loads a GatedMoE checkpoint (gate + expert_0 + expert_1) trained by
train_moe_learned.py and, for every fragment, computes the gate ONCE and both
experts ONCE, then derives BOTH inference modes from that single pass:

    hard (PRIMARY): y = expert_k(x), k = argmax(gate(x))       -- one expert's
                     raw output, untouched by the other expert. This is what
                     gets compared against the rule-based/random baselines
                     (same one-expert-per-fragment structure); computing both
                     experts here anyway (needed for soft) doesn't change what
                     the hard MASK actually is.
    soft (SECONDARY): y = g0*expert_0(x) + g1*expert_1(x)       -- blended.

Outputs (per --output-root):
    hard/binary_masks/{sample_name}.npz   -- key "masks", same convention as
    soft/binary_masks/{sample_name}.npz      MedSAM/rule-based cnnNoROI, so
                                              evaluate_medsam_pengwin.py reads
                                              either one unchanged.
    gate_weights.csv -- one row per fragment: sample_name, case_id,
        medsam_instance_id, fragment_index, fragment_id, category_name,
        medsam_area (the pre-computed foreground-pixel "area" column the
        original rule-based routing used), g0, g1, argmax_expert.
        gt_area/dice are NOT here -- join against an evaluate_medsam_pengwin.py
        output CSV on (sample_name, fragment_index) to add those, see
        merge_gate_analysis.py.

Usage:
    python infer_moe_learned.py --checkpoint checkpoints/cnnNoROI_moe_learned/run_X/best.pth
    python infer_moe_learned.py --checkpoint ... --split test --limit 10   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- path setup -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR      = PROJECT_ROOT / "src"
GATING_DIR   = SRC_DIR / "gating_mechanism"

for _p in (str(SRC_DIR), str(GATING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloader_utils import load_prediction_masks, resolve_existing_path   # noqa: E402
from sdf_utils import sdf_channel_from_mask                                  # noqa: E402
from cnnNoROI.cnnMoE import GatedMoE                                         # noqa: E402
# --------------------------------------------------------------------------

FEAT_SIZE   = 64
OUTPUT_SIZE = 1024


def load_gated_moe(checkpoint_path: Path, device: torch.device) -> GatedMoE:
    model = GatedMoE(c_in=258, gate_hidden=64, gate_noise_eps=0.0).to(device)
    # weights_only=False: this checkpoint's "config" dict (see train_moe_learned.py)
    # contains pathlib.Path objects from argparse (e.g. --out-dir), which aren't on
    # torch's weights_only=True safe-globals allowlist (PyTorch >=2.6 default).
    # Safe here since this is our own checkpoint produced by train_moe_learned.py,
    # not a downloaded/untrusted file.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.gate.load_state_dict(ckpt["gate_state"])
    model.expert_0.load_state_dict(ckpt["expert_0_state"])
    model.expert_1.load_state_dict(ckpt["expert_1_state"])
    model.eval()
    print(
        f"Loaded {checkpoint_path} (epoch={ckpt.get('epoch')}, "
        f"checkpoint_metric={ckpt.get('checkpoint_metric')}, "
        f"score={ckpt.get('checkpoint_score')})"
    )
    return model


@torch.no_grad()
def refine_fragment_learned(
    embedding: torch.Tensor,   # (256, 64, 64), on device, float32
    binary_mask: np.ndarray,   # (1024, 1024) uint8/float
    model: GatedMoE,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Returns (hard_mask, soft_mask, g0, g1, argmax_expert) for one fragment."""
    mask_t = (
        torch.from_numpy(binary_mask.astype(np.float32))
        .unsqueeze(0).unsqueeze(0)
        .to(device)
    )                                                              # (1, 1, 1024, 1024)

    mask_small = F.interpolate(
        mask_t, size=(FEAT_SIZE, FEAT_SIZE), mode="nearest"
    )                                                              # (1, 1, 64, 64)

    sdf_np = sdf_channel_from_mask(mask_small[0, 0].cpu().numpy())  # (1, 64, 64)
    sdf    = torch.from_numpy(sdf_np).unsqueeze(0).to(device)        # (1, 1, 64, 64)

    x = torch.cat([embedding.unsqueeze(0), mask_small, sdf], dim=1)  # (1, 258, 64, 64)

    logits = model.gate(x)                    # (1, 2) -- noise-free regardless
    g = torch.softmax(logits, dim=1)          # of model.eval(); Gate has no dropout/noise itself.
    y0 = model.expert_0(x)                    # (1, 1, 64, 64)
    y1 = model.expert_1(x)                    # (1, 1, 64, 64)

    k = int(g.argmax(dim=1).item())
    y_hard = y0 if k == 0 else y1
    g0f, g1f = float(g[0, 0]), float(g[0, 1])
    y_soft = g0f * y0 + g1f * y1

    def _upsample_threshold(y: torch.Tensor) -> np.ndarray:
        y_up = F.interpolate(
            y, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False,
        )
        return (y_up[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    hard_mask = _upsample_threshold(y_hard)
    soft_mask = _upsample_threshold(y_soft)

    return hard_mask, soft_mask, g0f, g1f, k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best.pth saved by train_moe_learned.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "data" / "moe-learned-predictions")
    parser.add_argument(
        "--gating-csv-dir", type=Path, default=None,
        help="Directory containing gated_{split}_records.csv. Defaults to src/gating_mechanism/.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (smoke test).")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}\nRun train_moe_learned.py first.")
    model = load_gated_moe(args.checkpoint, device)

    gating_csv_dir = args.gating_csv_dir or GATING_DIR
    csv_path = gating_csv_dir / f"gated_{args.split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Gated CSV not found: {csv_path}")
    print(f"gating csv dir: {gating_csv_dir}")

    df = pd.read_csv(csv_path)

    hard_dir = args.output_root / "hard" / "binary_masks"
    soft_dir = args.output_root / "soft" / "binary_masks"
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    gate_weights_path = args.output_root / f"gate_weights_{args.split}.csv"
    gate_fields = [
        "sample_name", "case_id", "medsam_instance_id", "fragment_index", "fragment_id",
        "category_name", "medsam_area", "g0", "g1", "argmax_expert",
    ]

    sample_names = df["sample_name"].unique()
    if args.limit is not None:
        sample_names = sample_names[: args.limit]

    print(f"Running learned-MoE inference (hard + soft) on {len(sample_names)} images …")

    with gate_weights_path.open("w", newline="") as gw_file:
        writer = csv.DictWriter(gw_file, fieldnames=gate_fields)
        writer.writeheader()

        for sample_name in tqdm(sample_names, desc="Inference"):
            rows = (
                df[df["sample_name"] == sample_name]
                .sort_values("medsam_instance_id")
                .reset_index(drop=True)
            )

            embedding_path = resolve_existing_path(rows.iloc[0]["embedding_path"])
            embedding = torch.from_numpy(np.load(embedding_path)).float().to(device)   # (256, 64, 64)

            masks_path = resolve_existing_path(rows.iloc[0]["binary_masks_path"])
            all_masks  = load_prediction_masks(masks_path)                              # (N, 1024, 1024)

            hard_refined: list[np.ndarray] = []
            soft_refined: list[np.ndarray] = []

            for _, row in rows.iterrows():
                instance_idx = int(row["medsam_instance_id"]) - 1   # 0-based
                binary_mask  = all_masks[instance_idx]

                hard_mask, soft_mask, g0, g1, k = refine_fragment_learned(
                    embedding, binary_mask, model, device,
                )
                hard_refined.append(hard_mask)
                soft_refined.append(soft_mask)

                writer.writerow({
                    "sample_name":        sample_name,
                    "case_id":            row.get("case_id"),
                    "medsam_instance_id": int(row["medsam_instance_id"]),
                    "fragment_index":     instance_idx,   # matches evaluate_medsam_pengwin.py's join key
                    "fragment_id":        row.get("fragment_id"),
                    "category_name":      row.get("category_name"),
                    "medsam_area":        row.get("area"),
                    "g0":                 g0,
                    "g1":                 g1,
                    "argmax_expert":      k,
                })

            np.savez_compressed(hard_dir / f"{sample_name}.npz", masks=np.stack(hard_refined))
            np.savez_compressed(soft_dir / f"{sample_name}.npz", masks=np.stack(soft_refined))

    print(f"\nDone.")
    print(f"  hard masks: {hard_dir}")
    print(f"  soft masks: {soft_dir}")
    print(f"  gate weights: {gate_weights_path}")


if __name__ == "__main__":
    main()
