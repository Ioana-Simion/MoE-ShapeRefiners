#!/usr/bin/env python3
"""RAM-H1200 fragment dataset + gated-CSV builder for expert training.

Mirrors `dataset.py` (PENGWIN) with the same three names (FragmentDataset,
stratified_subsample, build_expert_dataloaders) so train_flowsdf_ramh1200.py
and train_moe_ramh1200.py can be near-verbatim forks of their PENGWIN
counterparts -- only the dataset import line changes.

The one real difference from PENGWIN's dataset.py: GT here comes from
RAM-H1200's own per-fragment .npz masks (gt_masks_path, already produced by
prepare_ramh1200_boxes.py / prepare_ramh1200_training_data.py), not PENGWIN's
bit-packed .tif labels. GT masks are resized to 448x448 (nearest-neighbor,
binary-preserving) to match the fixed resolution PENGWIN's FragmentDataset
returns, since train_moe.py uses gt_mask at that resolution directly with no
further resizing of its own.

Usage (building the gated CSVs -- run once per split before training):
    python ramh1200_dataset.py --split train --threshold 3433
    python ramh1200_dataset.py --split val   --threshold 3433
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader_utils import load_prediction_masks, resize_binary_nearest, resolve_existing_path  # noqa: E402

CSV_DIR = Path(__file__).parent
GT_SIZE = 448  # matches PENGWIN's fixed GT resolution -- see module docstring


class FragmentDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        embedding = np.load(resolve_existing_path(rec["embedding_path"]))  # (256, 64, 64)
        masks = load_prediction_masks(rec["binary_masks_path"])  # (N, 1024, 1024)
        binary_mask = masks[rec["medsam_instance_id"] - 1]  # (1024, 1024)

        gt_masks = np.load(resolve_existing_path(rec["gt_masks_path"]))["masks"]  # (N, H, W) native res
        gt_mask_native = gt_masks[rec["medsam_instance_id"] - 1]
        gt_mask = resize_binary_nearest(gt_mask_native, (GT_SIZE, GT_SIZE))  # (448, 448)

        return {
            "embedding": torch.from_numpy(embedding).float(),  # (256, 64, 64)
            "binary_mask": torch.from_numpy(binary_mask).unsqueeze(0).float(),  # (1, 1024, 1024)
            "gt_mask": torch.from_numpy(gt_mask).unsqueeze(0).float(),  # (1, 448, 448)
        }


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    """Subsample n records from df preserving the bone-category distribution."""
    return (
        df.groupby("category_name", group_keys=False)
        .apply(lambda x: x.sample(frac=n / len(df), random_state=seed))
        .reset_index(drop=True)
        .to_dict("records")
    )


def build_expert_dataloaders(
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    large_subsample: int | None = None,
):
    """
    Load gated_ramh1200_{split}_records.csv and return one DataLoader per expert.
    Run this file's own main() (or run_ramh1200_flowsdf.py's gating) first if
    the CSV does not exist.

    Args:
        large_subsample: if set, subsample expert_large to this many fragments
                         while preserving bone-category proportions. Mirrors
                         PENGWIN's actual policy (194591 large vs 17335 small
                         fragments, trained with --large-subsample 17000 --
                         downsample the majority to roughly match the minority,
                         not oversample / reweight).
    """
    csv_path = CSV_DIR / f"gated_ramh1200_{split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python ramh1200_dataset.py --split {split} --threshold <px>` first."
        )

    df = pd.read_csv(csv_path)
    shuffle = split == "train"

    loaders = {}
    for expert in ["expert_small", "expert_large"]:
        subset = df[df.expert == expert]

        if expert == "expert_large" and large_subsample is not None:
            n = min(large_subsample, len(subset))
            records = stratified_subsample(subset, n=n)
        else:
            records = subset.to_dict("records")

        print(f"[{split}] {expert}: {len(records)} fragments")
        loaders[expert] = DataLoader(
            FragmentDataset(records),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


# ---------------------------------------------------------------------------
# Gated-CSV builder -- routes fragments by area threshold, mirrors
# run_ramh1200_flowsdf.py::compute_gating but at an explicit chosen threshold
# (from ramh1200_threshold_analysis.ipynb) rather than the RAM-H1200-relative
# median, and includes gt_masks_path per row (needed for training, not eval).
# ---------------------------------------------------------------------------
def build_gated_csv(medsam_pred_root: Path, split: str, threshold: float, output_csv: Path) -> pd.DataFrame:
    metadata_path = medsam_pred_root / split / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run prepare_ramh1200_training_data.py for --splits {split} first."
        )

    records = [json.loads(l) for l in metadata_path.open()]
    rows = []
    for record in records:
        masks = load_prediction_masks(record["binary_masks_path"])
        for frag, mask in zip(record["fragments"], masks):
            area = int(mask.sum())
            rows.append({
                "sample_name": record["sample_name"],
                "medsam_instance_id": frag["medsam_instance_id"],
                "category_name": frag["category_name"],
                "area": area,
                "expert": "expert_small" if area <= threshold else "expert_large",
                "embedding_path": record["embedding_path"],
                "binary_masks_path": record["binary_masks_path"],
                "gt_masks_path": record["gt_masks_path"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[{split}] threshold={threshold:.1f}px, saved {len(df)} fragments -> {output_csv}")
    print(df["expert"].value_counts().to_string())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, required=True, help="Area threshold (px) from ramh1200_threshold_analysis.ipynb, run on the SAME split.")
    parser.add_argument("--medsam-pred-root", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "ramh1200" / "medsam-predictions")
    parser.add_argument("--output-csv", type=Path, default=None, help="Defaults to gated_ramh1200_<split>_records.csv in this directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv or (CSV_DIR / f"gated_ramh1200_{args.split}_records.csv")
    build_gated_csv(args.medsam_pred_root, args.split, args.threshold, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
