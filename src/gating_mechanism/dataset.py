import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataloader_utils import load_pengwin_label, decode_pengwin_fragment_from_record, resolve_existing_path, load_prediction_masks
from paths import TOPLEVEL_DATA_ROOT

CSV_DIR = Path(__file__).parent
GT_DIR  = TOPLEVEL_DATA_ROOT / "pengwin" / "original" / "task2_xray" / "train" / "output" / "images" / "x-ray"


class FragmentDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        embedding   = np.load(resolve_existing_path(rec["embedding_path"]))        # (256, 64, 64)
        masks       = load_prediction_masks(rec["binary_masks_path"])                      # (N, 1024, 1024)
        binary_mask = masks[rec["medsam_instance_id"] - 1]                         # (1024, 1024)

        gt_path = GT_DIR / f"{rec['sample_name'].replace('XRAY_PENGWIN_', '')}.tif"
        gt      = load_pengwin_label(gt_path)                                      # (H, W) uint32
        gt_mask = decode_pengwin_fragment_from_record(gt, rec)                     # (H, W) uint8

        return {
            "embedding":   torch.from_numpy(embedding).float(),                    # (256, 64, 64)
            "binary_mask": torch.from_numpy(binary_mask).unsqueeze(0).float(),     # (1, 1024, 1024)
            "gt_mask":     torch.from_numpy(gt_mask).unsqueeze(0).float(),         # (1, 448, 448)
        }


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    """Subsample n records from df preserving the SA/LI/RI class distribution."""
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
    csv_dir: Path | None = None,
):
    """
    Load gated_{split}_records.csv and return one DataLoader per expert.
    Run gating_mechanism.py first if the CSV does not exist.

    Args:
        large_subsample: if set, subsample expert_large to this many fragments
                         while preserving SA/LI/RI class proportions.
        csv_dir: directory containing gated_{split}_records.csv. Defaults to
                 CSV_DIR (this file's directory) — the canonical gating output.
                 Pass a different directory to use an alternate gating run
                 (e.g. a different area threshold or random routing) without
                 touching the canonical CSVs.
    """
    csv_dir  = csv_dir or CSV_DIR
    csv_path = csv_dir / f"gated_{split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python gating_mechanism.py` first."
        )

    df      = pd.read_csv(csv_path)
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
            shuffle=shuffle and len(records) > 0,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


def build_moe_union_dataloader(
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    large_subsample: int | None = None,
    csv_dir: Path | None = None,
    seed: int = 42,
    record_list_out: Path | None = None,
):
    """Load gated_{split}_records.csv and return ONE DataLoader over the union
    of expert_small + expert_large fragments — for jointly-trained soft-gate
    MoE, where both experts (and the gate) must see every fragment rather than
    a pre-routed subset.

    The pre-computed "expert" column (area-threshold routing) is used ONLY to
    reproduce the existing expert_large subsampling for training-budget parity
    with the rule-based-gate baseline — it does NOT restrict which fragments a
    given expert or the gate sees at train time.

    Args:
        large_subsample: if set, subsample expert_large to this many fragments
            (stratified by SA/LI/RI) before unioning with expert_small, same
            as build_expert_dataloaders. Recommended for a first run so the
            joint MoE trains on the same fragment budget as the paper's
            rule-based gate.
        record_list_out: if set, write the exact fragment list used (case_id,
            sample_name, medsam_instance_id, fragment_id, area) to this CSV —
            for provenance / to confirm the gate saw an unbiased union.
    """
    csv_dir  = csv_dir or CSV_DIR
    csv_path = csv_dir / f"gated_{split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python gating_mechanism.py` first."
        )

    df      = pd.read_csv(csv_path)
    shuffle = split == "train"

    small_df = df[df.expert == "expert_small"]
    large_df = df[df.expert == "expert_large"]

    if large_subsample is not None:
        n = min(large_subsample, len(large_df))
        large_records = stratified_subsample(large_df, n=n, seed=seed)
    else:
        large_records = large_df.to_dict("records")

    records = small_df.to_dict("records") + large_records
    print(f"[{split}] moe union: {len(small_df)} small + {len(large_records)} large "
          f"= {len(records)} fragments (both experts + gate see all of these)")

    if record_list_out is not None:
        record_list_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records)[
            ["case_id", "sample_name", "medsam_instance_id", "fragment_id", "area"]
        ].to_csv(record_list_out, index=False)
        print(f"[{split}] fragment list saved: {record_list_out}")

    return DataLoader(
        FragmentDataset(records),
        batch_size=batch_size,
        shuffle=shuffle and len(records) > 0,
        num_workers=num_workers,
        pin_memory=True,
    )
