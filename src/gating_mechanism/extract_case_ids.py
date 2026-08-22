#!/usr/bin/env python3
"""Extract the case_ids referenced by gated_{split}_records.csv.

Purpose: when MedSAM embeddings/masks must be regenerated on a new cluster
(rather than copying the derived .npy/.npz files across clusters), this
gives `prepare_pengwin_xray_boxes_for_medsam_inference.py` and
`run_medsam_with_pengwin_boxes.py` an exact list of which PENGWIN cases to
process, so the regenerated data covers precisely the same fragments the
existing gated_train/val/test_records.csv were built from — no drift in
the train/val/test split or the small/large fragment mix.

Output: one case_id per line, e.g. `001_0000`, deduplicated. By default
writes one combined file (train ∪ val ∪ test) since the downstream MedSAM
scripts run per-case, not per-split, and process each case's full image
once regardless of which split(s) reference it. Pass --per-split to also
write train/val/test files separately for inspection.

Usage:
    python extract_case_ids.py
    python extract_case_ids.py --csv-dir /path/to/gating_mechanism --out case_ids.txt
    python extract_case_ids.py --per-split
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CSV_DIR = Path(__file__).parent
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR,
                        help="Directory containing gated_{split}_records.csv (default: this file's directory).")
    parser.add_argument("--out", type=Path, default=CSV_DIR / "case_ids_all_splits.txt",
                        help="Output path for the combined (train+val+test) case_id list.")
    parser.add_argument("--per-split", action="store_true",
                        help="Also write case_ids_{split}.txt next to --out, one per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_case_ids: set[str] = set()
    per_split_ids: dict[str, set[str]] = {}

    for split in SPLITS:
        csv_path = args.csv_dir / f"gated_{split}_records.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found. Run gating_mechanism.py first.")
        df = pd.read_csv(csv_path)
        ids = set(df["case_id"].astype(str).unique())
        per_split_ids[split] = ids
        all_case_ids |= ids
        print(f"[{split}] {len(ids)} unique case_ids ({len(df)} fragments)")

    overlap = per_split_ids["train"] & per_split_ids["val"] & per_split_ids["test"]
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} case_ids appear in more than one split — "
            f"train/val/test are supposed to be disjoint by case_id. Example: {sorted(overlap)[:5]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(all_case_ids)) + "\n")
    print(f"\nwrote {len(all_case_ids)} unique case_ids (all splits) -> {args.out}")

    if args.per_split:
        for split, ids in per_split_ids.items():
            out_path = args.out.parent / f"case_ids_{split}.txt"
            out_path.write_text("\n".join(sorted(ids)) + "\n")
            print(f"wrote {len(ids)} case_ids [{split}] -> {out_path}")


if __name__ == "__main__":
    main()
