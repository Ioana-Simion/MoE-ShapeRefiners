#!/usr/bin/env python3
"""Repoint gated_{split}_records.csv's embedding_path/binary_masks_path columns
at freshly-regenerated MedSAM predictions on a new cluster.

Use case: gated_{train,val,test}_records.csv were built on a different
cluster/account (e.g. /gpfs/home5/aramautar2/...) and copied over as-is.
resolve_existing_path() (dataloader_utils.py) only bridges /gpfs/home5 vs
/home path-prefix variants on the SAME account -- it cannot resolve paths
from a different account/host, so those two columns are stale here and must
be rewritten. Everything else in the CSVs (case_id, area, expert routing,
category, fragment_id, ...) is left untouched -- this does NOT re-route
fragments or change split membership, it only fixes two path columns by
joining on sample_name against the new metadata.jsonl.

Usage:
    python repath_gated_csvs.py \\
        --new-metadata data/medsam-predictions/metadata.jsonl \\
        --csv-dir src/gating_mechanism
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

CSV_DIR = Path(__file__).parent
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-metadata", type=Path, required=True,
                        help="Freshly-generated data/medsam-predictions/metadata.jsonl on THIS cluster.")
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR,
                        help="Directory containing gated_{split}_records.csv (default: this file's directory).")
    parser.add_argument("--backup-suffix", type=str, default=".prerepath.bak",
                        help="Suffix for the pre-rewrite backup copy (default: %(default)s). "
                             "Distinct from any existing .bak files already in --csv-dir.")
    return parser.parse_args()


def load_path_lookup(metadata_path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with metadata_path.open() as f:
        for line in f:
            record = json.loads(line)
            lookup[record["sample_name"]] = {
                "embedding_path": record["embedding_path"],
                "binary_masks_path": record["binary_masks_path"],
            }
    return lookup


def main() -> None:
    args = parse_args()
    if not args.new_metadata.exists():
        raise FileNotFoundError(f"{args.new_metadata} not found -- run run_medsam_with_pengwin_boxes.py first.")

    lookup = load_path_lookup(args.new_metadata)
    print(f"loaded {len(lookup)} sample_name -> path entries from {args.new_metadata}")

    for split in SPLITS:
        csv_path = args.csv_dir / f"gated_{split}_records.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")

        df = pd.read_csv(csv_path)
        missing = sorted(set(df["sample_name"].unique()) - set(lookup))
        if missing:
            raise KeyError(
                f"[{split}] {len(missing)} sample_names have no entry in {args.new_metadata} "
                f"-- did run_medsam_with_pengwin_boxes.py --case-ids-file cover every case_id "
                f"referenced by this CSV? Example missing: {missing[:5]}"
            )

        df["embedding_path"] = df["sample_name"].map(lambda s: lookup[s]["embedding_path"])
        df["binary_masks_path"] = df["sample_name"].map(lambda s: lookup[s]["binary_masks_path"])

        backup_path = csv_path.with_suffix(csv_path.suffix + args.backup_suffix)
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)
            print(f"[{split}] backed up original -> {backup_path}")
        else:
            print(f"[{split}] backup already exists, not overwriting: {backup_path}")

        df.to_csv(csv_path, index=False)
        print(f"[{split}] repathed {len(df)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
