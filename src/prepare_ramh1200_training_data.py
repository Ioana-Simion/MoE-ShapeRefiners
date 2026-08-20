#!/usr/bin/env python3
"""Prepare RAM-H1200 train/val data for expert training.

Thin orchestration script: for each of --splits (default train, val), runs
the same adapter + MedSAM Stage-1 pipeline already used for the test split
(prepare_ramh1200_boxes.process_split + run_ramh1200_flowsdf.run_medsam_shard),
writing to the same directory layout:

    data/ramh1200/bounding-boxes/<split>/       boxes, gt_masks, metadata.jsonl
    data/ramh1200/medsam-predictions/<split>/   binary_masks, embeddings, metadata.jsonl

No new logic here -- this only exists because those two functions were only
ever invoked for --split test before. See ramh1200_threshold_analysis.ipynb
(re-run with SPLIT = "train" once this has populated medsam-predictions/train/)
and src/gating_mechanism/ramh1200_dataset.py for the next steps.

Usage:
    python src/prepare_ramh1200_training_data.py --limit 10   # smoke test
    python src/prepare_ramh1200_training_data.py              # full train+val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MEDSAM_REPO_ROOT = WORKSPACE_ROOT / "MedSAM"

for _p in (REPO_ROOT, REPO_ROOT / "src", MEDSAM_REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prepare_ramh1200_boxes import process_split  # noqa: E402
from run_ramh1200_flowsdf import run_medsam_shard, pick_devices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "RAM-H1200-v1_dataset" / "Segmentation")
    parser.add_argument("--boxes-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "bounding-boxes")
    parser.add_argument("--medsam-pred-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "medsam-predictions")
    parser.add_argument("--medsam-checkpoint", type=Path, default=MEDSAM_REPO_ROOT / "medsam_vit_b.pth")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images per split (smoke test).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip MedSAM outputs that already exist on disk.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = pick_devices(1)[0]
    print(f"device: {device}")

    for split in args.splits:
        print(f"\n=== {split}: adapter (COCO -> boxes/gt_masks/metadata) ===")
        process_split(
            dataset_root=args.dataset_root,
            output_root=args.boxes_root,
            split=split,
            image_size=1024,
            limit=args.limit,
        )
        records = [json.loads(l) for l in (args.boxes_root / split / "metadata.jsonl").open()]
        print(f"{split}: {len(records)} images")

        print(f"\n=== {split}: MedSAM Stage-1 ===")
        medsam_pred_records = run_medsam_shard(
            records, device, args.medsam_checkpoint, args.medsam_pred_root, split, args.skip_existing
        )
        pred_metadata_path = args.medsam_pred_root / split / "metadata.jsonl"
        with pred_metadata_path.open("w") as f:
            for r in medsam_pred_records:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: wrote {len(medsam_pred_records)} MedSAM prediction records -> {pred_metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
