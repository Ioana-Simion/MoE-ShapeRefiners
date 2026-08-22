#!/usr/bin/env python3
"""Evaluate MoE refined masks against PENGWIN GT.

Wraps evaluate_medsam_pengwin.py with MoE-specific defaults.
Optionally computes a delta CSV comparing MoE vs MedSAM per fragment.

Usage:
    # Full evaluation with boundary metrics
    python evaluate_moe_pengwin.py

    # Fast evaluation (Dice + IoU only)
    python evaluate_moe_pengwin.py --skip-boundary

    # Smoke test (10 images)
    python evaluate_moe_pengwin.py --skip-boundary --limit 10

    # Evaluate + compute delta vs MedSAM baseline
    python evaluate_moe_pengwin.py --delta

    # Evaluate a specific expert's predictions
    python evaluate_moe_pengwin.py --pred-mask-root data/moe-predictions/binary_masks
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT  = PROJECT_ROOT / "evaluation" / "evaluate_medsam_pengwin.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pred-mask-root", type=Path,
        default=PROJECT_ROOT / "data" / "moe-predictions" / "binary_masks",
        help="Directory with MoE refined .npz mask files.",
    )
    parser.add_argument(
        "--pred-metadata", type=Path,
        default=PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl",
        help="Prediction metadata JSONL (MedSAM metadata works for MoE too).",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=PROJECT_ROOT / "data" / "moe-predictions" / "evaluation_moe.csv",
    )
    parser.add_argument(
        "--medsam-csv", type=Path,
        default=PROJECT_ROOT / "data" / "medsam-predictions" / "evaluation_pengwin.csv",
        help="MedSAM evaluation CSV, used only with --delta.",
    )
    parser.add_argument(
        "--delta-csv", type=Path,
        default=PROJECT_ROOT / "data" / "moe-predictions" / "evaluation_delta.csv",
        help="Output path for per-fragment delta CSV (MoE − MedSAM).",
    )
    parser.add_argument("--skip-boundary", action="store_true",
                        help="Compute only Dice and IoU (much faster).")
    parser.add_argument(
        "--area-threshold", type=float, default=None,
        help="Passed through to evaluate_medsam_pengwin.py's --area-threshold. Pin this to the "
             "SAME value across every model being compared (see PLAN_C_EVAL §6) -- without it, "
             "evaluate_medsam_pengwin.py defaults to the median bbox area of whatever it evaluated, "
             "which drifts between runs/subsets and makes 'by size' rows non-comparable.",
    )
    parser.add_argument("--delta", action="store_true",
                        help="After evaluation, compute delta CSV vs --medsam-csv.")
    parser.add_argument("--delta-only", action="store_true",
                        help="Skip evaluation and only compute delta. Requires --output-csv to already exist.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate at most N images (smoke test).")
    return parser.parse_args()


def filter_metadata_to_existing(metadata_path: Path, pred_mask_root: Path) -> Path:
    """Write a temp JSONL containing only records that have a prediction file.

    The full metadata.jsonl covers all 50k images (train+val+test), but MoE
    inference is typically run on one split at a time. Passing the full file
    to evaluate_medsam_pengwin.py would crash on every missing prediction.
    This returns a NamedTemporaryFile path that the caller must not delete
    until the subprocess finishes.
    """
    with open(metadata_path, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    existing = [
        r for r in records
        if (pred_mask_root / f"{r['sample_name']}.npz").exists()
    ]

    print(f"Metadata filter: {len(existing)}/{len(records)} images have predictions in {pred_mask_root.name}/")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in existing:
        tmp.write(json.dumps(r) + "\n")
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def run_evaluator(args: argparse.Namespace) -> None:
    filtered_metadata = filter_metadata_to_existing(args.pred_metadata, args.pred_mask_root)

    try:
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--pred-mask-root", str(args.pred_mask_root),
            "--pred-metadata",  str(filtered_metadata),
            "--output-csv",     str(args.output_csv),
            "--num-workers",    str(args.num_workers),
        ]
        if args.skip_boundary:
            cmd.append("--skip-boundary")
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.area_threshold is not None:
            cmd += ["--area-threshold", str(args.area_threshold)]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        filtered_metadata.unlink(missing_ok=True)


def compute_delta(
    moe_csv: Path,
    medsam_csv: Path,
    delta_csv: Path,
) -> None:
    """Compute per-fragment metric delta (MoE − MedSAM) and save CSV."""
    moe    = pd.read_csv(moe_csv)
    medsam = pd.read_csv(medsam_csv)

    join_cols = ["sample_name", "fragment_index"]
    merged = moe.merge(medsam, on=join_cols, suffixes=("_moe", "_medsam"))

    for metric in ["dice", "iou", "hd95", "assd"]:
        col_moe    = f"{metric}_moe"
        col_medsam = f"{metric}_medsam"
        if col_moe in merged.columns and col_medsam in merged.columns:
            merged[f"delta_{metric}"] = merged[col_moe] - merged[col_medsam]

    delta_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(delta_csv, index=False)
    print(f"\nDelta CSV saved: {delta_csv}")

    print("\nMoE vs MedSAM summary:")
    for metric in ["dice", "iou"]:
        col = f"delta_{metric}"
        if col in merged.columns:
            delta = merged[col].dropna()
            print(
                f"  {metric}: mean delta = {delta.mean():+.4f}  "
                f"improved = {(delta > 0).mean() * 100:.1f}% of fragments  "
                f"n = {len(delta)}"
            )


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.delta_only:
        if not args.output_csv.exists():
            raise FileNotFoundError(
                f"--delta-only requires an existing MoE evaluation CSV: {args.output_csv}\n"
                "Run without --delta-only first."
            )
        print(f"--delta-only: skipping evaluation, using existing {args.output_csv}")
    else:
        run_evaluator(args)

    if args.delta or args.delta_only:
        if not args.medsam_csv.exists():
            print(f"WARNING: MedSAM CSV not found at {args.medsam_csv} — skipping delta.")
        else:
            compute_delta(args.output_csv, args.medsam_csv, args.delta_csv)


if __name__ == "__main__":
    main()
