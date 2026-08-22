#!/usr/bin/env python3
"""Dilation-control check for the FlowSDF zero-shot transfer result on RAM-H1200.

Question: FlowSDF binarizes its sampled SDF at `sdf <= 0.03`, a threshold tuned
on PENGWIN. That's a fixed *absolute* margin -- a few pixels regardless of
object size -- which could disproportionately dilate thin, elongated hand
bones relative to chunky pelvis fragments (higher perimeter/area ratio ->
bigger relative effect from the same absolute margin). If MedSAM's baseline
already under-segments thin structures (a known failure mode), a completely
untrained morphological dilation could recover some of that boundary and show
up as a dice "improvement" with zero relationship to whether FlowSDF learned
anything about hand-bone shape.

This script applies plain `scipy.ndimage.binary_dilation` (no model, no
FlowSDF) to the already-computed MedSAM baseline masks at a range of radii and
evaluates against GT, on the full RAM-H1200 test split (267 images). Compare
the resulting dice/iou against FlowSDF's own reported numbers on the same
split (from run_ramh1200_flowsdf.py --ode-steps 4, run separately on cluster):

    MedSAM baseline (overall):                 dice 0.744
    MedSAM + frozen FlowSDF (overall):         dice 0.714
    MedSAM baseline, expert_small group:       dice 0.682
    MedSAM + frozen FlowSDF, expert_small:     dice 0.710

If plain dilation at some radius gets close to FlowSDF's expert_small dice,
that's evidence the "improvement" is mostly margin, not learned correction. If
FlowSDF clearly beats the best dilation radius, that's evidence of real
shape-informed correction beyond a fixed-margin artifact.

Dice/IoU only (fast, no distance transforms) for the radius sweep -- add
--boundary-radius to also compute HD95/ASSD for one specific radius.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dataloader_utils import load_prediction_masks, resize_binary_nearest  # noqa: E402
from evaluation.evaluate_medsam_pengwin import dice_score, iou_score, boundary_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--boxes-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "bounding-boxes")
    parser.add_argument("--medsam-pred-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "medsam-predictions")
    parser.add_argument("--radii", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8],
                         help="Dilation radii in pixels, applied in the 1024x1024 mask space (same space FlowSDF's margin acts in before upsampling).")
    parser.add_argument("--boundary-radii", type=int, nargs="+", default=None,
                         help="If set, also compute HD95/ASSD (slow) for these radii, in one pass (reuses loaded masks/GT instead of reloading per radius).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "dilation_control.csv")
    return parser.parse_args()


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    structure = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_dilation(mask.astype(bool), structure=structure, iterations=radius).astype(np.uint8)


def main() -> int:
    args = parse_args()

    records = [json.loads(l) for l in (args.boxes_root / args.split / "metadata.jsonl").open()]
    if args.limit is not None:
        records = records[: args.limit]
    print(f"{len(records)} images")

    # --- Recompute the RAM-H1200-relative gating threshold, same procedure as
    # the main script (median predicted-mask area) -- purely for the by-size-group
    # breakdown, doesn't affect the dilation control itself.
    areas = []
    baseline_masks_cache: dict[str, np.ndarray] = {}
    for record in tqdm(records, desc="loading baseline masks + computing areas"):
        masks = load_prediction_masks(args.medsam_pred_root / args.split / "binary_masks" / f"{record['sample_name']}.npz")
        baseline_masks_cache[record["sample_name"]] = masks
        for i in range(masks.shape[0]):
            areas.append(int(masks[i].sum()))
    threshold = float(np.median(areas))
    print(f"gating threshold (RAM-H1200-relative median): {threshold:.1f}px")

    frag_size_group: dict[tuple[str, int], str] = {}
    for record in records:
        masks = baseline_masks_cache[record["sample_name"]]
        for frag, mask in zip(record["fragments"], masks):
            area = int(mask.sum())
            frag_size_group[(record["sample_name"], frag["medsam_instance_id"])] = (
                "expert_small" if area <= threshold else "expert_large"
            )

    # --- Radius sweep: dice + iou only (fast) -----------------------------------
    sweep_rows = []
    for radius in args.radii:
        for record in tqdm(records, desc=f"radius={radius}px"):
            sample_name = record["sample_name"]
            gt_masks = np.load(record["gt_masks_path"])["masks"]
            baseline_masks = baseline_masks_cache[sample_name]

            for i, frag in enumerate(record["fragments"]):
                gt = gt_masks[i]
                pred = dilate_mask(baseline_masks[i], radius)
                pred = resize_binary_nearest(pred, gt.shape)
                sweep_rows.append({
                    "radius": radius,
                    "sample_name": sample_name,
                    "size_group": frag_size_group[(sample_name, frag["medsam_instance_id"])],
                    "category_name": frag["category_name"],
                    "dice": dice_score(pred, gt),
                    "iou": iou_score(pred, gt),
                })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(args.output_csv, index=False)
    print(f"\nsaved {len(sweep_df)} rows -> {args.output_csv}")

    print("\n=== Dilation-control sweep: overall dice/iou by radius ===")
    print(sweep_df.groupby("radius")[["dice", "iou"]].mean())

    print("\n=== Dilation-control sweep: dice/iou by radius and size group ===")
    print(sweep_df.groupby(["radius", "size_group"])[["dice", "iou"]].mean())

    print("\n=== Dilation-control sweep: dice/iou by radius and bone category ===")
    print(sweep_df.groupby(["radius", "category_name"])[["dice", "iou"]].mean().to_string())

    # --- Optional: full boundary metrics for one or more radii, one pass ---------
    if args.boundary_radii:
        print(f"\n=== Full metrics (dice/iou/hd95/assd) at radii={args.boundary_radii} ===")
        rows = []
        for record in tqdm(records, desc=f"boundary metrics radii={args.boundary_radii}"):
            sample_name = record["sample_name"]
            gt_masks = np.load(record["gt_masks_path"])["masks"]
            baseline_masks = baseline_masks_cache[sample_name]

            for radius in args.boundary_radii:
                for i, frag in enumerate(record["fragments"]):
                    gt = gt_masks[i]
                    pred = dilate_mask(baseline_masks[i], radius)
                    pred = resize_binary_nearest(pred, gt.shape)
                    hd95, assd = boundary_metrics(pred, gt)
                    rows.append({
                        "radius": radius,
                        "sample_name": sample_name,
                        "size_group": frag_size_group[(sample_name, frag["medsam_instance_id"])],
                        "category_name": frag["category_name"],
                        "dice": dice_score(pred, gt),
                        "iou": iou_score(pred, gt),
                        "hd95": hd95, "assd": assd,
                    })
        boundary_df = pd.DataFrame(rows)
        radii_tag = "-".join(str(r) for r in args.boundary_radii)
        boundary_df.to_csv(args.output_csv.with_name(f"dilation_control_boundary_r{radii_tag}.csv"), index=False)
        print(boundary_df.groupby("radius")[["dice", "iou", "hd95", "assd"]].mean())
        print(boundary_df.groupby(["radius", "size_group"])[["dice", "iou", "hd95", "assd"]].mean())
        print(boundary_df.groupby(["radius", "category_name"])[["dice", "iou", "hd95", "assd"]].mean().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
