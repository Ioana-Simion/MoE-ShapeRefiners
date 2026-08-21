#!/usr/bin/env python3
"""PLAN_B (cnnNoROI variant): MedSAM baseline + cnnNoROI eval on RAM-H1200,
at a given routing threshold. Works for frozen PENGWIN checkpoints (the
original zero-shot transfer question) or freshly-trained RAM-H1200 checkpoints
(--cnnnoroi-checkpoint-dir checkpoints/cnnNoROI_ramh1200) -- same script, just
point it at a different checkpoint directory.

Reuses the generic (non-FlowSDF-specific) pieces of run_ramh1200_flowsdf.py
directly rather than duplicating them: the adapter, MedSAM Stage-1, gating/
reroute-by-threshold machinery, evaluation, and visualization are identical
regardless of which expert architecture Step 4 uses. Only Step 4 itself
(batched cnnNoROI forward passes instead of FlowSDF's ODE sampling) is new.

Single-device only (no --num-gpus sharding) -- cnnNoROI is cheap enough
(~371k params, one forward pass per fragment, no ODE sampling) that this
hasn't been needed the way it was for FlowSDF.

Usage:
    # Evaluate against a specific (e.g. train-derived) threshold, matching
    # whatever threshold the checkpoints were actually trained/routed with:
    python src/run_ramh1200_cnnnoroi.py --split test \\
        --cnnnoroi-checkpoint-dir checkpoints/cnnNoROI_ramh1200 \\
        --custom-threshold 3951

    # Zero-shot transfer of the frozen PENGWIN cnnNoROI experts (matches
    # what inference_new_dataset.ipynb did, but as a script with --custom-threshold
    # / --pengwin-reroute / --small-expert-only support):
    python src/run_ramh1200_cnnnoroi.py --split test --pengwin-reroute
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MEDSAM_REPO_ROOT = WORKSPACE_ROOT / "MedSAM"

for _p in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "gating_mechanism",
    MEDSAM_REPO_ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prepare_ramh1200_boxes import process_split  # noqa: E402
from dataloader_utils import load_prediction_masks, resize_binary_nearest  # noqa: E402
from sdf_utils import sdf_channel_from_mask  # noqa: E402
from cnnNoROI.cnnMoE import CNNExpert  # noqa: E402

# Reused directly from run_ramh1200_flowsdf.py -- these are generic (adapter/
# MedSAM/gating/eval/viz), not FlowSDF-specific. Importing the module is safe:
# everything in it is guarded by `if __name__ == "__main__":`.
from run_ramh1200_flowsdf import (  # noqa: E402
    pick_devices,
    run_medsam_shard,
    compute_gating,
    reroute_to_area_threshold,
    evaluate_predictions,
    evaluate_threshold_reroute,
    save_visualizations,
    PENGWIN_SMALL_THRESHOLD,
)

FEAT_SIZE = 64
OUTPUT_SIZE = 1024


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "RAM-H1200-v1_dataset" / "Segmentation")
    parser.add_argument("--boxes-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "bounding-boxes")
    parser.add_argument("--medsam-pred-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "medsam-predictions")
    parser.add_argument("--cnnnoroi-pred-root", type=Path, default=None,
                         help="Defaults to data/ramh1200/cnnnoroi-predictions/<checkpoint-dir-name> so runs against "
                              "different checkpoints (frozen vs. freshly-trained) don't collide.")
    parser.add_argument("--figures-root", type=Path, default=None, help="Defaults alongside --cnnnoroi-pred-root.")
    parser.add_argument("--medsam-checkpoint", type=Path, default=MEDSAM_REPO_ROOT / "medsam_vit_b.pth")
    parser.add_argument("--cnnnoroi-checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints" / "cnnNoROI",
                         help="Frozen PENGWIN checkpoints by default. Pass checkpoints/cnnNoROI_ramh1200 to "
                              "evaluate the freshly-trained RAM-H1200 experts instead.")
    parser.add_argument("--batch-size", type=int, default=32, help="Fragments per batched cnnNoROI forward pass (same expert).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images (smoke test).")
    parser.add_argument("--n-viz-each", type=int, default=3, help="How many best cases to save panels for.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", help="Skip MedSAM/cnnNoROI outputs that already exist on disk.")
    parser.add_argument(
        "--pengwin-reroute", action="store_true",
        help="Additionally evaluate at PENGWIN's own absolute area threshold instead of this dataset's "
             "relative median. See run_ramh1200_flowsdf.py --pengwin-reroute for the full rationale.",
    )
    parser.add_argument(
        "--custom-threshold", type=float, default=None,
        help="ADDITIVE comparison pass at an arbitrary absolute area threshold (px), on top of the "
             "median-routed primary run below (only recomputes fragments that flip expert). Use "
             "--gating-threshold instead if you don't want the median pass at all.",
    )
    parser.add_argument(
        "--gating-threshold", type=float, default=None,
        help="Skip the RAM-H1200-relative median entirely and make THIS value the primary routing "
             "threshold (Steps 3/4/5) instead -- a full replacement, not an additive pass, so no "
             "median-routed inference/eval happens. Use this to evaluate a model trained specifically "
             "at one threshold (e.g. data/ramh1200/gating_threshold_train.txt) without wasting a full "
             "inference pass on a median routing the model wasn't even trained with.",
    )
    parser.add_argument(
        "--small-expert-only", action="store_true",
        help="Additionally force EVERY fragment through expert_small regardless of area.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage: batched cnnNoROI inference (one checkpoint dir, one device)
# ---------------------------------------------------------------------------
def load_cnnnoroi_experts(checkpoint_dir: Path, device: str) -> dict[str, CNNExpert]:
    experts = {}
    for expert_id in ("expert_small", "expert_large"):
        ckpt_path = checkpoint_dir / f"{expert_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        model = CNNExpert(c_in=258).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        experts[expert_id] = model
        print(f"loaded {expert_id} from {ckpt_path}")
    return experts


def run_cnnnoroi_batch(model: CNNExpert, embeddings: torch.Tensor, binary_masks: list[np.ndarray], device: str) -> np.ndarray:
    """embeddings: (B, 256, 64, 64) already on device. binary_masks: list of (1024, 1024) arrays.
    Returns (B, 1024, 1024) uint8."""
    mask_t = torch.from_numpy(np.stack(binary_masks).astype(np.float32)).unsqueeze(1).to(device)  # (B, 1, 1024, 1024)
    mask_small = F.interpolate(mask_t, size=(FEAT_SIZE, FEAT_SIZE), mode="nearest")  # (B, 1, 64, 64)
    sdf = torch.from_numpy(np.stack([
        sdf_channel_from_mask(mask_small[i, 0].cpu().numpy()) for i in range(mask_small.shape[0])
    ])).to(device)  # (B, 1, 64, 64)

    x = torch.cat([embeddings, mask_small, sdf], dim=1)  # (B, 258, 64, 64)
    with torch.no_grad():
        pred = model(x)  # (B, 1, 64, 64)
    pred_up = F.interpolate(pred, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear", align_corners=False)
    return (pred_up[:, 0].cpu().numpy() > 0.5).astype(np.uint8)


def run_cnnnoroi_shard(
    sample_names: list[str],
    gated_df: pd.DataFrame,
    device: str,
    checkpoint_dir: Path,
    batch_size: int,
    cnnnoroi_masks_dir: Path,
    skip_existing: bool,
) -> None:
    experts = load_cnnnoroi_experts(checkpoint_dir, device)

    pending_sample_names = sample_names
    if skip_existing:
        pending_sample_names = [s for s in sample_names if not (cnnnoroi_masks_dir / f"{s}.npz").exists()]
    if not pending_sample_names:
        return

    shard_df = gated_df[gated_df["sample_name"].isin(pending_sample_names)]
    tasks_by_expert: dict[str, list[dict]] = {"expert_small": [], "expert_large": []}
    for t in shard_df.to_dict("records"):
        tasks_by_expert[t["expert"]].append(t)

    embedding_cache: dict[str, torch.Tensor] = {}
    medsam_masks_cache: dict[str, np.ndarray] = {}

    def get_embedding(path: str) -> torch.Tensor:
        if path not in embedding_cache:
            embedding_cache[path] = torch.from_numpy(np.load(path)).float()
        return embedding_cache[path]

    def get_medsam_masks(path: str) -> np.ndarray:
        if path not in medsam_masks_cache:
            medsam_masks_cache[path] = load_prediction_masks(path)
        return medsam_masks_cache[path]

    out_arrays: dict[str, np.ndarray] = {}
    remaining_by_sample: dict[str, int] = shard_df.groupby("sample_name").size().to_dict()

    for expert_id, expert_tasks in tasks_by_expert.items():
        model = experts[expert_id]
        for batch_start in tqdm(range(0, len(expert_tasks), batch_size), desc=f"[{device}] cnnNoROI {expert_id}"):
            batch_tasks = expert_tasks[batch_start:batch_start + batch_size]

            embeddings = torch.stack([get_embedding(t["embedding_path"]) for t in batch_tasks]).to(device)
            binary_masks = []
            for t in batch_tasks:
                masks = get_medsam_masks(t["binary_masks_path"])
                binary_masks.append(masks[int(t["medsam_instance_id"]) - 1])

            refined_batch = run_cnnnoroi_batch(model, embeddings, binary_masks, device)

            for t, refined in zip(batch_tasks, refined_batch):
                sample_name = t["sample_name"]
                if sample_name not in out_arrays:
                    n_frag = get_medsam_masks(t["binary_masks_path"]).shape[0]
                    out_arrays[sample_name] = np.zeros((n_frag, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
                idx = int(t["medsam_instance_id"]) - 1
                out_arrays[sample_name][idx] = refined
                remaining_by_sample[sample_name] -= 1
                if remaining_by_sample[sample_name] == 0:
                    np.savez_compressed(cnnnoroi_masks_dir / f"{sample_name}.npz", masks=out_arrays.pop(sample_name))

    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_cnnnoroi_reroute_shard(
    sample_names: list[str],
    gated_df: pd.DataFrame,
    gated_df_target: pd.DataFrame,
    device: str,
    checkpoint_dir: Path,
    batch_size: int,
    source_masks_dir: Path,
    dest_masks_dir: Path,
    skip_existing: bool,
    label: str = "reroute",
) -> None:
    """Same 'only recompute what changed' pattern as run_ramh1200_flowsdf.py's
    run_reroute_shard, adapted for batched cnnNoROI forward passes."""
    pending_sample_names = sample_names
    if skip_existing:
        pending_sample_names = [s for s in sample_names if not (dest_masks_dir / f"{s}.npz").exists()]
    if not pending_sample_names:
        return

    orig_by_key = {
        (r["sample_name"], int(r["medsam_instance_id"])): r["expert"]
        for r in gated_df[gated_df["sample_name"].isin(pending_sample_names)].to_dict("records")
    }
    new_rows = gated_df_target[gated_df_target["sample_name"].isin(pending_sample_names)].to_dict("records")
    flipped_tasks = [
        r for r in new_rows
        if orig_by_key.get((r["sample_name"], int(r["medsam_instance_id"]))) != r["expert"]
    ]
    print(f"[{device}] {label}: {len(flipped_tasks)} / {len(new_rows)} fragments change expert assignment")

    out_arrays: dict[str, np.ndarray] = {
        s: load_prediction_masks(source_masks_dir / f"{s}.npz").copy() for s in pending_sample_names
    }

    if flipped_tasks:
        experts: dict[str, CNNExpert] = {}
        tasks_by_expert: dict[str, list[dict]] = {}
        for t in flipped_tasks:
            tasks_by_expert.setdefault(t["expert"], []).append(t)

        embedding_cache: dict[str, torch.Tensor] = {}
        medsam_masks_cache: dict[str, np.ndarray] = {}

        def get_embedding(path: str) -> torch.Tensor:
            if path not in embedding_cache:
                embedding_cache[path] = torch.from_numpy(np.load(path)).float()
            return embedding_cache[path]

        def get_medsam_masks(path: str) -> np.ndarray:
            if path not in medsam_masks_cache:
                medsam_masks_cache[path] = load_prediction_masks(path)
            return medsam_masks_cache[path]

        for expert_id, expert_tasks in tasks_by_expert.items():
            if expert_id not in experts:
                ckpt_path = checkpoint_dir / f"{expert_id}_best.pth"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
                model = CNNExpert(c_in=258).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                experts[expert_id] = model
            model = experts[expert_id]

            for batch_start in tqdm(range(0, len(expert_tasks), batch_size), desc=f"[{device}] {label} {expert_id}"):
                batch_tasks = expert_tasks[batch_start:batch_start + batch_size]
                embeddings = torch.stack([get_embedding(t["embedding_path"]) for t in batch_tasks]).to(device)
                binary_masks = []
                for t in batch_tasks:
                    masks = get_medsam_masks(t["binary_masks_path"])
                    binary_masks.append(masks[int(t["medsam_instance_id"]) - 1])

                refined_batch = run_cnnnoroi_batch(model, embeddings, binary_masks, device)
                for t, refined in zip(batch_tasks, refined_batch):
                    idx = int(t["medsam_instance_id"]) - 1
                    out_arrays[t["sample_name"]][idx] = refined

    for sample_name, arr in out_arrays.items():
        np.savez_compressed(dest_masks_dir / f"{sample_name}.npz", masks=arr)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_tag = args.cnnnoroi_checkpoint_dir.name
    if args.cnnnoroi_pred_root is None:
        args.cnnnoroi_pred_root = REPO_ROOT / "data" / "ramh1200" / "cnnnoroi-predictions" / ckpt_tag
    if args.figures_root is None:
        args.figures_root = REPO_ROOT / "figures" / "ramh1200_cnnnoroi" / ckpt_tag

    device = pick_devices(1)[0]
    print(f"device: {device}")
    print(f"cnnnoroi-checkpoint-dir: {args.cnnnoroi_checkpoint_dir}")
    print(f"cnnnoroi-pred-root: {args.cnnnoroi_pred_root}")
    print(f"figures-root: {args.figures_root}")

    t0 = time.time()

    # --- Step 1: adapter -----------------------------------------------------
    print("\n=== Step 1: adapter (COCO -> boxes/gt_masks/metadata) ===")
    process_split(
        dataset_root=args.dataset_root,
        output_root=args.boxes_root,
        split=args.split,
        image_size=1024,
        limit=args.limit,
    )
    records = [json.loads(l) for l in (args.boxes_root / args.split / "metadata.jsonl").open()]
    print(f"{len(records)} images")

    # --- Step 2: MedSAM Stage-1 -----------------------------------------------
    print("\n=== Step 2: MedSAM Stage-1 ===")
    medsam_pred_records = run_medsam_shard(
        records, device, args.medsam_checkpoint, args.medsam_pred_root, args.split, args.skip_existing
    )
    pred_metadata_path = args.medsam_pred_root / args.split / "metadata.jsonl"
    with pred_metadata_path.open("w") as f:
        for r in medsam_pred_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(medsam_pred_records)} MedSAM prediction records")

    # --- Step 3: gating (RAM-H1200-relative median, or --gating-threshold override) --
    print("\n=== Step 3: gating threshold ===")
    gated_df, area_threshold = compute_gating(medsam_pred_records, threshold_override=args.gating_threshold)
    if args.gating_threshold is not None:
        print(f"--gating-threshold set: skipping the RAM-H1200-relative median, routing on {area_threshold:.1f}px directly")
    gating_dir = REPO_ROOT / "src" / "gating_mechanism"
    gated_csv_path = gating_dir / f"gated_ramh1200_cnnnoroi_{args.split}_records.csv"
    gated_df.to_csv(gated_csv_path, index=False)
    print(f"area threshold: {area_threshold:.1f}px, saved {len(gated_df)} gated fragments -> {gated_csv_path}")
    print(gated_df["expert"].value_counts().to_string())

    # --- Step 4: batched cnnNoROI inference -----------------------------------
    print(f"\n=== Step 4: cnnNoROI inference (batch_size={args.batch_size}) ===")
    cnnnoroi_masks_dir = args.cnnnoroi_pred_root / args.split / "binary_masks"
    cnnnoroi_masks_dir.mkdir(parents=True, exist_ok=True)

    sample_names = [r["sample_name"] for r in medsam_pred_records]
    run_cnnnoroi_shard(
        sample_names, gated_df, device, args.cnnnoroi_checkpoint_dir,
        args.batch_size, cnnnoroi_masks_dir, args.skip_existing,
    )

    eval_dir = args.cnnnoroi_pred_root / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 4b/4c/4d: optional reroutes, mirroring run_ramh1200_flowsdf.py -----
    reroute_outputs: dict[str, dict] = {}
    if args.pengwin_reroute:
        print(f"\n=== Step 4b: PENGWIN-absolute reroute (threshold={PENGWIN_SMALL_THRESHOLD:.0f}px) ===")
        gated_df_pengwin = reroute_to_area_threshold(gated_df, PENGWIN_SMALL_THRESHOLD)
        gated_df_pengwin.to_csv(gating_dir / f"gated_ramh1200_cnnnoroi_pengwin_{args.split}_records.csv", index=False)
        changed = int((gated_df_pengwin["expert"] != gated_df["expert"]).sum())
        print(f"{changed} / {len(gated_df)} fragments change expert assignment ({changed / len(gated_df):.1%})")
        print(gated_df_pengwin["expert"].value_counts().to_string())

        pengwin_masks_dir = args.cnnnoroi_pred_root / args.split / "binary_masks_pengwin_convention"
        pengwin_masks_dir.mkdir(parents=True, exist_ok=True)
        run_cnnnoroi_reroute_shard(
            sample_names, gated_df, gated_df_pengwin, device, args.cnnnoroi_checkpoint_dir,
            args.batch_size, cnnnoroi_masks_dir, pengwin_masks_dir, args.skip_existing,
            label="PENGWIN-absolute routing",
        )
        reroute_outputs["pengwin"] = {"gated_df": gated_df_pengwin, "masks_dir": pengwin_masks_dir}

    if args.small_expert_only:
        print("\n=== Step 4c: expert_small on every fragment ===")
        gated_df_all_small = gated_df.copy()
        gated_df_all_small["expert"] = "expert_small"
        n_recompute = int((gated_df["expert"] != "expert_small").sum())
        print(f"{n_recompute} / {len(gated_df)} fragments are not already expert_small and will be recomputed")

        all_small_masks_dir = args.cnnnoroi_pred_root / args.split / "binary_masks_all_small"
        all_small_masks_dir.mkdir(parents=True, exist_ok=True)
        run_cnnnoroi_reroute_shard(
            sample_names, gated_df, gated_df_all_small, device, args.cnnnoroi_checkpoint_dir,
            args.batch_size, cnnnoroi_masks_dir, all_small_masks_dir, args.skip_existing,
            label="expert_small on everything",
        )
        reroute_outputs["all_small"] = {"gated_df": gated_df, "masks_dir": all_small_masks_dir}

    if args.custom_threshold is not None:
        print(f"\n=== Step 4d: custom-threshold reroute (threshold={args.custom_threshold:.1f}px) ===")
        gated_df_custom = reroute_to_area_threshold(gated_df, args.custom_threshold)
        gated_df_custom.to_csv(gating_dir / f"gated_ramh1200_cnnnoroi_custom_{args.split}_records.csv", index=False)
        changed = int((gated_df_custom["expert"] != gated_df["expert"]).sum())
        print(f"{changed} / {len(gated_df)} fragments change expert assignment ({changed / len(gated_df):.1%})")
        print(gated_df_custom["expert"].value_counts().to_string())

        custom_masks_dir = args.cnnnoroi_pred_root / args.split / "binary_masks_custom_threshold"
        custom_masks_dir.mkdir(parents=True, exist_ok=True)
        run_cnnnoroi_reroute_shard(
            sample_names, gated_df, gated_df_custom, device, args.cnnnoroi_checkpoint_dir,
            args.batch_size, cnnnoroi_masks_dir, custom_masks_dir, args.skip_existing,
            label="custom-threshold routing",
        )
        reroute_outputs["custom"] = {"gated_df": gated_df_custom, "masks_dir": custom_masks_dir}

    # --- Step 5: evaluation ----------------------------------------------------
    print("\n=== Step 5: evaluation ===")
    baseline_df = evaluate_predictions(medsam_pred_records, args.medsam_pred_root / args.split / "binary_masks", gated_df, desc="eval: MedSAM baseline")
    cnnnoroi_df = evaluate_predictions(medsam_pred_records, cnnnoroi_masks_dir, gated_df, desc="eval: cnnNoROI (RAM-H1200-relative)")

    baseline_df.to_csv(eval_dir / "evaluation_medsam_baseline.csv", index=False)
    cnnnoroi_df.to_csv(eval_dir / "evaluation_cnnnoroi.csv", index=False)

    print("\nMedSAM baseline (overall):")
    print(baseline_df[["dice", "iou", "hd95", "assd"]].mean())
    print("\nMedSAM + cnnNoROI (overall):")
    print(cnnnoroi_df[["dice", "iou", "hd95", "assd"]].mean())
    print("\nMedSAM baseline by size group:")
    print(baseline_df.groupby("size_group")[["dice", "iou", "hd95", "assd"]].mean())
    print("\nMedSAM + cnnNoROI by size group:")
    print(cnnnoroi_df.groupby("size_group")[["dice", "iou", "hd95", "assd"]].mean())

    if args.pengwin_reroute:
        evaluate_threshold_reroute(
            "pengwin", gated_df, reroute_outputs["pengwin"]["gated_df"], reroute_outputs["pengwin"]["masks_dir"],
            medsam_pred_records, args.medsam_pred_root, args.split, cnnnoroi_masks_dir, eval_dir,
        )

    if args.custom_threshold is not None:
        evaluate_threshold_reroute(
            "custom", gated_df, reroute_outputs["custom"]["gated_df"], reroute_outputs["custom"]["masks_dir"],
            medsam_pred_records, args.medsam_pred_root, args.split, cnnnoroi_masks_dir, eval_dir,
        )

    if args.small_expert_only:
        all_small_df = evaluate_predictions(medsam_pred_records, reroute_outputs["all_small"]["masks_dir"], gated_df, desc="eval: expert_small on everything")
        all_small_df.to_csv(eval_dir / "evaluation_cnnnoroi_all_small.csv", index=False)
        print("\nMedSAM + cnnNoROI, expert_small on EVERY fragment (overall):")
        print(all_small_df[["dice", "iou", "hd95", "assd"]].mean())
        print("\nMedSAM + cnnNoROI, expert_small on EVERY fragment, by original size group:")
        print(all_small_df.groupby("size_group")[["dice", "iou", "hd95", "assd"]].mean())

    # --- Step 6: visualizations --------------------------------------------------
    print("\n=== Step 6: visualizations ===")
    all_manifest_rows = []
    all_manifest_rows += save_visualizations(
        medsam_pred_records, args.medsam_pred_root, cnnnoroi_masks_dir,
        args.split, args.figures_root, args.n_viz_each, tag="primary_ram_h1200_relative",
    )
    for tag, info in reroute_outputs.items():
        all_manifest_rows += save_visualizations(
            medsam_pred_records, args.medsam_pred_root, info["masks_dir"],
            args.split, args.figures_root, args.n_viz_each, tag=tag,
        )

    manifest_df = pd.DataFrame(all_manifest_rows)
    manifest_path = args.figures_root / args.split / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"wrote combined manifest ({len(manifest_df)} rows) -> {manifest_path}")

    elapsed = time.time() - t0
    print(f"\ntotal elapsed: {elapsed / 3600:.2f} hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
