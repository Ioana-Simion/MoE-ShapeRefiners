#!/usr/bin/env python3
"""Visualize learned-MoE (hard or soft) predictions overlaid on the X-ray + GT.

Unlike visualize_moe_pengwin.py, this does NOT need a MedSAM comparison /
delta CSV — just the prediction masks + the evaluation CSV that scores them
(evaluate_medsam_pengwin.py's output for whichever mode you're looking at).
If you also have gate_weights_{split}.csv, pass --gate-weights to annotate
each figure with (g0, g1, argmax_expert) so you can see the routing decision
next to the actual mask.

Selects fragments by four criteria and saves one PNG per fragment:
    best   — highest Dice
    worst  — lowest Dice
    small  — random from the bbox-area "small" group (eval CSV's size_group)
    large  — random from the bbox-area "large" group

Each figure shows 4 panels, cropped to the fragment bounding box:
    X-ray  |  Prediction overlay  |  GT overlay  |  Diff (pred vs GT)

Diff colour key:
    white = true positive (both foreground)   red = false negative (missed)
    blue  = false positive (over-predicted)   dark = true negative (background)

Usage:
    python evaluation/visualize_moe_learned.py \\
        --eval-csv     data/moe-learned-predictions/evaluation_hard.csv \\
        --pred-mask-root data/moe-learned-predictions/hard/binary_masks \\
        --gate-weights data/moe-learned-predictions/gate_weights_test.csv \\
        --method-label "Learned MoE (hard)" \\
        --output-dir   data/moe-learned-predictions/visualizations_hard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader_utils import (                       # noqa: E402
    decode_pengwin_fragment_from_record,
    load_pengwin_label,
    load_prediction_masks,
    resolve_existing_path,
)
from visualize_moe_pengwin import (                    # noqa: E402
    build_image_map, load_xray, resize_mask, compute_bbox, crop,
    xray_to_display, overlay, annotate_panel_score,
)

COLOUR_PRED = (1.0, 0.5, 0.0)   # orange
COLOUR_GT   = (0.2, 0.9, 0.2)   # green
OVERLAY_ALPHA = 0.45
MASK_SIZE = 1024


def diff_pred_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """White=TP  Red=FN (missed)  Blue=FP (over-predicted)  Dark=TN."""
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[pred_b & gt_b]   = [1.00, 1.00, 1.00]   # TP — white
    rgb[~pred_b & gt_b]  = [0.90, 0.10, 0.10]   # FN — red
    rgb[pred_b & ~gt_b]  = [0.20, 0.40, 1.00]   # FP — blue
    rgb[~pred_b & ~gt_b] = [0.15, 0.15, 0.15]   # TN — dark
    return rgb


def visualize_fragment(
    row: pd.Series,
    xray_map: dict[str, Path],
    pred_root: Path,
    output_dir: Path,
    group_label: str,
    method_label: str,
    gate_row: pd.Series | None,
) -> None:
    sample_name   = row["sample_name"]
    fragment_idx  = int(row["fragment_index"])
    category_id   = int(row["category_id"])
    fragment_id   = int(row["fragment_id"])
    category_name = str(row.get("category_name", ""))
    size_group    = str(row.get("size_group", ""))
    dice          = float(row.get("dice", float("nan")))
    label_path    = resolve_existing_path(str(row.get("original_label_path", "")))

    pred_file = resolve_existing_path(pred_root / f"{sample_name}.npz")
    if not pred_file.exists():
        print(f"  SKIP {sample_name}: mask file missing")
        return
    pred_mask = load_prediction_masks(pred_file)[fragment_idx]   # (1024, 1024)

    if not label_path.exists():
        print(f"  SKIP {sample_name}: GT label missing at {label_path}")
        return
    seg = load_pengwin_label(label_path)
    gt_448 = decode_pengwin_fragment_from_record(
        seg, {"category_id": category_id, "fragment_id": fragment_id}
    )
    gt_1024 = resize_mask(gt_448, (MASK_SIZE, MASK_SIZE))

    bbox = compute_bbox(gt_1024) or compute_bbox(pred_mask)
    if bbox is None:
        print(f"  SKIP {sample_name}: no foreground pixels found")
        return

    xray_full = load_xray(xray_map.get(sample_name, Path("__missing__")))
    if xray_full is not None and xray_full.shape != (MASK_SIZE, MASK_SIZE):
        from PIL import Image
        xray_full = np.array(
            Image.fromarray(xray_full).resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR),
            dtype=np.uint8,
        )
    crop_hw = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    xray_disp = xray_to_display(crop(xray_full, bbox) if xray_full is not None else None, crop_hw)

    pred_crop = crop(pred_mask, bbox)
    gt_crop   = crop(gt_1024, bbox)

    panel_xray = np.stack([xray_disp / 255.0] * 3, axis=-1)
    panel_pred = overlay(xray_disp, pred_crop, COLOUR_PRED, OVERLAY_ALPHA)
    panel_gt   = overlay(xray_disp, gt_crop, COLOUR_GT, OVERLAY_ALPHA)
    panel_diff = diff_pred_gt(pred_crop, gt_crop)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("#ffffff")
    panels = [panel_xray, panel_pred, panel_gt, panel_diff]
    titles = ["X-ray", method_label, "Ground truth", "Diff (pred vs GT)"]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, color="black", fontsize=11, pad=4)
        ax.axis("off")

    gate_text = f"Dice {dice:.3f}"
    if gate_row is not None:
        gate_text += (
            f"\ng=[{gate_row['g0']:.2f},{gate_row['g1']:.2f}] "
            f"-> expert {int(gate_row['argmax_expert'])}"
        )
    annotate_panel_score(axes[1], gate_text, "#ffaa44")

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=(1.00, 1.00, 1.00), label="TP"),
        mpatches.Patch(color=(0.90, 0.10, 0.10), label="FN (missed)"),
        mpatches.Patch(color=(0.20, 0.40, 1.00), label="FP (over-pred)"),
        mpatches.Patch(color=(0.15, 0.15, 0.15), label="TN"),
    ]
    axes[3].legend(
        handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, fontsize=8, framealpha=0.3, labelcolor="black", facecolor="#ffffff",
    )

    suptitle = f"{sample_name}  ·  {category_name}  ·  {size_group}  ·  [{group_label}]"
    fig.suptitle(suptitle, color="black", fontsize=11, y=1.01)
    plt.tight_layout()

    fname = f"{group_label}__{sample_name}__frag{fragment_idx:03d}__{category_name}__{size_group}__dice{dice:.3f}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {out_path.name}")


def select_fragments(
    df: pd.DataFrame, n_best: int, n_worst: int, n_random_small: int, n_random_large: int, seed: int = 42,
) -> list[tuple[pd.Series, str]]:
    rng = np.random.default_rng(seed)
    selected: list[tuple[pd.Series, str]] = []
    valid = df.dropna(subset=["dice"])

    def pick(subset: pd.DataFrame, n: int, label: str, ascending: bool) -> None:
        if subset.empty or n == 0:
            return
        for _, row in subset.sort_values("dice", ascending=ascending).head(n).iterrows():
            selected.append((row, label))

    pick(valid, n_best, "best", ascending=False)
    pick(valid, n_worst, "worst", ascending=True)

    if "size_group" in df.columns:
        for group_name, n in (("small", n_random_small), ("large", n_random_large)):
            subset = valid[valid["size_group"] == group_name]
            if not subset.empty and n > 0:
                idx = rng.choice(len(subset), size=min(n, len(subset)), replace=False)
                for i in idx:
                    selected.append((subset.iloc[i], f"random_{group_name}"))

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-csv", type=Path, required=True,
                        help="evaluate_medsam_pengwin.py output for the mode being visualized "
                             "(e.g. data/moe-learned-predictions/evaluation_hard.csv).")
    parser.add_argument("--pred-mask-root", type=Path, required=True,
                        help="Directory with this mode's .npz mask files "
                             "(e.g. data/moe-learned-predictions/hard/binary_masks).")
    parser.add_argument("--gate-weights", type=Path, default=None,
                        help="Optional gate_weights_{split}.csv to annotate g0/g1/argmax_expert.")
    parser.add_argument("--metadata", type=Path,
                        default=PROJECT_ROOT / "data" / "medsam-predictions" / "metadata.jsonl",
                        help="Used only to find each sample's raw X-ray path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", default="Learned MoE")
    parser.add_argument("--n-best", type=int, default=5)
    parser.add_argument("--n-worst", type=int, default=5)
    parser.add_argument("--n-random-small", type=int, default=5)
    parser.add_argument("--n-random-large", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eval_csv.exists():
        raise FileNotFoundError(f"Eval CSV not found: {args.eval_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval CSV: {args.eval_csv}")
    df = pd.read_csv(args.eval_csv)
    print(f"  {len(df)} fragments")

    gate_df = None
    if args.gate_weights is not None:
        if not args.gate_weights.exists():
            raise FileNotFoundError(f"Gate weights CSV not found: {args.gate_weights}")
        gate_df = pd.read_csv(args.gate_weights).set_index(["sample_name", "fragment_index"])

    print("Building X-ray image map …")
    xray_map = build_image_map(args.metadata)

    fragments = select_fragments(
        df, n_best=args.n_best, n_worst=args.n_worst,
        n_random_small=args.n_random_small, n_random_large=args.n_random_large, seed=args.seed,
    )
    print(f"Selected {len(fragments)} fragments — generating figures …\n")

    for row, group_label in fragments:
        print(f"[{group_label}] {row['sample_name']}")
        gate_row = None
        if gate_df is not None:
            key = (row["sample_name"], int(row["fragment_index"]))
            if key in gate_df.index:
                gate_row = gate_df.loc[key]
        visualize_fragment(
            row=row, xray_map=xray_map, pred_root=args.pred_mask_root,
            output_dir=args.output_dir, group_label=group_label,
            method_label=args.method_label, gate_row=gate_row,
        )

    print(f"\nDone. {len(fragments)} figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
