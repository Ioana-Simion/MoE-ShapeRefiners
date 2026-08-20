#!/usr/bin/env python3
"""RAM-H1200 gating threshold -- quality-cliff analysis (script version).

Non-interactive port of ramh1200_threshold_analysis.ipynb, for running as a
cluster job (job 2 of 3: prepare_ramh1200_training_data.py -> THIS ->
train_*_ramh1200.py) rather than by hand in a notebook.

Same methodology PENGWIN's own 5402px threshold came from
(data_distribution/analyze_area_threshold.py: rolling dice-vs-area on GT
fragment area vs. MedSAM baseline dice, finding the area where segmentation
quality actually degrades), reused directly, not reimplemented.

Writes the derived threshold (the "mean dice crosses 0.70" candidate -- the
criterion PENGWIN's own 5402px value sits closest to) to a plain single-number
text file so a downstream job can read it with `cat` / `$()` without any
JSON/CSV parsing:

    data/ramh1200/gating_threshold_<split>.txt

Also writes the full candidates/cutoff-evaluation tables and the rolling
dice/failure-rate plot, for the record.

Usage:
    python src/ramh1200_threshold_analysis.py --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "data_distribution"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataloader_utils import load_prediction_masks, resize_binary_nearest  # noqa: E402
from evaluation.evaluate_medsam_pengwin import dice_score  # noqa: E402
from analyze_area_threshold import rolling_analysis, threshold_candidates, evaluate_cutoffs  # noqa: E402

PENGWIN_ABSOLUTE_THRESHOLD = 5402.0  # for reference/comparison only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                         help="Deriving the routing threshold from a different split than the one "
                              "experts are trained on is a subtle leakage -- use 'train' for a training run.")
    parser.add_argument("--boxes-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "bounding-boxes")
    parser.add_argument("--medsam-pred-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "medsam-predictions")
    parser.add_argument("--window-size", type=int, default=300,
                         help="Rolling window size. PENGWIN used 5000 over ~265k fragments (~1.9%%); "
                              "RAM-H1200 has far fewer fragments, so the default here is smaller.")
    parser.add_argument("--failure-dice-threshold", type=float, default=0.5,
                         help="Dice below this counts as a 'failure' for the rolling failure-rate curve.")
    parser.add_argument("--criterion", default="mean_dice_above_0.70",
                         choices=["mean_dice_above_0.70", "mean_dice_above_0.75", "median_dice_above_0.70",
                                  "failure_rate_below_0.25", "failure_rate_below_0.20", "failure_rate_below_0.15"],
                         help="Which candidate to use as THE derived threshold. Default matches the "
                              "criterion PENGWIN's own 5402px threshold sits closest to.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "ramh1200")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures" / "ramh1200_threshold_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = [json.loads(l) for l in (args.boxes_root / args.split / "metadata.jsonl").open()]
    print(f"{len(records)} images")

    # --- Per-fragment (GT area, MedSAM baseline dice) table ---------------------
    rows = []
    for record in tqdm(records, desc=f"[{args.split}] building (area, dice) table"):
        sample_name = record["sample_name"]
        gt_masks = np.load(record["gt_masks_path"])["masks"]
        baseline_masks = load_prediction_masks(args.medsam_pred_root / args.split / "binary_masks" / f"{sample_name}.npz")

        for i, frag in enumerate(record["fragments"]):
            gt = gt_masks[i]
            pred = resize_binary_nearest(baseline_masks[i], gt.shape)
            rows.append({
                "sample_name": sample_name,
                "category_name": frag["category_name"],
                "area": int(gt.astype(bool).sum()),
                "medsam_dice": dice_score(pred, gt),
            })

    df = pd.DataFrame(rows)
    df = df[df["area"] > 0].sort_values("area").reset_index(drop=True)
    print(f"{len(df)} fragments")

    median_threshold = float(df["area"].median())
    print(f"[{args.split}] GT-area median: {median_threshold:.1f}px")

    # --- Rolling dice / failure-rate vs area -------------------------------------
    rolling = rolling_analysis(df, window_size=args.window_size, failure_threshold=args.failure_dice_threshold)
    print(f"{len(rolling)} rolling-window rows")

    # --- Threshold candidates + cutoff evaluation --------------------------------
    candidates = threshold_candidates(rolling)
    print("\nThreshold candidates:")
    print(candidates.to_string(index=False))

    all_cutoffs = (
        list(candidates["area_cutoff"].dropna())
        + [median_threshold, PENGWIN_ABSOLUTE_THRESHOLD]
        + [df["area"].quantile(q) for q in (0.10, 0.25, 0.33, 0.66)]
    )
    cutoff_eval = evaluate_cutoffs(df, all_cutoffs, args.failure_dice_threshold)
    print("\nCutoff evaluation:")
    print(cutoff_eval.sort_values("area_cutoff").to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / f"threshold_candidates_{args.split}.csv", index=False)
    cutoff_eval.to_csv(args.output_dir / f"threshold_cutoff_evaluation_{args.split}.csv", index=False)

    # --- Derived threshold: the chosen criterion ---------------------------------
    derived_row = candidates[candidates["criterion"] == args.criterion]
    if derived_row.empty or np.isnan(derived_row["area_cutoff"].iloc[0]):
        raise RuntimeError(
            f"criterion '{args.criterion}' did not cross in this data -- see the candidates table above "
            f"for other options and pass one via --criterion."
        )
    derived_threshold = float(derived_row["area_cutoff"].iloc[0])

    threshold_path = args.output_dir / f"gating_threshold_{args.split}.txt"
    threshold_path.write_text(f"{derived_threshold:.1f}\n")
    print(f"\nderived threshold ({args.criterion}): {derived_threshold:.1f}px -> {threshold_path}")

    summary_path = args.output_dir / f"gating_threshold_{args.split}_summary.json"
    summary_path.write_text(json.dumps({
        "split": args.split,
        "criterion": args.criterion,
        "derived_threshold_px": derived_threshold,
        "ram_h1200_median_threshold_px": median_threshold,
        "pengwin_absolute_threshold_px": PENGWIN_ABSOLUTE_THRESHOLD,
        "n_fragments": len(df),
        "window_size": args.window_size,
    }, indent=2))

    # --- Plot ---------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, ycol, ylabel, title in [
        (axes[0], "rolling_mean_dice", "MedSAM dice (rolling mean/median)", f"RAM-H1200 [{args.split}]: MedSAM dice vs. GT fragment area"),
        (axes[1], "rolling_failure_rate", f"MedSAM failure rate (dice < {args.failure_dice_threshold})", f"RAM-H1200 [{args.split}]: MedSAM failure rate vs. GT fragment area"),
    ]:
        x_min, x_max = float(rolling["area"].min()), float(rolling["area"].max())

        for thr, color, lbl in [
            (median_threshold, "#888888", f"RAM-H1200 median ({median_threshold:.0f}px)"),
            (PENGWIN_ABSOLUTE_THRESHOLD, "#1f77b4", f"PENGWIN absolute ({PENGWIN_ABSOLUTE_THRESHOLD:.0f}px)"),
            (derived_threshold, "#FF2B2B", f"RAM-H1200 derived ({derived_threshold:.0f}px)"),
        ]:
            linestyle = "--" if thr == derived_threshold else ":"
            linewidth = 2.2 if thr == derived_threshold else 1.8
            ax.axvline(thr, color=color, linestyle=linestyle, linewidth=linewidth, label=lbl)

        ax.plot(rolling["area"], rolling[ycol], linewidth=2.2, color="black")
        if ycol == "rolling_mean_dice":
            ax.plot(rolling["area"], rolling["rolling_median_dice"], linewidth=1.6, color="gray", linestyle="-.", label="rolling median dice")
            ax.axhline(0.70, color="green", linestyle=":", linewidth=1, alpha=0.6, label="dice=0.70")

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("GT fragment area (px, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.figures_dir / f"{args.split}_rolling_dice_and_failure_rate_vs_area.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"plot saved -> {plot_path}")

    print(f"\n=== derived threshold for --split {args.split}: {derived_threshold:.1f}px ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
