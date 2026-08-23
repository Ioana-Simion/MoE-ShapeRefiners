#!/usr/bin/env python3
"""Post-hoc gate analysis for the learned soft-gate MoE (PLAN_C_EVAL §3).

The gate was never given size/shape descriptors — it only saw the pooled
MedSAM embedding + mask + SDF (see cnnMoE.Gate). This asks, after the fact:
did it nonetheless learn a size- or shape-aligned partition?

Joins:
    gate_weights_{split}.csv (from infer_moe_learned.py)
        + evaluate_medsam_pengwin.py's hard-mode output CSV
          (on sample_name, fragment_index)              -> dice, gt_area, size_group
        + data_distribution features.csv (optional)
          (on case_id, medsam_instance_id, fragment_id)  -> shape descriptors

Produces:
    gate_analysis_summary.json  -- all numeric findings from §3.1-§3.5
    gate_weight_vs_area.png     -- scatter, argmax/g1 vs area
    gate_confidence_hist.png    -- histogram of max(g0, g1)

Usage:
    python gate_analysis.py \\
        --gate-weights data/moe-learned-predictions/gate_weights_test.csv \\
        --eval-csv     data/moe-learned-predictions/evaluation_hard.csv \\
        --output-dir   data/moe-learned-predictions/gate_analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gating_mechanism.gating_mechanism import (   # noqa: E402
    DEFAULT_FEATURES_CSV, DEFAULT_THRESHOLD, LEARNED_FEATURE_COLS, LEARNED_JOIN_KEYS,
    route_to_expert,
)

SHAPE_COLS = [c for c in LEARNED_FEATURE_COLS if c != "area"]  # area handled separately (size, not shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate-weights", type=Path, required=True,
                        help="gate_weights_{split}.csv from infer_moe_learned.py.")
    parser.add_argument("--eval-csv", type=Path, required=True,
                        help="evaluate_medsam_pengwin.py output for the HARD-mode predictions "
                             "(adds dice/gt_area/size_group/bbox_area_1024/category_name).")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV,
                        help="Shape-descriptor CSV (data_distribution/analyze_dataset.py output). "
                             "Optional -- shape correlation (§3.2) is skipped with a warning if missing.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def point_biserial(binary: np.ndarray, continuous: np.ndarray) -> tuple[float, float]:
    """Correlation between a 0/1 group label and a continuous variable. Returns (r, p)."""
    mask = ~np.isnan(continuous)
    if mask.sum() < 3 or len(np.unique(binary[mask])) < 2:
        return float("nan"), float("nan")
    r, p = stats.pointbiserialr(binary[mask], continuous[mask])
    return float(r), float(p)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gw = pd.read_csv(args.gate_weights)
    ev = pd.read_csv(args.eval_csv)
    print(f"gate_weights: {len(gw)} rows, eval_csv: {len(ev)} rows")

    merged = gw.merge(
        ev[["sample_name", "fragment_index", "dice", "iou", "gt_area", "bbox_area_1024",
            "category_name", "size_group"]],
        on=["sample_name", "fragment_index"],
        how="inner",
        suffixes=("", "_eval"),
    )
    if len(merged) != len(gw):
        print(f"WARNING: {len(gw) - len(merged)} gate-weight rows had no matching eval row "
              f"(sample_name, fragment_index) -- were gate_weights and eval_csv built from the "
              f"SAME split/run? Proceeding with the {len(merged)} matched rows only.")

    category_name = merged["category_name"].where(merged["category_name"].notna(), merged["category_name_eval"]) \
        if "category_name_eval" in merged.columns else merged["category_name"]
    merged["category_name"] = category_name

    summary: dict = {"n_fragments": int(len(merged))}

    # --- §3.1 Size correlation -------------------------------------------------
    argmax = merged["argmax_expert"].to_numpy()
    for area_col in ("medsam_area", "gt_area", "bbox_area_1024"):
        if area_col not in merged.columns:
            continue
        area = merged[area_col].to_numpy(dtype=float)
        r_argmax, p_argmax = point_biserial(argmax, area)
        r_g1, p_g1 = stats.spearmanr(merged["g1"], area, nan_policy="omit")
        mean_by_group = merged.groupby("argmax_expert")[area_col].mean().to_dict()
        summary[f"size_correlation_{area_col}"] = {
            "point_biserial_argmax_vs_area_r": r_argmax,
            "point_biserial_argmax_vs_area_p": p_argmax,
            "spearman_g1_vs_area_r": float(r_g1),
            "spearman_g1_vs_area_p": float(p_g1),
            "mean_area_by_expert": {str(k): float(v) for k, v in mean_by_group.items()},
        }

    # Does argmax_expert recover the ORIGINAL area-threshold ROUTING rule's
    # split? Deliberately NOT using the eval CSV's "size_group" here -- that's
    # a bbox-area threshold computed for METRIC REPORTING (evaluate_medsam_
    # pengwin.py), a different quantity from the foreground-pixel-count
    # threshold (DEFAULT_THRESHOLD=5402 on medsam_area) that actually decided
    # expert_small/expert_large routing for the real rule-based cnnNoROI
    # model (see gating_mechanism.py). medsam_area is already the exact same
    # column gating_mechanism.py computed that routing decision from, so the
    # true rule label is re-derived directly here instead of proxying through
    # a different threshold.
    if "medsam_area" in merged.columns:
        rule_label = merged["medsam_area"].apply(route_to_expert)
        rule_small = (rule_label == "expert_small").astype(int)
        # Whichever expert index is more common among rule-small fragments is
        # the "learned analogue" of expert_small, for a same-direction comparison.
        analogue_small = int(merged.loc[rule_small == 1, "argmax_expert"].mode().iat[0]) \
            if (rule_small == 1).any() else None
        if analogue_small is not None:
            learned_small = (merged["argmax_expert"] == analogue_small).astype(int)
            agreement = float((learned_small == rule_small).mean())
            true_small_frac = float(rule_small.mean())
            # Trivial baseline: always guess the majority class under the true
            # rule (here, almost always "large"), never even looking at the
            # fragment. If raw agreement is BELOW this, the gate's routing
            # imbalance (see balance_check) is costing more agreement than a
            # baseline that does no work at all -- essential context, since
            # raw agreement alone reads as "pretty good" without it.
            majority_baseline = float(max(true_small_frac, 1.0 - true_small_frac))
            summary["agreement_with_rule_based_split"] = {
                "routing_threshold_px": DEFAULT_THRESHOLD,
                "learned_expert_analogous_to_expert_small": analogue_small,
                "true_fraction_small_under_rule": true_small_frac,
                "trivial_always_majority_class_baseline": majority_baseline,
                "fraction_agreeing_with_area_threshold_rule": agreement,
                "agreement_minus_trivial_baseline": agreement - majority_baseline,
            }

    # --- §3.2 Shape correlation --------------------------------------------
    if args.features_csv.exists():
        features = pd.read_csv(args.features_csv)
        shape_merge = merged.merge(
            features[LEARNED_JOIN_KEYS + SHAPE_COLS], on=LEARNED_JOIN_KEYS, how="left",
        )
        shape_summary = {}
        for col in SHAPE_COLS:
            if col not in shape_merge.columns:
                continue
            vals = shape_merge[col].to_numpy(dtype=float)
            r_argmax, p_argmax = point_biserial(shape_merge["argmax_expert"].to_numpy(), vals)
            shape_summary[col] = {"point_biserial_argmax_r": r_argmax, "point_biserial_argmax_p": p_argmax}
        summary["shape_correlation"] = shape_summary
    else:
        print(f"WARNING: features CSV not found at {args.features_csv} -- skipping §3.2 shape correlation. "
              "Run data_distribution/analyze_dataset.py --mask-source medsam first if you want this.")
        summary["shape_correlation"] = None

    # --- §3.3 Anatomical correlation ----------------------------------------
    crosstab = pd.crosstab(merged["category_name"], merged["argmax_expert"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(crosstab) if crosstab.shape[0] > 1 and crosstab.shape[1] > 1 else (float("nan"),) * 4
    summary["anatomical_correlation"] = {
        "crosstab": {str(k): {str(k2): int(v2) for k2, v2 in v.items()} for k, v in crosstab.to_dict(orient="index").items()},
        "chi2": float(chi2) if not isinstance(chi2, tuple) else None,
        "chi2_p": float(chi2_p) if not isinstance(chi2_p, tuple) else None,
    }

    # --- §3.4 Confidence -----------------------------------------------------
    max_gate = merged[["g0", "g1"]].max(axis=1)
    summary["gate_confidence"] = {
        "mean": float(max_gate.mean()),
        "median": float(max_gate.median()),
        "fraction_above_0.9": float((max_gate > 0.9).mean()),
        "fraction_below_0.6": float((max_gate < 0.6).mean()),
    }

    # --- §3.5 Balance check ----------------------------------------------------
    routed_counts = merged["argmax_expert"].value_counts().to_dict()
    total = len(merged)
    summary["balance_check"] = {
        "routed_counts": {str(k): int(v) for k, v in routed_counts.items()},
        "routed_fractions": {str(k): float(v / total) for k, v in routed_counts.items()},
    }
    min_frac = min(summary["balance_check"]["routed_fractions"].values())
    if min_frac < 0.15:
        print(f"WARNING: at inference, one expert receives only {min_frac:.1%} of hard-routed fragments "
              f"-- possible near-collapse despite load-balancing during training.")

    # --- Plots ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "gt_area" in merged.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            for k in sorted(merged["argmax_expert"].unique()):
                subset = merged[merged["argmax_expert"] == k]
                ax.scatter(subset["gt_area"], subset["g1"], s=4, alpha=0.4, label=f"argmax=expert_{k}")
            ax.set_xscale("log")
            ax.set_xlabel("GT fragment area (px, log scale)")
            ax.set_ylabel("g1 (gate weight toward expert_1)")
            ax.set_title("Gate weight vs. fragment size")
            ax.legend()
            fig.savefig(args.output_dir / "gate_weight_vs_area.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(max_gate, bins=40, range=(0.5, 1.0))
        ax.set_xlabel("max(g0, g1) — gate confidence")
        ax.set_ylabel("count")
        ax.set_title("Gate confidence distribution")
        fig.savefig(args.output_dir / "gate_confidence_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to {args.output_dir}")
    except Exception as exc:
        print(f"WARNING: could not save plots ({exc})")

    out_json = args.output_dir / "gate_analysis_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: {out_json}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
