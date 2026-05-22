#!/usr/bin/env python3
"""Prepare PENGWIN Task 2 X-ray data as .npy files for MedSAM fine-tuning.

Produces the directory layout expected by MedSAM's NpyDataset:

    <output_root>/
        imgs/   {case_id}.npy   float32 (1024, 1024, 3) in [0, 1]
        gts/    {case_id}.npy   uint8   (1024, 1024) with instance labels

Image preprocessing matches the MedSAM inference pipeline:
  neglog → quantile window [0.01, 0.95] → [0, 1] float32 → resize 1024×1024.

GT instance labels: 1…N per fragment, 0 = background.  MedSAM's NpyDataset
randomly selects one fragment per training step, so all fragments are covered
across an epoch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import transform
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(os.environ.get("SLURM_SUBMIT_DIR", str(Path(__file__).resolve().parents[2])))
PENGWIN_ROOT = WORKSPACE_ROOT / "data" / "pengwin"
XRAY_IMAGE_ROOT = PENGWIN_ROOT / "original" / "task2_xray" / "train" / "input" / "images" / "x-ray"
XRAY_LABEL_ROOT = PENGWIN_ROOT / "original" / "task2_xray" / "train" / "output" / "images" / "x-ray"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "npy" / "pengwin_xray"

CATEGORY_NAMES = {1: "SA", 2: "LI", 3: "RI"}
IMAGE_SIZE = 1024


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors run_medsam_with_pengwin_boxes.py)
# ---------------------------------------------------------------------------
def preprocess_xray(
    path: str | Path,
    image_size: int = IMAGE_SIZE,
    window_lower: float = 0.01,
    window_upper: float = 0.95,
) -> np.ndarray:
    """Return float32 (image_size, image_size, 3) in [0, 1]."""
    image = np.array(Image.open(path)).astype(np.float32)

    image += image.min() + 1e-3
    image = -np.log(image)

    q_lo = np.quantile(image, window_lower)
    q_hi = np.quantile(image, window_upper)
    if q_lo == q_hi:
        lo, hi = image.min(), image.max()
        image = (image - lo) / (hi - lo + 1e-7)
    else:
        image = (image - q_hi) / (q_lo - q_hi + 1e-7)
    image = np.clip(image, 0.0, 1.0)

    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    if image_uint8.ndim == 2:
        image_3c = np.stack([image_uint8, image_uint8, image_uint8], axis=-1)
    else:
        image_3c = image_uint8

    resized = transform.resize(
        image_3c,
        (image_size, image_size),
        order=1,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True,
    ).astype(np.float32) / 255.0

    return resized  # (H, W, 3) float32 [0, 1]


# ---------------------------------------------------------------------------
# Segmentation decoding (mirrors prepare_pengwin_xray_boxes_for_medsam_inference.py)
# ---------------------------------------------------------------------------
def decode_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Return uint8 (H, W) instance label map; 0 = background."""
    instance_map = np.zeros(segmentation.shape, dtype=np.uint8)
    instance_id = 1
    for category_id in sorted(CATEGORY_NAMES):
        for fragment_id in range(1, 11):
            shift = 10 * (category_id - 1) + fragment_id
            mask = ((segmentation >> shift) & 1).astype(bool)
            if np.any(mask):
                instance_map[mask] = instance_id
                instance_id += 1
    return instance_map


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------
def iter_case_ids(image_root: Path, label_root: Path) -> list[str]:
    image_ids = {p.stem for p in image_root.glob("*.tif")}
    label_ids = {p.stem for p in label_root.glob("*.tif")}
    return sorted(image_ids & label_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=XRAY_IMAGE_ROOT)
    parser.add_argument("--label-root", type=Path, default=XRAY_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of cases to reserve for validation (written to val/ subdirs).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    case_ids = iter_case_ids(args.image_root, args.label_root)
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    indices = rng.permutation(len(case_ids))
    n_val = max(1, int(len(case_ids) * args.val_split))
    val_set = {case_ids[i] for i in indices[:n_val]}
    train_set = {case_ids[i] for i in indices[n_val:]}

    splits = {"train": train_set, "val": val_set}
    for split, ids in splits.items():
        (args.output_root / split / "imgs").mkdir(parents=True, exist_ok=True)
        (args.output_root / split / "gts").mkdir(parents=True, exist_ok=True)

    total_skipped = 0
    for split, ids in splits.items():
        imgs_dir = args.output_root / split / "imgs"
        gts_dir = args.output_root / split / "gts"

        for case_id in tqdm(sorted(ids), desc=f"Processing {split}"):
            img_out = imgs_dir / f"{case_id}.npy"
            gt_out = gts_dir / f"{case_id}.npy"

            if not args.overwrite and img_out.exists() and gt_out.exists():
                total_skipped += 1
                continue

            image_path = args.image_root / f"{case_id}.tif"
            label_path = args.label_root / f"{case_id}.tif"

            img_array = preprocess_xray(image_path, image_size=args.image_size)

            segmentation = np.array(Image.open(label_path))
            instance_map = decode_segmentation(segmentation)

            gt_resized = transform.resize(
                instance_map,
                (args.image_size, args.image_size),
                order=0,  # nearest-neighbor for labels
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            ).astype(np.uint8)

            if not np.any(gt_resized > 0):
                # skip images with no positive fragments
                total_skipped += 1
                continue

            np.save(img_out, img_array)
            np.save(gt_out, gt_resized)

    print(f"Done. Output: {args.output_root}")
    if total_skipped:
        print(f"Skipped {total_skipped} cases (empty or already processed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
