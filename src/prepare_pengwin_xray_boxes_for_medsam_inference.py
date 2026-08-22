#!/usr/bin/env python3
"""Prepare PENGWIN Task 2 X-ray bounding boxes for MedSAM inference.

This script extracts fragment bounding boxes from the PENGWIN segmentation
labels and writes to:

    <output_root>/
        imgs/           empty — populated by downstream MedSAM inference
        boxes/          one .npy per image, shape [N, 4] float32 (bbox_xyxy_1024)
        metadata.jsonl  one record per image

No images are copied or resized — metadata records point back to the original
.tif paths so that the downstream MedSAM inference script can load and resize
on the fly.

Each JSONL record contains:
- case_id, sample_name
- original image and label paths
- original image shape
- box_path: path to the corresponding .npy file in boxes/
- per-fragment bounding boxes in both original-pixel and 1024-grid coordinates
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(
    os.environ.get("SLURM_SUBMIT_DIR", str(Path(__file__).resolve().parents[2]))
)
PENGWIN_ROOT = WORKSPACE_ROOT / "data" / "pengwin"
XRAY_IMAGE_ROOT = (
    PENGWIN_ROOT / "original" / "task2_xray" / "train" / "input" / "images" / "x-ray"
)
XRAY_LABEL_ROOT = (
    PENGWIN_ROOT / "original" / "task2_xray" / "train" / "output" / "images" / "x-ray"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "bounding-boxes-xrays"
)

CATEGORY_NAMES = {1: "SA", 2: "LI", 3: "RI"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=XRAY_IMAGE_ROOT)
    parser.add_argument("--label-root", type=Path, default=XRAY_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--bbox-pad",
        type=int,
        default=0,
        help="Padding in original-image pixels added to each side of each box.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Write records for images with no positive fragments.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N cases (for debugging).",
    )
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one case_id per line (e.g. produced by "
            "gating_mechanism/extract_case_ids.py). Restricts processing to "
            "exactly these cases, so a regenerated dataset covers the same "
            "fragments as an existing gated_{split}_records.csv without "
            "reprocessing the full PENGWIN corpus."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already present in metadata.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute all cases even if metadata.jsonl exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra progress information.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="When --verbose, print a summary every N processed cases.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Segmentation decoding
# ---------------------------------------------------------------------------
def decode_segmentation(
    segmentation: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]]:
    """Decode bit-packed segmentation into per-fragment binary masks."""
    masks: list[np.ndarray] = []
    fragments: list[dict] = []
    for category_id in sorted(CATEGORY_NAMES):
        for fragment_id in range(1, 11):
            shift = 10 * (category_id - 1) + fragment_id
            mask = ((segmentation >> shift) & 1).astype(np.uint8)
            if np.any(mask):
                masks.append(mask)
                fragments.append(
                    {
                        "category_id": category_id,
                        "category_name": CATEGORY_NAMES[category_id],
                        "fragment_id": fragment_id,
                    }
                )
    for label_id, fragment in enumerate(fragments, start=1):
        fragment["medsam_instance_id"] = label_id
    return masks, fragments


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------
def compute_bbox_xyxy(mask: np.ndarray, pad: int = 0) -> np.ndarray:
    """Return [x_min, y_min, x_max, y_max] for a binary mask."""
    y_idx, x_idx = np.where(mask > 0)
    if y_idx.size == 0:
        raise ValueError("cannot compute a bounding box for an empty mask")
    height, width = mask.shape
    x_min = max(0, int(x_idx.min()) - pad)
    y_min = max(0, int(y_idx.min()) - pad)
    x_max = min(width, int(x_idx.max()) + 1 + pad)
    y_max = min(height, int(y_idx.max()) + 1 + pad)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)


def scale_boxes_xyxy(
    boxes: np.ndarray,
    original_shape: tuple[int, int],
    image_size: int,
) -> np.ndarray:
    """Scale boxes from original pixel coords to the resized grid."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    oh, ow = original_shape
    scale = np.array(
        [image_size / ow, image_size / oh, image_size / ow, image_size / oh],
        dtype=np.float32,
    )
    return boxes.astype(np.float32) * scale


# ---------------------------------------------------------------------------
# Case discovery and metadata I/O
# ---------------------------------------------------------------------------
def iter_case_ids(image_root: Path, label_root: Path) -> list[str]:
    image_ids = {p.stem for p in image_root.glob("*.tif")}
    label_ids = {p.stem for p in label_root.glob("*.tif")}
    return sorted(image_ids & label_ids)


def load_existing_case_ids(metadata_path: Path) -> set[str]:
    """Return case_ids already recorded in an existing metadata file."""
    case_ids: set[str] = set()
    if not metadata_path.exists():
        return case_ids
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = record.get("case_id")
            if cid:
                case_ids.add(cid)
    return case_ids


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------
def process_case(
    case_id: str,
    image_root: Path,
    label_root: Path,
    image_size: int,
    bbox_pad: int,
) -> dict:
    image_path = image_root / f"{case_id}.tif"
    label_path = label_root / f"{case_id}.tif"

    image = np.array(Image.open(image_path))
    segmentation = np.array(Image.open(label_path))
    masks, fragments = decode_segmentation(segmentation)

    original_boxes = (
        np.stack([compute_bbox_xyxy(m, pad=bbox_pad) for m in masks], axis=0)
        if masks
        else np.zeros((0, 4), dtype=np.int32)
    )
    resized_boxes = scale_boxes_xyxy(
        original_boxes, original_shape=image.shape[:2], image_size=image_size
    )

    fragment_records = []
    for frag, obox, rbox in zip(
        fragments, original_boxes.tolist(), resized_boxes.tolist()
    ):
        rec = dict(frag)
        rec["bbox_xyxy"] = obox
        rec["bbox_xyxy_1024"] = rbox
        fragment_records.append(rec)

    return {
        "sample_name": f"XRAY_PENGWIN_{case_id}",
        "case_id": case_id,
        "original_image_path": str(image_path.resolve()),
        "original_label_path": str(label_path.resolve()),
        "image_shape": list(image.shape),
        "image_size": image_size,
        "bbox_pad": bbox_pad,
        "num_fragments": len(fragments),
        "fragments": fragment_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    imgs_dir = args.output_root / "imgs"
    boxes_dir = args.output_root / "boxes"
    imgs_dir.mkdir(exist_ok=True)
    boxes_dir.mkdir(exist_ok=True)
    metadata_path = args.output_root / "metadata.jsonl"

    case_ids = iter_case_ids(args.image_root, args.label_root)

    if args.case_ids_file is not None:
        wanted = {
            line.strip() for line in args.case_ids_file.read_text().splitlines() if line.strip()
        }
        missing = wanted - set(case_ids)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} case_ids from {args.case_ids_file} have no matching "
                f"image+label pair under --image-root/--label-root. Example: {sorted(missing)[:5]}"
            )
        case_ids = sorted(wanted)
        if args.verbose:
            print(f"--case-ids-file: restricting to {len(case_ids)} cases from {args.case_ids_file}")

    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    # ---- resume filtering ------------------------------------------------
    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = load_existing_case_ids(metadata_path)
    elif args.overwrite and metadata_path.exists():
        metadata_path.unlink()

    pending = [cid for cid in case_ids if cid not in existing_ids]
    skipped_existing = len(case_ids) - len(pending)

    if args.verbose:
        print(f"total matched cases: {len(case_ids)}")
        print(f"pending cases:       {len(pending)}")
        print(f"skipped (resume):    {skipped_existing}")
        print(f"output:              {metadata_path}")

    # ---- process ---------------------------------------------------------
    metadata_mode = "a" if args.resume else "w"
    kept = 0
    skipped_empty = 0

    with metadata_path.open(metadata_mode, encoding="utf-8") as f:
        for case_id in tqdm(pending, desc="Extracting PENGWIN X-ray boxes"):
            record = process_case(
                case_id=case_id,
                image_root=args.image_root,
                label_root=args.label_root,
                image_size=args.image_size,
                bbox_pad=args.bbox_pad,
            )
            if not record["fragments"] and not args.keep_empty:
                skipped_empty += 1
                continue
            boxes = np.array(
                [frag["bbox_xyxy_1024"] for frag in record["fragments"]],
                dtype=np.float32,
            )
            box_path = boxes_dir / f"{record['sample_name']}.npy"
            np.save(box_path, boxes)
            record["box_path"] = str(box_path)
            f.write(json.dumps(record) + "\n")
            f.flush()
            kept += 1
            if args.verbose and kept % args.log_every == 0:
                print(f"progress: {kept}/{len(pending)}, {skipped_empty} empty skipped")

    # ---- summary ---------------------------------------------------------
    print(f"wrote {kept} records to {metadata_path}")
    if skipped_existing:
        print(f"skipped {skipped_existing} existing cases (--resume)")
    if skipped_empty:
        print(f"skipped {skipped_empty} empty-label cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())