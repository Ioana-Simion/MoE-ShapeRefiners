#!/usr/bin/env python3
"""Prepare RAM-H1200 bone-instance boxes + GT masks for MedSAM inference.

PLAN_B (zero-shot transfer of frozen PENGWIN refinement experts) adapter.
Converts RAM-H1200's per-bone COCO RLE annotations into the same
boxes/*.npy + metadata.jsonl layout that
`prepare_pengwin_xray_boxes_for_medsam_inference.py` produces, plus a
gt_masks/*.npz per image (key "masks", original resolution) since
RAM-H1200 has no PENGWIN-style bit-packed label to decode on the fly.

    <output_root>/<split>/
        boxes/<sample_name>.npy      float32 (N, 4) bbox_xyxy_1024
        gt_masks/<sample_name>.npz   uint8 (N, H, W) key "masks", original res
        metadata.jsonl               one record per image

Bone categories are kept; non-bony COCO categories (soft tissue, metal
implants, IV cannulas, rings) are dropped. See PLAN_B section 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "RAM-H1200-v1_dataset" / "Segmentation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "ramh1200" / "bounding-boxes"

NON_BONE_CATEGORY_NAMES = {
    "bone-6U7D-RuyD",  # supercategory placeholder, id 0
    "SoftTissue",
    "Metal Implant",
    "Intravenous cannula",
    "Ring",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images (debugging).")
    return parser.parse_args()


def decode_rle(segmentation: dict) -> np.ndarray:
    """Decode a COCO RLE segmentation dict into a binary uint8 mask (H, W)."""
    seg = dict(segmentation)
    counts = seg["counts"]
    if isinstance(counts, str):
        seg["counts"] = counts.encode("utf-8")
    return maskUtils.decode(seg).astype(np.uint8)


def bbox_xywh_to_xyxy(bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def scale_boxes_xyxy(boxes: np.ndarray, original_shape: tuple[int, int], image_size: int) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    oh, ow = original_shape
    scale = np.array(
        [image_size / ow, image_size / oh, image_size / ow, image_size / oh],
        dtype=np.float32,
    )
    return boxes.astype(np.float32) * scale


def process_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    image_size: int,
    limit: int | None,
) -> None:
    split_dir = dataset_root / split
    coco_path = split_dir / "_annotations_bone_rle.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"missing COCO annotations: {coco_path}")

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    keep_category_ids = {cid for cid, name in categories.items() if name not in NON_BONE_CATEGORY_NAMES}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in keep_category_ids:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]
    if limit is not None:
        images = images[:limit]

    out_split_dir = output_root / split
    boxes_dir = out_split_dir / "boxes"
    gt_masks_dir = out_split_dir / "gt_masks"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    gt_masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_split_dir / "metadata.jsonl"

    kept = 0
    skipped_empty = 0

    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for image_rec in tqdm(images, desc=f"[{split}] RAM-H1200 boxes"):
            image_id = image_rec["id"]
            file_name = image_rec["file_name"]
            stem = Path(file_name).stem
            h, w = image_rec["height"], image_rec["width"]

            anns = sorted(anns_by_image.get(image_id, []), key=lambda a: a["id"])
            if not anns:
                skipped_empty += 1
                continue

            masks = [decode_rle(a["segmentation"]) for a in anns]
            gt_masks = np.stack(masks, axis=0).astype(np.uint8)  # (N, H, W) original res

            original_boxes = np.stack([bbox_xywh_to_xyxy(a["bbox"]) for a in anns], axis=0)
            resized_boxes = scale_boxes_xyxy(original_boxes, original_shape=(h, w), image_size=image_size)

            sample_name = f"RAMH1200_{split}_{stem}"

            fragments = []
            for i, a in enumerate(anns, start=1):
                fragments.append(
                    {
                        "category_id": a["category_id"],
                        "category_name": categories[a["category_id"]],
                        "fragment_id": a["id"],
                        "medsam_instance_id": i,
                        "bbox_xyxy": original_boxes[i - 1].tolist(),
                        "bbox_xyxy_1024": resized_boxes[i - 1].tolist(),
                    }
                )

            box_path = boxes_dir / f"{sample_name}.npy"
            np.save(box_path, resized_boxes.astype(np.float32))

            gt_masks_path = gt_masks_dir / f"{sample_name}.npz"
            np.savez_compressed(gt_masks_path, masks=gt_masks)

            record = {
                "sample_name": sample_name,
                "case_id": stem,
                "split": split,
                "original_image_path": str((split_dir / file_name).resolve()),
                "image_shape": [h, w],
                "image_size": image_size,
                "num_fragments": len(fragments),
                "fragments": fragments,
                "box_path": str(box_path.resolve()),
                "gt_masks_path": str(gt_masks_path.resolve()),
            }
            meta_f.write(json.dumps(record) + "\n")
            kept += 1

    print(f"[{split}] wrote {kept} records to {metadata_path}")
    if skipped_empty:
        print(f"[{split}] skipped {skipped_empty} images with no bone fragments")


def main() -> int:
    args = parse_args()
    process_split(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        split=args.split,
        image_size=args.image_size,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
