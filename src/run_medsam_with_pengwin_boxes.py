#!/usr/bin/env python3
"""Run MedSAM on PENGWIN X-ray images using precomputed box prompts.

Consumes the dataset produced by `prepare_pengwin_xray_boxes_for_medsam_inference.py`:

    <input_root>/
      boxes/          one .npy per image, shape (N, 4) float32 bbox_xyxy_1024
      metadata.jsonl  original_image_path + fragment metadata per image

Images are loaded on the fly from the original .tif paths recorded in
metadata.jsonl and preprocessed (neglog normalisation → uint8 → resize to
1024×1024×3) before being passed to MedSAM.

For each image the model embeds the image once, then runs the mask decoder once
per bounding box. Outputs are written to:

    <output_root>/
      binary_masks/   one .npz per image, uint8 shape (N, 1024, 1024), key "masks"
      embeddings/     one .npy per image, float16 shape (256, 64, 64)
      metadata.jsonl  paths + fragment metadata per prediction

N is the number of fragments (bounding boxes) for that image.  Slice [i]
corresponds to fragments[i] in the metadata record.

Load masks:    np.load("XRAY_PENGWIN_001_0000.npz")["masks"]
Load embedding: np.load("XRAY_PENGWIN_001_0000.npy").astype(np.float32)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MEDSAM_REPO_ROOT = WORKSPACE_ROOT / "MedSAM"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "bounding-boxes-xrays"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "medsam-predictions"
DEFAULT_CHECKPOINT = MEDSAM_REPO_ROOT / "work_dir" / "MedSAM" / "medsam_vit_b.pth"

if str(MEDSAM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM_REPO_ROOT))

from segment_anything import sam_model_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=tuple(sam_model_registry.keys()),
        help="MedSAM model type for sam_model_registry.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to create binary masks.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional original case id to run, for example '001_0000'.",
    )
    parser.add_argument(
        "--sample-name",
        default=None,
        help="Optional sample name to run, for example 'XRAY_PENGWIN_001_0000'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a smaller smoke test.",
    )
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one case_id per line (e.g. produced by "
            "gating_mechanism/extract_case_ids.py). Restricts inference to "
            "exactly these cases, so regenerated embeddings/masks cover the "
            "same fragments as an existing gated_{split}_records.csv."
        ),
    )
    parser.add_argument(
        "--finetuned-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional fine-tuned MedSAM checkpoint (.pth) saved by train_medsam_pengwin.py. "
            "If provided, the base SAM weights are overwritten with the fine-tuned weights. "
            "Checkpoint format: {'model': state_dict, 'epoch': int, ...}"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in output metadata.jsonl and append to it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess all samples even if output already exists.",
    )
    return parser.parse_args()


def load_records(
    metadata_path: Path,
    case_id: str | None,
    sample_name: str | None,
    limit: int | None,
    case_ids_file: Path | None = None,
) -> list[dict]:
    wanted_case_ids: set[str] | None = None
    if case_ids_file is not None:
        wanted_case_ids = {
            line.strip() for line in case_ids_file.read_text().splitlines() if line.strip()
        }

    records: list[dict] = []
    seen_case_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if case_id is not None and record.get("case_id") != case_id:
                continue
            if sample_name is not None and record.get("sample_name") != sample_name:
                continue
            if wanted_case_ids is not None and record.get("case_id") not in wanted_case_ids:
                continue
            records.append(record)
            seen_case_ids.add(record.get("case_id"))
            if limit is not None and len(records) >= limit:
                break

    if wanted_case_ids is not None:
        missing = wanted_case_ids - seen_case_ids
        if missing and limit is None:
            raise FileNotFoundError(
                f"{len(missing)} case_ids from {case_ids_file} have no matching record in "
                f"{metadata_path}. Example: {sorted(missing)[:5]} — did you run "
                "prepare_pengwin_xray_boxes_for_medsam_inference.py --case-ids-file first?"
            )

    return records


def load_done_sample_names(output_metadata_path: Path) -> set[str]:
    """Return sample names already successfully written to output metadata."""
    done: set[str] = set()
    if not output_metadata_path.exists():
        return done
    with output_metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record.get("sample_name")
            if name:
                done.add(name)
    return done


def load_and_preprocess_image(
    path: str | Path,
    image_size: int = 1024,
    window_lower: float = 0.01,
    window_upper: float = 0.95,
) -> np.ndarray:
    """Load a raw DRR TIF and return a uint8 (image_size, image_size, 3) array.

    Matches the PENGWIN inference pipeline from build_augmentation(train=False):
    neglog → quantile window [window_lower, window_upper] → uint8 → resize.
    """
    image = np.array(Image.open(path)).astype(np.float32)

    # neglog: match neglog_fn — add min+epsilon so relative log-spacing is preserved
    image += image.min() + 1e-3
    image = -np.log(image)

    # Quantile windowing matching pengwin_utils.window(0.01, 0.95).
    # window() passes args to window_() swapped, so the 95th-pct value is
    # subtracted and the denominator is (1st-pct - 95th-pct), which is
    # negative — this inverts the image (air→1/bright, bone→0/dark).
    q_lo = np.quantile(image, window_lower)   # 1st-pct value
    q_hi = np.quantile(image, window_upper)   # 95th-pct value
    if q_lo == q_hi:
        lo, hi = image.min(), image.max()
        image = (image - lo) / (hi - lo + 1e-7)
    else:
        image = (image - q_hi) / (q_lo - q_hi + 1e-7)
    image = np.clip(image, 0.0, 1.0)

    # scale to uint8 and resize
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = np.array(
        Image.fromarray(image).resize((image_size, image_size), Image.BILINEAR)
    )

    # ensure 3-channel (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    return image


@torch.no_grad()
def run_medsam_with_boxes(
    model: torch.nn.Module,
    image: np.ndarray,
    boxes: np.ndarray,
    device: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_masks, embedding).

    binary_masks: uint8 (N, 1024, 1024)
    embedding:    float16 (256, 64, 64) — the ViT-B image embedding for downstream use
    """
    if image.shape != (1024, 1024, 3):
        raise ValueError(f"expected image shape (1024, 1024, 3), got {image.shape}")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"expected boxes shape (N, 4), got {boxes.shape}")

    n = boxes.shape[0]
    use_amp = device.startswith("cuda")

    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(image))
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    ) / 255.0

    with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        image_embedding = model.image_encoder(image_tensor)  # (1, 256, 64, 64)

        if n == 0:
            embedding = image_embedding[0].cpu().half().numpy()
            return np.zeros((0, 1024, 1024), dtype=np.uint8), embedding

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)  # (N, 4)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=boxes_tensor,
            masks=None,
        )
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding.expand(n, -1, -1, -1),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # (N, 1, 256, 256)

        probs = torch.sigmoid(low_res_masks)
        probs = F.interpolate(probs, size=(1024, 1024), mode="bilinear", align_corners=False)

    binary_masks = (probs[:, 0].cpu().float().numpy() > threshold).astype(np.uint8)
    embedding = image_embedding[0].cpu().half().numpy()  # (256, 64, 64) float16
    return binary_masks, embedding


def main() -> int:
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    boxes_root = args.input_root / "boxes"
    input_metadata_path = args.input_root / "metadata.jsonl"

    if not boxes_root.exists():
        raise FileNotFoundError(f"missing boxes directory: {boxes_root}")
    if not input_metadata_path.exists():
        raise FileNotFoundError(f"missing metadata: {input_metadata_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing MedSAM checkpoint: {args.checkpoint}")

    binary_masks_root = args.output_root / "binary_masks"
    embeddings_root = args.output_root / "embeddings"
    binary_masks_root.mkdir(parents=True, exist_ok=True)
    embeddings_root.mkdir(parents=True, exist_ok=True)
    output_metadata_path = args.output_root / "metadata.jsonl"

    records = load_records(
        input_metadata_path,
        case_id=args.case_id,
        sample_name=args.sample_name,
        limit=args.limit,
        case_ids_file=args.case_ids_file,
    )

    if not records:
        raise RuntimeError("no matching samples found in metadata")

    # Resume: filter out samples already recorded in output metadata.
    done: set[str] = set()
    if args.resume:
        done = load_done_sample_names(output_metadata_path)
    elif args.overwrite and output_metadata_path.exists():
        output_metadata_path.unlink()

    pending = [r for r in records if r["sample_name"] not in done]
    skipped = len(records) - len(pending)

    if not pending:
        print(f"nothing to do — all {len(records)} samples already processed")
        return 0

    model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    if args.finetuned_ckpt is not None:
        if not args.finetuned_ckpt.exists():
            raise FileNotFoundError(f"fine-tuned checkpoint not found: {args.finetuned_ckpt}")
        ckpt = torch.load(args.finetuned_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned weights from epoch {ckpt.get('epoch', '?')}: {args.finetuned_ckpt}")
    model = model.to(args.device)
    model.eval()

    written = 0
    metadata_mode = "a" if args.resume else "w"

    def _load_image(rec: dict) -> np.ndarray:
        return load_and_preprocess_image(
            rec["original_image_path"], image_size=rec.get("image_size", 1024)
        )

    with (
        output_metadata_path.open(metadata_mode, encoding="utf-8") as metadata_handle,
        ThreadPoolExecutor(max_workers=1) as loader,
    ):
        prefetch: Future = loader.submit(_load_image, pending[0])

        for i, record in enumerate(tqdm(pending, desc="Running MedSAM")):
            image = prefetch.result()

            # Kick off next image load before GPU work starts.
            if i + 1 < len(pending):
                prefetch = loader.submit(_load_image, pending[i + 1])

            sample_name = record["sample_name"]
            boxes_path = Path(record.get("box_path", boxes_root / f"{sample_name}.npy"))
            binary_masks_path = binary_masks_root / f"{sample_name}.npz"
            embedding_path = embeddings_root / f"{sample_name}.npy"

            if not boxes_path.exists():
                raise FileNotFoundError(f"missing boxes file: {boxes_path}")

            boxes = np.load(boxes_path).astype(np.float32)

            binary_masks, embedding = run_medsam_with_boxes(
                model, image, boxes, args.device, threshold=args.threshold
            )

            np.savez_compressed(binary_masks_path, masks=binary_masks)
            np.save(embedding_path, embedding)

            output_record = {
                "sample_name": sample_name,
                "case_id": record.get("case_id"),
                "original_image_path": record.get("original_image_path"),
                "input_boxes_path": str(boxes_path.resolve()),
                "num_boxes": int(boxes.shape[0]),
                "threshold": args.threshold,
                "binary_masks_path": str(binary_masks_path.resolve()),
                "embedding_path": str(embedding_path.resolve()),
                "fragments": record.get("fragments", []),
            }
            metadata_handle.write(json.dumps(output_record) + "\n")
            metadata_handle.flush()
            written += 1

    print(f"processed {written} samples")
    if skipped:
        print(f"skipped {skipped} already-done samples (--resume)")
    print(f"prediction root: {args.output_root}")
    print(f"binary masks dir: {binary_masks_root}")
    print(f"embeddings dir:   {embeddings_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())