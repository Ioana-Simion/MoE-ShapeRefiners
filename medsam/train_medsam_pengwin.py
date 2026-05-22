#!/usr/bin/env python3
"""Fine-tune MedSAM on PENGWIN Task 2 X-ray data.

Uses the same train/val/test splits and expert-balancing logic as the CNN and
FlowSDF experiments to ensure a fair comparison.

Data flow:
  gated_{split}_records.csv  (same splits as CNN / FlowSDF)
        ↓   join on sample_name
  data/bounding-boxes-xrays/metadata.jsonl  (bbox_xyxy_1024, image/label paths)
        ↓
  raw .tif X-ray  →  neglog preprocess  →  MedSAM image encoder
  raw .tif label  →  bit-decode fragment →  GT binary mask

The 'expert_large' subset is subsampled to preserve SA/LI/RI class balance,
identical to the CNN training (--large-subsample).  Both expert subsets are
then merged into a single dataloader for MedSAM (single model, no routing).

Trains image encoder + mask decoder; prompt encoder is frozen.
Loss = Dice + BCE.  Best checkpoint saved by validation loss.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import monai

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MEDSAM_SRC = Path(__file__).resolve().parent / "MedSAM-main"
SRC_DIR = REPO_ROOT / "src"

DEFAULT_GATED_CSV_DIR = SRC_DIR / "gating_mechanism"
DEFAULT_BOX_METADATA = REPO_ROOT / "data" / "bounding-boxes-xrays" / "metadata.jsonl"
DEFAULT_CHECKPOINT = MEDSAM_SRC / "medsam_vit_b.pth"
DEFAULT_WORK_DIR = REPO_ROOT / "medsam_pengwin"

if str(MEDSAM_SRC) not in sys.path:
    sys.path.insert(0, str(MEDSAM_SRC))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segment_anything import sam_model_registry  # noqa: E402
from dataloader_utils import (  # noqa: E402
    load_pengwin_label,
    decode_pengwin_fragment_from_record,
    resize_binary_nearest,
    resolve_existing_path,
    load_jsonl,
)


# ---------------------------------------------------------------------------
# Image preprocessing  (mirrors run_medsam_with_pengwin_boxes.py)
# ---------------------------------------------------------------------------
def preprocess_xray(path: str | Path, image_size: int = 1024) -> np.ndarray:
    """Load a PENGWIN DRR .tif → float32 (image_size, image_size, 3) in [0, 1]."""
    from PIL import Image as PILImage

    image = np.array(PILImage.open(path)).astype(np.float32)

    image += image.min() + 1e-3
    image = -np.log(image)

    q_lo = np.quantile(image, 0.01)
    q_hi = np.quantile(image, 0.95)
    if q_lo == q_hi:
        lo, hi = image.min(), image.max()
        image = (image - lo) / (hi - lo + 1e-7)
    else:
        image = (image - q_hi) / (q_lo - q_hi + 1e-7)
    image = np.clip(image, 0.0, 1.0)

    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    image_3c = np.stack([image_uint8, image_uint8, image_uint8], axis=-1) if image_uint8.ndim == 2 else image_uint8

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
# Dataset
# ---------------------------------------------------------------------------
def load_box_metadata_map(path: str | Path) -> dict[str, dict]:
    """Load bounding-box metadata.jsonl → {sample_name: record}."""
    path = resolve_existing_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Box metadata not found: {path}")
    return {rec["sample_name"]: rec for rec in load_jsonl(path)}


class PengwinGatedDataset(Dataset):
    """Fragment-level dataset backed by the gated CSV splits.

    Loads raw X-ray images and GT fragment masks on the fly.  Uses bbox
    coordinates from the box metadata JSONL — no pre-converted .npy files.
    """

    def __init__(
        self,
        records: list[dict],
        box_meta: dict[str, dict],
        bbox_shift: int = 20,
        image_size: int = 1024,
    ):
        self.records = [r for r in records if r["sample_name"] in box_meta]
        missing = len(records) - len(self.records)
        if missing:
            print(f"[dataset] WARNING: {missing} records had no box metadata entry — skipped")
        self.box_meta = box_meta
        self.bbox_shift = bbox_shift
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        sample_name = rec["sample_name"]
        meta = self.box_meta[sample_name]

        # Locate the bbox for this specific fragment
        target_id = int(rec["medsam_instance_id"])
        frag_meta = next(
            (f for f in meta["fragments"] if int(f["medsam_instance_id"]) == target_id),
            None,
        )
        if frag_meta is None:
            raise KeyError(f"medsam_instance_id {target_id} not found in {sample_name}")
        bbox = np.array(frag_meta["bbox_xyxy_1024"], dtype=np.float32)

        # Random bbox perturbation (training only; pass bbox_shift=0 for val)
        if self.bbox_shift > 0:
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0.0, x_min - random.randint(0, self.bbox_shift))
            y_min = max(0.0, y_min - random.randint(0, self.bbox_shift))
            x_max = min(float(self.image_size), x_max + random.randint(0, self.bbox_shift))
            y_max = min(float(self.image_size), y_max + random.randint(0, self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # Load and preprocess image
        image_path = resolve_existing_path(meta["original_image_path"])
        image = preprocess_xray(image_path, image_size=self.image_size)  # (H, W, 3) float32

        # Load and decode GT fragment
        label_path = resolve_existing_path(meta["original_label_path"])
        segmentation = load_pengwin_label(label_path)
        gt_mask = decode_pengwin_fragment_from_record(segmentation, rec)  # (H, W) uint8
        gt_resized = resize_binary_nearest(gt_mask, (self.image_size, self.image_size))

        return (
            torch.tensor(image.transpose(2, 0, 1)).float(),   # (3, 1024, 1024)
            torch.tensor(gt_resized[None, :, :]).long(),       # (1, 1024, 1024)
            torch.tensor(bbox).float(),                        # (4,)
            f"{sample_name}_frag{target_id}",
        )


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    """Subsample n records preserving the SA/LI/RI class distribution."""
    return (
        df.groupby("category_name", group_keys=False)
        .apply(lambda x: x.sample(frac=n / len(df), random_state=seed))
        .reset_index(drop=True)
        .to_dict("records")
    )


def build_medsam_dataloader(
    gated_csv_dir: Path,
    box_meta: dict[str, dict],
    split: str,
    batch_size: int,
    num_workers: int,
    large_subsample: int | None = None,
    bbox_shift: int = 20,
    image_size: int = 1024,
) -> DataLoader:
    """Build a dataloader for a split using the gated CSV, mirroring build_expert_dataloaders.

    Both experts are merged into one dataset for single-model MedSAM training.
    expert_large is subsampled with the same stratified logic as the CNN experiments.
    """
    csv_path = gated_csv_dir / f"gated_{split}_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run gating_mechanism.py first.")

    df = pd.read_csv(csv_path)
    all_records: list[dict] = []

    for expert in ["expert_small", "expert_large"]:
        subset = df[df["expert"] == expert]
        if expert == "expert_large" and large_subsample is not None:
            n = min(large_subsample, len(subset))
            records = stratified_subsample(subset, n=n)
        else:
            records = subset.to_dict("records")
        print(f"  [{split}] {expert}: {len(records)} fragments")
        all_records.extend(records)

    print(f"  [{split}] total: {len(all_records)} fragments")

    dataset = PengwinGatedDataset(
        all_records, box_meta, bbox_shift=bbox_shift, image_size=image_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, box: np.ndarray) -> torch.Tensor:
        # Run image encoder on the full batch (the expensive op).
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # The standard SAM mask decoder repeats the image embedding by the number
        # of prompts, so it only handles (1 image, N boxes) — not (B images, B boxes).
        # Process prompt encoder + mask decoder one sample at a time to stay compatible.
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
        if box_torch.ndim == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        masks_list = []
        for i in range(image.shape[0]):
            with torch.no_grad():
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=None, boxes=box_torch[i : i + 1], masks=None
                )
            low_res_mask, _ = self.mask_decoder(
                image_embeddings=image_embedding[i : i + 1],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            masks_list.append(low_res_mask)

        low_res_masks = torch.cat(masks_list, dim=0)  # (B, 1, 256, 256)
        return F.interpolate(low_res_masks, size=image.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------
def run_epoch(
    model: MedSAM,
    loader: DataLoader,
    seg_loss,
    ce_loss,
    optimizer,
    device: str,
    use_amp: bool,
    scaler,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    n_steps = 0

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for image, gt2D, boxes, _ in loader:
            boxes_np = boxes.detach().cpu().numpy()
            image = image.to(device)
            gt2D = gt2D.to(device)

            if train:
                optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(image, boxes_np)
                    loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                pred = model(image, boxes_np)
                loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1

    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Data
    parser.add_argument("--gated-csv-dir", type=Path, default=DEFAULT_GATED_CSV_DIR,
                        help="Directory containing gated_{train,val,test}_records.csv")
    parser.add_argument("--box-metadata", type=Path, default=DEFAULT_BOX_METADATA,
                        help="metadata.jsonl from prepare_pengwin_xray_boxes_for_medsam_inference.py")
    parser.add_argument("--large-subsample", type=int, default=None,
                        help="Subsample expert_large to this many fragments (SA/LI/RI proportions "
                             "preserved). Match the value used in CNN/FlowSDF experiments.")
    # Model
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="MedSAM pretrained checkpoint (medsam_vit_b.pth).")
    parser.add_argument("--model-type", type=str, default="vit_b")
    # Training
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--task-name", type=str, default="MedSAM-PENGWIN")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bbox-shift", type=int, default=20)
    parser.add_argument("--use-amp", action="store_true", default=False)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if not args.checkpoint.exists():
        print(
            f"ERROR: MedSAM checkpoint not found at {args.checkpoint}\n"
            "Download medsam_vit_b.pth from "
            "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN"
            f" and place it at {args.checkpoint}"
        )
        return 1

    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.task_name, config=vars(args))

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    save_dir = args.work_dir / f"{args.task_name}-{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, save_dir / f"{run_id}_{Path(__file__).name}")

    device = torch.device(args.device)

    print("Loading box metadata...")
    box_meta = load_box_metadata_map(args.box_metadata)
    print(f"  {len(box_meta)} samples in box metadata")

    print("Building dataloaders...")
    train_loader = build_medsam_dataloader(
        args.gated_csv_dir, box_meta, "train",
        args.batch_size, args.num_workers,
        large_subsample=args.large_subsample,
        bbox_shift=args.bbox_shift,
    )
    val_loader = build_medsam_dataloader(
        args.gated_csv_dir, box_meta, "val",
        args.batch_size, args.num_workers,
        large_subsample=None,   # no subsampling for val — evaluate on everything
        bbox_shift=0,
    )

    print("Loading MedSAM model...")
    sam_model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    trainable = list(model.image_encoder.parameters()) + list(model.mask_decoder.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in trainable)
    print(f"Total params: {total_p:,}  |  Trainable: {train_p:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Resumed from epoch {ckpt['epoch']}: {args.resume}")

    train_losses, val_losses = [], []
    best_val_loss = 1e10

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = run_epoch(
            model, train_loader, seg_loss, ce_loss, optimizer,
            args.device, args.use_amp, scaler, train=True,
        )
        val_loss = run_epoch(
            model, val_loader, seg_loss, ce_loss, optimizer,
            args.device, args.use_amp, scaler, train=False,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ts = datetime.now().strftime("%Y%m%d-%H%M")
        print(f"[{ts}] Epoch {epoch:04d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if args.use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(ckpt, save_dir / "medsam_pengwin_latest.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "medsam_pengwin_best.pth")
            print(f"  -> new best val loss: {best_val_loss:.4f}")

        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.title("Dice + BCE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_dir / f"{args.task_name}_loss.png", bbox_inches="tight", dpi=150)
        plt.close()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
