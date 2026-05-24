#!/usr/bin/env python3
"""Fine-tune MedSAM on PENGWIN Task 2 X-ray data (single-GPU and DDP).

Uses the same train/val/test splits and expert-balancing logic as the CNN and
FlowSDF experiments to ensure a fair comparison.

Data flow:
  gated_{split}_records.csv  (same splits as CNN / FlowSDF)
        ↓   join on sample_name
  data/bounding-boxes-xrays/metadata.jsonl  (bbox_xyxy_1024, image/label paths)
        ↓
  raw .tif X-ray  →  neglog preprocess  →  MedSAM image encoder
  raw .tif label  →  bit-decode fragment →  GT binary mask

Launch (single GPU):
    python train_medsam_pengwin.py [OPTIONS]

Launch (multi-GPU, one node):
    torchrun --nproc_per_node=4 train_medsam_pengwin.py [OPTIONS]

Launch (SLURM, one node):
    srun --ntasks=4 --gpus-per-task=1 --cpus-per-task=6 \\
        torchrun --nproc_per_node=4 train_medsam_pengwin.py [OPTIONS]
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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
    path = resolve_existing_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Box metadata not found: {path}")
    return {rec["sample_name"]: rec for rec in load_jsonl(path)}


class PengwinGatedDataset(Dataset):
    """Fragment-level dataset backed by the gated CSV splits."""

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

        target_id = int(rec["medsam_instance_id"])
        frag_meta = next(
            (f for f in meta["fragments"] if int(f["medsam_instance_id"]) == target_id),
            None,
        )
        if frag_meta is None:
            raise KeyError(f"medsam_instance_id {target_id} not found in {sample_name}")
        bbox = np.array(frag_meta["bbox_xyxy_1024"], dtype=np.float32)

        if self.bbox_shift > 0:
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0.0, x_min - random.randint(0, self.bbox_shift))
            y_min = max(0.0, y_min - random.randint(0, self.bbox_shift))
            x_max = min(float(self.image_size), x_max + random.randint(0, self.bbox_shift))
            y_max = min(float(self.image_size), y_max + random.randint(0, self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        image_path = resolve_existing_path(meta["original_image_path"])
        image = preprocess_xray(image_path, image_size=self.image_size)

        label_path = resolve_existing_path(meta["original_label_path"])
        segmentation = load_pengwin_label(label_path)
        gt_mask = decode_pengwin_fragment_from_record(segmentation, rec)
        gt_resized = resize_binary_nearest(gt_mask, (self.image_size, self.image_size))

        return (
            torch.tensor(image.transpose(2, 0, 1)).float(),   # (3, 1024, 1024)
            torch.tensor(gt_resized[None, :, :]).float(),      # (1, 1024, 1024)
            torch.tensor(bbox).float(),                        # (4,)
            f"{sample_name}_frag{target_id}",
        )


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    return (
        df.groupby("category_name", group_keys=False)
        .apply(lambda x: x.sample(frac=n / len(df), random_state=seed), include_groups=False)
        .reset_index(drop=True)
        .to_dict("records")
    )


def build_medsam_dataset(
    gated_csv_dir: Path,
    box_meta: dict[str, dict],
    split: str,
    large_subsample: int | None = None,
    bbox_shift: int = 20,
    image_size: int = 1024,
    is_main: bool = True,
) -> PengwinGatedDataset:
    """Build a PengwinGatedDataset for a split. Separated from DataLoader creation for DDP."""
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
        if is_main:
            print(f"  [{split}] {expert}: {len(records)} fragments")
        all_records.extend(records)

    if is_main:
        print(f"  [{split}] total: {len(all_records)} fragments")

    return PengwinGatedDataset(all_records, box_meta, bbox_shift=bbox_shift, image_size=image_size)


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
    model: nn.Module,
    loader: DataLoader,
    seg_loss,
    ce_loss,
    optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
    train: bool,
    use_wandb: bool = False,
    global_step: int = 0,
    log_every: int = 50,
    ddp: bool = False,
) -> tuple[float, int]:
    model.train(train)
    total_loss = 0.0
    n_steps = 0

    phase = "train" if train else "val"
    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        pbar = tqdm(loader, desc=phase, leave=False, dynamic_ncols=True)
        for image, gt2D, boxes, _ in pbar:
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
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / n_steps:.4f}")

            if use_wandb and train and global_step % log_every == 0:
                import wandb
                wandb.log({"train/step_loss": loss.item(), "train/avg_loss": total_loss / n_steps}, step=global_step)

    avg_loss = total_loss / max(n_steps, 1)

    # Average loss across all ranks so the logged value is meaningful.
    if ddp:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss, global_step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Data
    parser.add_argument("--gated-csv-dir", type=Path, default=DEFAULT_GATED_CSV_DIR)
    parser.add_argument("--box-metadata", type=Path, default=DEFAULT_BOX_METADATA)
    parser.add_argument("--large-subsample", type=int, default=None,
                        help="Subsample expert_large to this many fragments (SA/LI/RI proportions "
                             "preserved). Match the value used in CNN/FlowSDF experiments.")
    # Model
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-type", type=str, default="vit_b")
    # Training
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--task-name", type=str, default="MedSAM-PENGWIN")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bbox-shift", type=int, default=20)
    parser.add_argument("--use-amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log train step loss to W&B every N steps.")
    parser.add_argument("--bucket-cap-mb", type=int, default=25,
                        help="DDP gradient bucket size in MB.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    # ---- DDP setup --------------------------------------------------------
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        if not args.checkpoint.exists():
            print(
                f"ERROR: MedSAM checkpoint not found at {args.checkpoint}\n"
                "Download medsam_vit_b.pth from "
                "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN"
            )
            return 1

        if args.use_wandb:
            import wandb
            wandb.login()
            wandb.init(project=args.task_name, config={**vars(args), "world_size": world_size})

        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        save_dir = args.work_dir / f"{args.task_name}-{run_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(__file__, save_dir / f"{run_id}_{Path(__file__).name}")
        print(f"Saving to {save_dir}")
    else:
        run_id = None
        save_dir = None

    # Broadcast save_dir from rank 0 so all ranks know where to load checkpoints from.
    if ddp:
        save_dir_str = [str(save_dir) if is_main else ""]
        dist.broadcast_object_list(save_dir_str, src=0)
        save_dir = Path(save_dir_str[0])

    # ---- Data -------------------------------------------------------------
    if is_main:
        print("Loading box metadata...")
    box_meta = load_box_metadata_map(args.box_metadata)
    if is_main:
        print(f"  {len(box_meta)} samples in box metadata")
        print("Building dataloaders...")

    train_dataset = build_medsam_dataset(
        args.gated_csv_dir, box_meta, "train",
        large_subsample=args.large_subsample,
        bbox_shift=args.bbox_shift,
        is_main=is_main,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Val runs on rank 0 only — no need to distribute evaluation.
    if is_main:
        val_dataset = build_medsam_dataset(
            args.gated_csv_dir, box_meta, "val",
            large_subsample=None,
            bbox_shift=0,
            is_main=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_loader = None

    # ---- Model ------------------------------------------------------------
    if is_main:
        print("Loading MedSAM model...")
    sam_model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            bucket_cap_mb=args.bucket_cap_mb,
        )

    # Access underlying module for optimizer (DDP wraps it under .module).
    raw_model = model.module if ddp else model
    trainable = list(raw_model.image_encoder.parameters()) + list(raw_model.mask_decoder.parameters())

    if is_main:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in trainable)
        print(f"Total params: {total_p:,}  |  Trainable: {train_p:,}  |  World size: {world_size}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    # ---- Resume -----------------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(args.resume, map_location=map_loc)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main:
            print(f"Resumed from epoch {ckpt['epoch']}: {args.resume}")
    if ddp:
        dist.barrier()

    # ---- Training loop ----------------------------------------------------
    train_losses, val_losses = [], []
    best_val_loss = 1e10
    global_step = 0

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, global_step = run_epoch(
            model, train_loader, seg_loss, ce_loss, optimizer,
            device, args.use_amp, scaler, train=True,
            use_wandb=args.use_wandb and is_main,
            global_step=global_step,
            log_every=args.log_every,
            ddp=ddp,
        )

        if is_main:
            val_loss, _ = run_epoch(
                model, val_loader, seg_loss, ce_loss, optimizer,
                device, args.use_amp, scaler, train=False,
                ddp=False,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            ts = datetime.now().strftime("%Y%m%d-%H%M")
            print(f"[{ts}] Epoch {epoch:04d}  train={train_loss:.4f}  val={val_loss:.4f}")

            if args.use_wandb:
                import wandb
                wandb.log({"epoch/train_loss": train_loss, "epoch/val_loss": val_loss}, step=global_step)

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

        if ddp:
            dist.barrier()

    if is_main:
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")

    if ddp:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
