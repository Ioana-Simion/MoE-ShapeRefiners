#!/usr/bin/env python3
"""PLAN_B (FlowSDF variant), cluster script.

Zero-shot transfer of frozen PENGWIN-trained FlowSDF experts to RAM-H1200,
batched and (optionally) multi-GPU. Mirrors inference_new_dataset_flowsdf.ipynb
but as a script meant for a SLURM job rather than an interactive notebook.

Pipeline:
    1. Adapter: RAM-H1200 COCO -> boxes/gt_masks/metadata (prepare_ramh1200_boxes).
    2. Stage-1 MedSAM inference (unchanged from the PENGWIN pipeline), sharded
       across GPUs by image when --num-gpus > 1.
    3. Gating: RAM-H1200-relative area threshold (median predicted-mask area).
    4. Frozen FlowSDF expert inference, ODE_STEPS Euler steps (default 40 --
       matches src/FlowSDF/jobs-slurmOutputs/2_infer_flowsdf.job, the job that
       produced this repo's reported PENGWIN numbers; the infer_flowsdf_moe.py
       script's own default of 4 is NOT representative -- see this repo's
       step-size ablation in jobs-slurmOutputs/slurm_evaluate_flowsdf_stepsize*).
       Fragments are batched (same expert, same ODE step) so a GPU forward
       pass processes --batch-size fragments at once instead of one at a time.
    5. Evaluation: MedSAM baseline vs. MedSAM + frozen FlowSDF, dice/iou/hd95/assd,
       overall + by size group.
    6. Visualization: worst / median / best cases by dice delta, each panel
       (GT / MedSAM baseline / FlowSDF refined) saved as its own PNG -- no
       merged figure -- named so the scan and the panel are both identifiable.

Single GPU (recommended starting point -- see note below):
    python src/run_ramh1200_flowsdf.py --split test --num-gpus 1

Smoke test first:
    python src/run_ramh1200_flowsdf.py --split test --num-gpus 1 --limit 15

Two GPUs, one SLURM task that owns both directly (this script spawns its own
worker subprocesses internally via multiprocessing + join(), all within one
process -- no cross-process synchronization needed):
    python src/run_ramh1200_flowsdf.py --split test --num-gpus 2

Deliberately NOT supported here: launching via `srun --ntasks=N` with one
independent process per GPU (RANK/WORLD_SIZE in the environment). That would
need cross-process barriers for the two points that need a global view --
computing the gating threshold after Stage-1 MedSAM, and evaluation/
visualization after FlowSDF -- which adds real coordination risk for a
workload where batching (--batch-size) already gives most of the win.
Single-GPU-with-batching first; revisit multi-GPU only if that's still too
slow once measured for real.
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
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
MEDSAM_REPO_ROOT = WORKSPACE_ROOT / "MedSAM"

for _p in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "gating_mechanism",
    REPO_ROOT / "src" / "FlowSDF",
    MEDSAM_REPO_ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prepare_ramh1200_boxes import process_split  # noqa: E402
try:
    from run_medsam_with_pengwin_boxes import (  # noqa: E402
        load_and_preprocess_image,
        run_medsam_with_boxes,
    )
    from segment_anything import sam_model_registry  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name == "segment_anything":
        raise ModuleNotFoundError(
            "segment_anything not importable. This is almost always a missing "
            "pip package, not a src/ layout problem -- run_medsam_with_pengwin_boxes.py "
            f"also expects a sibling 'MedSAM' checkout at {MEDSAM_REPO_ROOT} for its "
            "own default checkpoint path, but the segment_anything *package* itself "
            "should come from pip. Fix: `pip install segment-anything` in the active "
            "env (it's already in this repo's environment.yml). If the checkpoint "
            "file isn't at the assumed sibling-MedSAM location, pass "
            "--medsam-checkpoint /actual/path/to/medsam_vit_b.pth explicitly."
        ) from exc
    raise
from dataloader_utils import load_prediction_masks, resize_binary_nearest  # noqa: E402
from models import unet_segdiff  # noqa: E402
from evaluation.evaluate_medsam_pengwin import (  # noqa: E402
    dice_score,
    iou_score,
    boundary_metrics,
)

FLOWSDF_IMG_SIZE = 128
SIGMA_MIN = 1e-5
SDF_BINARY_THRESHOLD = 0.03
OUTPUT_SIZE = 1024

# PENGWIN's own small/large split point -- the threshold the frozen experts
# were actually trained with, which is their own 25th percentile fragment
# area (data_distribution_results/data_analysis_gt_with_medsam/threshold_suggestions.csv).
# RAM-H1200's own median fragment area is well below this. See --pengwin-reroute.
PENGWIN_SMALL_THRESHOLD = 5402.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "RAM-H1200-v1_dataset" / "Segmentation")
    parser.add_argument("--boxes-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "bounding-boxes")
    parser.add_argument("--medsam-pred-root", type=Path, default=REPO_ROOT / "data" / "ramh1200" / "medsam-predictions")
    parser.add_argument(
        "--flowsdf-pred-root", type=Path, default=None,
        help=(
            "Where FlowSDF masks/eval CSVs are written. Defaults to "
            "data/ramh1200/flowsdf-predictions/ode<N> where <N> is --ode-steps -- "
            "runs at different step counts land in separate directories automatically, "
            "so a second run doesn't silently reuse or overwrite a different step "
            "count's masks via --skip-existing. Pass this explicitly to override."
        ),
    )
    parser.add_argument(
        "--figures-root", type=Path, default=None,
        help="Where visualization panels are written. Defaults to figures/ramh1200_flowsdf/ode<N>, same reasoning as --flowsdf-pred-root.",
    )
    parser.add_argument("--medsam-checkpoint", type=Path, default=MEDSAM_REPO_ROOT / "medsam_vit_b.pth")
    parser.add_argument("--flowsdf-checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints" / "FlowSDF")
    parser.add_argument("--ode-steps", type=int, default=40, help="Euler ODE steps. 40 matches the job that produced this repo's reported PENGWIN numbers; the infer_flowsdf_moe.py default of 4 is not representative.")
    parser.add_argument("--n-eval", type=int, default=1, help="Number of stochastic ODE trajectories to average per fragment.")
    parser.add_argument("--batch-size", type=int, default=16, help="Fragments per batched FlowSDF forward pass (same expert).")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to shard images across via this script's own internal multiprocessing (one process, join()-based -- no cross-process sync needed). 1 = plain single-process run.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images (smoke test).")
    parser.add_argument("--n-viz-each", type=int, default=3, help="How many worst/median/best cases to save panels for.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", help="Skip MedSAM/FlowSDF outputs that already exist on disk.")
    parser.add_argument(
        "--pengwin-reroute", action="store_true",
        help=(
            "Additionally re-route fragments using PENGWIN's own absolute area threshold "
            f"({PENGWIN_SMALL_THRESHOLD:.0f}px) instead of this dataset's relative median, "
            "and re-run FlowSDF only on the fragments whose expert assignment changes "
            "(large -> small, since PENGWIN's threshold sits above RAM-H1200's own median). "
            "Reuses the primary run's masks for everything unchanged. See notebook Section 11."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_devices(num_gpus: int) -> list[str]:
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        n = max(1, min(num_gpus, available))
        return [f"cuda:{i}" for i in range(n)]
    if torch.backends.mps.is_available():
        if num_gpus > 1:
            print("WARNING: MPS has only one device; --num-gpus > 1 ignored.")
        return ["mps"]
    if num_gpus > 1:
        print("WARNING: no GPU available; --num-gpus > 1 ignored.")
    return ["cpu"]


# ---------------------------------------------------------------------------
# FlowSDF model
# ---------------------------------------------------------------------------
def build_flowsdf_model(img_cond_channels: int, device: str) -> torch.nn.Module:
    return unet_segdiff.UNetModel(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=3,
        attention_resolutions=(16, 8),
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        rrdb_blocks=12,
        img_cond_channels=img_cond_channels,
    ).to(device)


def load_flowsdf_expert(checkpoint_path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    img_cond_channels = int(ckpt.get("img_cond_channels", 257))
    model = build_flowsdf_model(img_cond_channels, device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    return model


def prepare_flowsdf_conditioning(embedding: torch.Tensor, binary_mask: np.ndarray, img_size: int, device: str) -> torch.Tensor:
    """Return (1, 257, img_size, img_size) conditioning for one fragment."""
    embedding = embedding.to(device=device, dtype=torch.float32)
    embedding_resized = F.interpolate(embedding.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False)
    mask = torch.from_numpy(binary_mask.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask, size=(img_size, img_size), mode="nearest")
    mask_resized = (mask_resized > 0.5).float()
    return torch.cat([embedding_resized, mask_resized], dim=1)


@torch.no_grad()
def sample_flowsdf_sdf_batched(model: torch.nn.Module, img_cond: torch.Tensor, sigma_min: float, ode_steps: int, n_eval: int) -> torch.Tensor:
    """Batched Euler ODE sample. img_cond: (B, 257, H, W). Returns (B, 1, H, W).

    Unchanged sampling math from infer_flowsdf_moe.py -- it was already
    batch-shape-agnostic (uses m.shape[0] throughout), so batching many
    fragments through one call is a pure efficiency change, not a behavior
    change relative to calling it B times with batch size 1.
    """
    device = img_cond.device
    batch, _, height, width = img_cond.shape
    samples = []

    for _ in range(n_eval):
        m0 = torch.randn((batch, 1, height, width), device=device)

        def func_conditional(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            t_curr = torch.ones(m.shape[0], device=m.device) * t
            m_ipt = (1 - (1 - sigma_min) * t) * m0 + t * m
            return model(m_ipt, t_curr.reshape(m.shape[0], -1), img_cond=img_cond)

        times = torch.linspace(0, 1, ode_steps, device=device)
        m = m0
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            m = m + dt * func_conditional(times[i], m)
        samples.append(m)

    if len(samples) == 1:
        return samples[0]
    return torch.stack(samples, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Stage 1: MedSAM (one worker's shard)
# ---------------------------------------------------------------------------
def run_medsam_shard(records: list[dict], device: str, checkpoint: Path, medsam_pred_root: Path, split: str, skip_existing: bool) -> list[dict]:
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    model.eval()

    binary_masks_dir = medsam_pred_root / split / "binary_masks"
    embeddings_dir = medsam_pred_root / split / "embeddings"
    binary_masks_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    out_records = []
    for record in tqdm(records, desc=f"[{device}] MedSAM Stage-1"):
        sample_name = record["sample_name"]
        masks_path = binary_masks_dir / f"{sample_name}.npz"
        embedding_path = embeddings_dir / f"{sample_name}.npy"

        if skip_existing and masks_path.exists() and embedding_path.exists():
            out_records.append({**record, "binary_masks_path": str(masks_path.resolve()), "embedding_path": str(embedding_path.resolve())})
            continue

        image = load_and_preprocess_image(record["original_image_path"], image_size=record["image_size"])
        boxes = np.load(record["box_path"]).astype(np.float32)
        binary_masks, embedding = run_medsam_with_boxes(model, image, boxes, device, threshold=0.5)

        np.savez_compressed(masks_path, masks=binary_masks)
        np.save(embedding_path, embedding)
        out_records.append({**record, "binary_masks_path": str(masks_path.resolve()), "embedding_path": str(embedding_path.resolve())})

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out_records


def _medsam_worker(rank: int, device: str, records: list[dict], checkpoint: Path, medsam_pred_root: Path, split: str, skip_existing: bool, out_path: Path) -> None:
    out_records = run_medsam_shard(records, device, checkpoint, medsam_pred_root, split, skip_existing)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Stage 2: gating (single process, needs the full shard-merged dataset)
# ---------------------------------------------------------------------------
def compute_gating(medsam_pred_records: list[dict]) -> tuple[pd.DataFrame, float]:
    areas = []
    for record in medsam_pred_records:
        masks = load_prediction_masks(record["binary_masks_path"])
        for i in range(masks.shape[0]):
            areas.append(int(masks[i].sum()))
    threshold = float(np.median(areas))

    rows = []
    for record in medsam_pred_records:
        masks = load_prediction_masks(record["binary_masks_path"])
        for frag, mask in zip(record["fragments"], masks):
            area = int(mask.sum())
            rows.append({
                "sample_name": record["sample_name"],
                "medsam_instance_id": frag["medsam_instance_id"],
                "category_name": frag["category_name"],
                "area": area,
                "expert": "expert_small" if area <= threshold else "expert_large",
                "embedding_path": record["embedding_path"],
                "binary_masks_path": record["binary_masks_path"],
            })
    return pd.DataFrame(rows), threshold


# ---------------------------------------------------------------------------
# Stage 3: batched FlowSDF inference (one worker's shard)
# ---------------------------------------------------------------------------
def run_flowsdf_shard(
    sample_names: list[str],
    gated_df: pd.DataFrame,
    device: str,
    checkpoint_dir: Path,
    ode_steps: int,
    n_eval: int,
    batch_size: int,
    flowsdf_masks_dir: Path,
    skip_existing: bool,
) -> None:
    experts = {}
    for expert_id in ("expert_small", "expert_large"):
        ckpt_path = checkpoint_dir / f"{expert_id}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        experts[expert_id] = load_flowsdf_expert(ckpt_path, device)

    shard_df = gated_df[gated_df["sample_name"].isin(sample_names)]

    # Build a flat task list: (sample_name, medsam_instance_id, expert, embedding_path, binary_masks_path)
    pending_sample_names = sample_names
    if skip_existing:
        pending_sample_names = [s for s in sample_names if not (flowsdf_masks_dir / f"{s}.npz").exists()]

    tasks = shard_df[shard_df["sample_name"].isin(pending_sample_names)].to_dict("records")
    if not tasks:
        return

    # Group by expert so a batch only ever hits one model.
    tasks_by_expert: dict[str, list[dict]] = {"expert_small": [], "expert_large": []}
    for t in tasks:
        tasks_by_expert[t["expert"]].append(t)

    # image_size cache to avoid reloading the same masks/embedding repeatedly
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

    # Accumulate refined masks per image, write once all its fragments are done.
    refined_by_sample: dict[str, np.ndarray] = {}
    remaining_by_sample: dict[str, int] = shard_df[shard_df["sample_name"].isin(pending_sample_names)].groupby("sample_name").size().to_dict()

    for expert_id, expert_tasks in tasks_by_expert.items():
        model = experts[expert_id]
        for batch_start in tqdm(range(0, len(expert_tasks), batch_size), desc=f"[{device}] FlowSDF {expert_id}"):
            batch_tasks = expert_tasks[batch_start:batch_start + batch_size]

            cond_list = []
            for t in batch_tasks:
                embedding = get_embedding(t["embedding_path"])
                masks = get_medsam_masks(t["binary_masks_path"])
                idx = int(t["medsam_instance_id"]) - 1
                cond_list.append(prepare_flowsdf_conditioning(embedding, masks[idx], FLOWSDF_IMG_SIZE, device))

            img_cond = torch.cat(cond_list, dim=0)
            sdf = sample_flowsdf_sdf_batched(model, img_cond, SIGMA_MIN, ode_steps, n_eval)
            mask_small = (sdf <= SDF_BINARY_THRESHOLD).float()
            mask_up = F.interpolate(mask_small, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="nearest")
            mask_up = mask_up[:, 0].cpu().numpy().astype(np.uint8)

            for t, refined in zip(batch_tasks, mask_up):
                sample_name = t["sample_name"]
                if sample_name not in refined_by_sample:
                    n_frag = get_medsam_masks(t["binary_masks_path"]).shape[0]
                    refined_by_sample[sample_name] = np.zeros((n_frag, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
                idx = int(t["medsam_instance_id"]) - 1
                refined_by_sample[sample_name][idx] = refined
                remaining_by_sample[sample_name] -= 1
                if remaining_by_sample[sample_name] == 0:
                    np.savez_compressed(flowsdf_masks_dir / f"{sample_name}.npz", masks=refined_by_sample.pop(sample_name))

    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _flowsdf_worker(rank: int, device: str, sample_names: list[str], gated_df: pd.DataFrame, checkpoint_dir: Path, ode_steps: int, n_eval: int, batch_size: int, flowsdf_masks_dir: Path, skip_existing: bool) -> None:
    run_flowsdf_shard(sample_names, gated_df, device, checkpoint_dir, ode_steps, n_eval, batch_size, flowsdf_masks_dir, skip_existing)


# ---------------------------------------------------------------------------
# Stage 3b: PENGWIN-absolute reroute (only recompute flipped fragments)
# ---------------------------------------------------------------------------
def reroute_pengwin_convention(gated_df: pd.DataFrame) -> pd.DataFrame:
    """Reassign expert using PENGWIN's absolute area threshold instead of this
    dataset's own relative median. See notebook Section 11: PENGWIN's 5402px
    threshold is their own 25th percentile fragment area, and RAM-H1200
    fragments sit much lower on that absolute scale."""
    rerouted = gated_df.copy()
    rerouted["expert"] = np.where(rerouted["area"] <= PENGWIN_SMALL_THRESHOLD, "expert_small", "expert_large")
    return rerouted


def run_pengwin_reroute_shard(
    sample_names: list[str],
    gated_df: pd.DataFrame,
    gated_df_pengwin: pd.DataFrame,
    device: str,
    checkpoint_dir: Path,
    ode_steps: int,
    n_eval: int,
    batch_size: int,
    source_masks_dir: Path,
    dest_masks_dir: Path,
    skip_existing: bool,
) -> None:
    """Seed every sample's output from the primary run's already-computed
    masks, then only recompute the fragments whose expert assignment changes
    under PENGWIN-absolute routing. Direction-agnostic (doesn't assume flips
    only go large->small), though in practice that's the only direction seen
    here since PENGWIN_SMALL_THRESHOLD sits above RAM-H1200's own median."""
    pending_sample_names = sample_names
    if skip_existing:
        pending_sample_names = [s for s in sample_names if not (dest_masks_dir / f"{s}.npz").exists()]
    if not pending_sample_names:
        return

    orig_by_key = {
        (r["sample_name"], int(r["medsam_instance_id"])): r["expert"]
        for r in gated_df[gated_df["sample_name"].isin(pending_sample_names)].to_dict("records")
    }
    new_rows = gated_df_pengwin[gated_df_pengwin["sample_name"].isin(pending_sample_names)].to_dict("records")
    flipped_tasks = [
        r for r in new_rows
        if orig_by_key.get((r["sample_name"], int(r["medsam_instance_id"]))) != r["expert"]
    ]
    print(f"[{device}] {len(flipped_tasks)} / {len(new_rows)} fragments flip expert under PENGWIN-absolute routing")

    out_arrays: dict[str, np.ndarray] = {
        s: load_prediction_masks(source_masks_dir / f"{s}.npz").copy() for s in pending_sample_names
    }

    if flipped_tasks:
        experts: dict[str, torch.nn.Module] = {}
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
                experts[expert_id] = load_flowsdf_expert(ckpt_path, device)
            model = experts[expert_id]

            for batch_start in tqdm(range(0, len(expert_tasks), batch_size), desc=f"[{device}] PENGWIN-reroute {expert_id}"):
                batch_tasks = expert_tasks[batch_start:batch_start + batch_size]

                cond_list = []
                for t in batch_tasks:
                    embedding = get_embedding(t["embedding_path"])
                    masks = get_medsam_masks(t["binary_masks_path"])
                    idx = int(t["medsam_instance_id"]) - 1
                    cond_list.append(prepare_flowsdf_conditioning(embedding, masks[idx], FLOWSDF_IMG_SIZE, device))

                img_cond = torch.cat(cond_list, dim=0)
                sdf = sample_flowsdf_sdf_batched(model, img_cond, SIGMA_MIN, ode_steps, n_eval)
                mask_small = (sdf <= SDF_BINARY_THRESHOLD).float()
                mask_up = F.interpolate(mask_small, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="nearest")
                mask_up = mask_up[:, 0].cpu().numpy().astype(np.uint8)

                for t, refined in zip(batch_tasks, mask_up):
                    idx = int(t["medsam_instance_id"]) - 1
                    out_arrays[t["sample_name"]][idx] = refined

    for sample_name, arr in out_arrays.items():
        np.savez_compressed(dest_masks_dir / f"{sample_name}.npz", masks=arr)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_predictions(medsam_pred_records: list[dict], pred_masks_dir: Path, gated_df: pd.DataFrame, desc: str = "evaluating") -> pd.DataFrame:
    rows = []
    for record in tqdm(medsam_pred_records, desc=desc):
        sample_name = record["sample_name"]
        gt_masks = np.load(record["gt_masks_path"])["masks"]
        pred_masks = load_prediction_masks(pred_masks_dir / f"{sample_name}.npz")

        frag_expert = {
            int(r["medsam_instance_id"]): r["expert"]
            for _, r in gated_df[gated_df["sample_name"] == sample_name].iterrows()
        }

        for i, frag in enumerate(record["fragments"]):
            gt = gt_masks[i]
            pred = resize_binary_nearest(pred_masks[i], gt.shape)
            dice = dice_score(pred, gt)
            iou = iou_score(pred, gt)
            hd95, assd = boundary_metrics(pred, gt)
            rows.append({
                "sample_name": sample_name,
                "category_name": frag["category_name"],
                "size_group": frag_expert.get(frag["medsam_instance_id"], "unknown"),
                "dice": dice, "iou": iou, "hd95": hd95, "assd": assd,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization: separate panels, no merged figure
# ---------------------------------------------------------------------------
def colorize(masks: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    overlay = np.zeros((*masks.shape[1:], 4))
    rng = np.random.default_rng(rng_seed)
    for i in range(masks.shape[0]):
        color = rng.uniform(0.3, 1.0, size=3)
        m = masks[i].astype(bool)
        overlay[m] = [*color, 0.45]
    return overlay


def save_panel(image: np.ndarray, masks: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = image.shape
    fig, ax = plt.subplots(figsize=(w / 150, h / 150), dpi=150)
    ax.imshow(image, cmap="gray", extent=(0, w, h, 0))
    ax.imshow(colorize(masks), extent=(0, w, h, 0))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_visualizations(
    medsam_pred_records: list[dict],
    medsam_pred_root: Path,
    flowsdf_masks_dir: Path,
    split: str,
    figures_root: Path,
    n_each: int,
) -> None:
    records_by_name = {r["sample_name"]: r for r in medsam_pred_records}

    delta_rows = []
    for record in medsam_pred_records:
        sample_name = record["sample_name"]
        gt_masks = np.load(record["gt_masks_path"])["masks"]
        baseline_masks = load_prediction_masks(medsam_pred_root / split / "binary_masks" / f"{sample_name}.npz")
        flowsdf_masks = load_prediction_masks(flowsdf_masks_dir / f"{sample_name}.npz")

        base_dices, flow_dices = [], []
        for i in range(gt_masks.shape[0]):
            gt = gt_masks[i]
            base_dices.append(dice_score(resize_binary_nearest(baseline_masks[i], gt.shape), gt))
            flow_dices.append(dice_score(resize_binary_nearest(flowsdf_masks[i], gt.shape), gt))

        delta_rows.append({
            "sample_name": sample_name,
            "base_dice": float(np.mean(base_dices)),
            "flowsdf_dice": float(np.mean(flow_dices)),
            "delta": float(np.mean(flow_dices) - np.mean(base_dices)),
        })

    delta_df = pd.DataFrame(delta_rows).sort_values("delta").reset_index(drop=True)
    n = len(delta_df)
    n_each = min(n_each, n // 2 if n >= 2 else 1)

    median_start = max(0, n // 2 - n_each // 2)
    # (case_label, rank_within_case (1-based, for the filename), row index into delta_df)
    selections = (
        [("worst", rank, i) for rank, i in enumerate(range(n_each), start=1)]
        + [("median", rank, i) for rank, i in enumerate(range(median_start, median_start + n_each), start=1)]
        + [("best", rank, i) for rank, i in enumerate(range(n - n_each, n), start=1)]
    )

    figures_dir = figures_root / split
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for case_label, rank, row_idx in selections:
        row = delta_df.iloc[row_idx]
        sample_name = row["sample_name"]
        record = records_by_name[sample_name]
        image = np.array(Image.open(record["original_image_path"]).convert("L"))

        gt_masks = np.load(record["gt_masks_path"])["masks"]
        baseline_masks = load_prediction_masks(medsam_pred_root / split / "binary_masks" / f"{sample_name}.npz")
        flowsdf_masks = load_prediction_masks(flowsdf_masks_dir / f"{sample_name}.npz")

        for method_label, masks in [("gt", gt_masks), ("medsam_baseline", baseline_masks), ("flowsdf_refined", flowsdf_masks)]:
            # naming: {case}{rank}_{sample_id}_{method}.png -- sample_id (scan) and
            # case/method are both in the filename so either can be grepped for.
            out_name = f"{case_label}{rank}_{sample_name}_{method_label}.png"
            out_path = figures_dir / out_name
            save_panel(image, masks, out_path)
            manifest.append({
                "case": case_label,
                "rank_within_case": rank,
                "sample_name": sample_name,
                "method": method_label,
                "base_dice": row["base_dice"],
                "flowsdf_dice": row["flowsdf_dice"],
                "delta": row["delta"],
                "path": str(out_path),
            })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(figures_dir / "manifest.csv", index=False)
    print(f"saved {len(manifest)} panels to {figures_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.flowsdf_pred_root is None:
        args.flowsdf_pred_root = REPO_ROOT / "data" / "ramh1200" / "flowsdf-predictions" / f"ode{args.ode_steps}"
    if args.figures_root is None:
        args.figures_root = REPO_ROOT / "figures" / "ramh1200_flowsdf" / f"ode{args.ode_steps}"

    devices = pick_devices(args.num_gpus)
    print(f"devices: {devices}")
    print(f"flowsdf-pred-root: {args.flowsdf_pred_root}")
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

    # --- Step 2: MedSAM Stage-1, sharded across devices -----------------------
    print("\n=== Step 2: MedSAM Stage-1 ===")
    shards = [records[i::len(devices)] for i in range(len(devices))]

    if len(devices) == 1:
        medsam_pred_records = run_medsam_shard(
            shards[0], devices[0], args.medsam_checkpoint, args.medsam_pred_root, args.split, args.skip_existing
        )
    else:
        tmp_dir = args.medsam_pred_root / args.split
        tmp_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [tmp_dir / f"_shard_medsam_{i}.jsonl" for i in range(len(devices))]
        ctx = mp.get_context("spawn")
        procs = []
        for i, (device, shard) in enumerate(zip(devices, shards)):
            p = ctx.Process(
                target=_medsam_worker,
                args=(i, device, shard, args.medsam_checkpoint, args.medsam_pred_root, args.split, args.skip_existing, shard_paths[i]),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"MedSAM worker failed with exit code {p.exitcode}")

        medsam_pred_records = []
        for sp in shard_paths:
            medsam_pred_records.extend(json.loads(l) for l in sp.open())
            sp.unlink()

    pred_metadata_path = args.medsam_pred_root / args.split / "metadata.jsonl"
    with pred_metadata_path.open("w") as f:
        for r in medsam_pred_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(medsam_pred_records)} MedSAM prediction records")

    # --- Step 3: gating (needs the full merged dataset) -----------------------
    print("\n=== Step 3: gating threshold ===")
    gated_df, area_threshold = compute_gating(medsam_pred_records)
    gating_dir = REPO_ROOT / "src" / "gating_mechanism"
    gated_csv_path = gating_dir / f"gated_ramh1200_flowsdf_{args.split}_records.csv"
    gated_df.to_csv(gated_csv_path, index=False)
    print(f"area threshold: {area_threshold:.1f}px, saved {len(gated_df)} gated fragments -> {gated_csv_path}")
    print(gated_df["expert"].value_counts().to_string())

    # --- Step 4: batched FlowSDF inference, sharded across devices -----------
    print(f"\n=== Step 4: FlowSDF inference (ode_steps={args.ode_steps}, batch_size={args.batch_size}) ===")
    flowsdf_masks_dir = args.flowsdf_pred_root / args.split / "binary_masks"
    flowsdf_masks_dir.mkdir(parents=True, exist_ok=True)

    sample_names = [r["sample_name"] for r in medsam_pred_records]
    name_shards = [sample_names[i::len(devices)] for i in range(len(devices))]

    if len(devices) == 1:
        run_flowsdf_shard(
            name_shards[0], gated_df, devices[0], args.flowsdf_checkpoint_dir,
            args.ode_steps, args.n_eval, args.batch_size, flowsdf_masks_dir, args.skip_existing,
        )
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for i, (device, shard) in enumerate(zip(devices, name_shards)):
            p = ctx.Process(
                target=_flowsdf_worker,
                args=(i, device, shard, gated_df, args.flowsdf_checkpoint_dir, args.ode_steps, args.n_eval, args.batch_size, flowsdf_masks_dir, args.skip_existing),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"FlowSDF worker failed with exit code {p.exitcode}")

    # --- Step 4b: PENGWIN-absolute reroute (optional) --------------------------
    gated_df_pengwin = None
    pengwin_masks_dir = None
    if args.pengwin_reroute:
        print(f"\n=== Step 4b: PENGWIN-absolute reroute (threshold={PENGWIN_SMALL_THRESHOLD:.0f}px) ===")
        gated_df_pengwin = reroute_pengwin_convention(gated_df)
        pengwin_csv_path = gating_dir / f"gated_ramh1200_flowsdf_pengwin_{args.split}_records.csv"
        gated_df_pengwin.to_csv(pengwin_csv_path, index=False)
        changed = int((gated_df_pengwin["expert"] != gated_df["expert"]).sum())
        print(f"{changed} / {len(gated_df)} fragments change expert assignment ({changed / len(gated_df):.1%})")
        print(gated_df_pengwin["expert"].value_counts().to_string())

        pengwin_masks_dir = args.flowsdf_pred_root / args.split / "binary_masks_pengwin_convention"
        pengwin_masks_dir.mkdir(parents=True, exist_ok=True)
        run_pengwin_reroute_shard(
            sample_names, gated_df, gated_df_pengwin, devices[0], args.flowsdf_checkpoint_dir,
            args.ode_steps, args.n_eval, args.batch_size, flowsdf_masks_dir, pengwin_masks_dir, args.skip_existing,
        )

    # --- Step 5: evaluation ----------------------------------------------------
    print("\n=== Step 5: evaluation ===")
    baseline_df = evaluate_predictions(medsam_pred_records, args.medsam_pred_root / args.split / "binary_masks", gated_df, desc="eval: MedSAM baseline")
    flowsdf_df = evaluate_predictions(medsam_pred_records, flowsdf_masks_dir, gated_df, desc="eval: FlowSDF (RAM-H1200-relative)")

    eval_dir = args.flowsdf_pred_root / args.split
    baseline_df.to_csv(eval_dir / "evaluation_medsam_baseline.csv", index=False)
    flowsdf_df.to_csv(eval_dir / "evaluation_flowsdf.csv", index=False)

    print("\nMedSAM baseline (overall):")
    print(baseline_df[["dice", "iou", "hd95", "assd"]].mean())
    print("\nMedSAM + frozen FlowSDF (overall):")
    print(flowsdf_df[["dice", "iou", "hd95", "assd"]].mean())
    print("\nMedSAM baseline by size group:")
    print(baseline_df.groupby("size_group")[["dice", "iou", "hd95", "assd"]].mean())
    print("\nMedSAM + frozen FlowSDF by size group:")
    print(flowsdf_df.groupby("size_group")[["dice", "iou", "hd95", "assd"]].mean())

    if args.pengwin_reroute:
        pengwin_df = evaluate_predictions(medsam_pred_records, pengwin_masks_dir, gated_df_pengwin, desc="eval: FlowSDF (PENGWIN-absolute)")
        pengwin_df.to_csv(eval_dir / "evaluation_flowsdf_pengwin_convention.csv", index=False)

        print("\nMedSAM + frozen FlowSDF, PENGWIN-absolute routing (overall):")
        print(pengwin_df[["dice", "iou", "hd95", "assd"]].mean())
        print("\nMedSAM + frozen FlowSDF, PENGWIN-absolute routing, by (new) size group:")
        print(pengwin_df.groupby("size_group")[["dice", "iou", "hd95", "assd"]].mean())

        # Isolate the effect: for exactly the fragments that flip expert, compare
        # baseline vs. old (RAM-H1200-relative) refined vs. new (PENGWIN-absolute)
        # refined, fragment by fragment -- holds the fragment set fixed and only
        # changes which expert refined it.
        flipped_keys = set(
            map(tuple, gated_df_pengwin.loc[
                gated_df_pengwin["expert"] != gated_df["expert"], ["sample_name", "medsam_instance_id"]
            ].values)
        )
        print(f"\nIsolating the {len(flipped_keys)} flipped fragments:")
        flipped_rows = []
        for sample_name in tqdm({k[0] for k in flipped_keys}, desc="eval: flipped fragments"):
            record = next(r for r in medsam_pred_records if r["sample_name"] == sample_name)
            gt_masks = np.load(record["gt_masks_path"])["masks"]
            baseline_masks = load_prediction_masks(args.medsam_pred_root / args.split / "binary_masks" / f"{sample_name}.npz")
            old_masks = load_prediction_masks(flowsdf_masks_dir / f"{sample_name}.npz")
            new_masks = load_prediction_masks(pengwin_masks_dir / f"{sample_name}.npz")

            for frag in record["fragments"]:
                key = (sample_name, frag["medsam_instance_id"])
                if key not in flipped_keys:
                    continue
                idx = frag["medsam_instance_id"] - 1
                gt = gt_masks[idx]
                flipped_rows.append({
                    "sample_name": sample_name,
                    "baseline_dice": dice_score(resize_binary_nearest(baseline_masks[idx], gt.shape), gt),
                    "expert_large_dice": dice_score(resize_binary_nearest(old_masks[idx], gt.shape), gt),
                    "expert_small_dice": dice_score(resize_binary_nearest(new_masks[idx], gt.shape), gt),
                })
        flipped_df = pd.DataFrame(flipped_rows)
        flipped_df.to_csv(eval_dir / "evaluation_flipped_fragments.csv", index=False)
        print(flipped_df[["baseline_dice", "expert_large_dice", "expert_small_dice"]].mean())

    # --- Step 6: visualizations (separate panels, not merged) ------------------
    print("\n=== Step 6: visualizations ===")
    save_visualizations(
        medsam_pred_records, args.medsam_pred_root, flowsdf_masks_dir,
        args.split, args.figures_root, args.n_viz_each,
    )

    elapsed = time.time() - t0
    print(f"\ntotal elapsed: {elapsed / 3600:.2f} hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
