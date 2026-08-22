#!/usr/bin/env python3
"""Boundary-focused loss for MoE mask refinement.

Components:
    1. Soft Dice loss          — global foreground overlap.
    2. Boundary-weighted BCE   — BCE upweighted near GT fragment edges.

Both operate on probabilities (sigmoid output), not logits.
"""
import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss, returns per-sample losses of shape (B,).

    Args:
        pred: (B, 1, H, W) probabilities in [0, 1].
        gt:   (B, 1, H, W) binary float mask.
    """
    pred_flat = pred.view(pred.size(0), -1)
    gt_flat   = gt.view(gt.size(0), -1)
    intersection = (pred_flat * gt_flat).sum(dim=1)
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + eps)


def _boundary_weight_map(
    gt: torch.Tensor,
    w_boundary: float,
    radius: int,
) -> torch.Tensor:
    """Per-pixel weight map: w_boundary near GT edges, 1.0 elsewhere.

    Boundary = dilated(gt) − eroded(gt), approximated with max-pool ops
    so it runs entirely on GPU without scipy.

    Args:
        gt:         (B, 1, H, W) binary float mask.
        w_boundary: upweight factor for boundary pixels.
        radius:     dilation radius in pixels.
    """
    kernel = 2 * radius + 1
    pad    = radius

    dilated  = F.max_pool2d(gt,  kernel_size=kernel, stride=1, padding=pad)
    eroded   = -F.max_pool2d(-gt, kernel_size=kernel, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0.0, 1.0)   # 1 near edge, 0 elsewhere

    return 1.0 + (w_boundary - 1.0) * boundary      # baseline 1.0, boosted at boundary


def boundary_weighted_bce(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    w_boundary: float = 3.0,
    boundary_radius: int = 3,
) -> torch.Tensor:
    """BCE with per-pixel upweighting near GT boundaries.

    Args:
        pred:            (B, 1, H, W) probabilities in [0, 1].
        gt:              (B, 1, H, W) binary float mask.
        w_boundary:      upweight factor for boundary pixels (e.g. 3–5×).
        boundary_radius: dilation radius in pixels (e.g. 3).
    """
    W   = _boundary_weight_map(gt, w_boundary, boundary_radius)
    bce = F.binary_cross_entropy(pred, gt, reduction="none")
    return (W * bce).mean()


def boundary_dice_bce_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    w_boundary: float = 3.0,
    boundary_radius: int = 3,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Combined Dice + boundary-weighted BCE loss.

    Args:
        pred:            (B, 1, H, W) probabilities in [0, 1].
        gt:              (B, 1, H, W) binary float mask.
        w_boundary:      boundary pixel upweight factor.
        boundary_radius: dilation radius in pixels.
        dice_weight:     weight for Dice; BCE weight = 1 - dice_weight.
    """
    l_dice = dice_loss(pred, gt).mean()
    l_bce  = boundary_weighted_bce(pred, gt, w_boundary, boundary_radius)
    return dice_weight * l_dice + (1.0 - dice_weight) * l_bce


def load_balance_loss(gate_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shazeer-style importance/load-balancing loss for a soft MoE gate.

    Penalises the gate for concentrating mass on one expert across the batch,
    which keeps both experts alive under fresh initialisation.

    Args:
        gate_weights: (B, n_experts) softmax gate weights (post-noise at train
            time). Values in [0, 1], rows sum to 1.
        eps: numerical stabiliser for the mean in the denominator.

    Returns:
        Scalar CV(importance)^2, where importance = gate_weights.sum(dim=0).
    """
    importance = gate_weights.sum(dim=0)  # (n_experts,)
    return (importance.std() / (importance.mean() + eps)) ** 2
