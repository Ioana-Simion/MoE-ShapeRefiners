#!/usr/bin/env python3
"""CNN Expert model for MoE mask refinement (Alt 3: no RoI crops).

Input:  (B, c_in, H, W)  — MedSAM embedding + resized binary mask + SDF
Output: (B, 1,    H, W)  — refined mask probabilities in [0, 1]

Two instances of CNNExpert are trained independently, one per expert (small/large).
"""
import torch
import torch.nn as nn


class CNNExpert(nn.Module):
    """Lightweight spatial decoder for per-fragment mask refinement.

    Three conv layers act as a spatial decoder, not a feature extractor —
    MedSAM already extracted features. No backbone needed.

    Args:
        c_in: number of input channels. Default 258 = 256 (embedding)
              + 1 (binary mask) + 1 (SDF).
    """

    def __init__(self, c_in: int = 258):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gate(nn.Module):
    """Tiny gate over 2 experts: global-average-pool x, then a 2-layer MLP.

    Input:  x (B, c_in, H, W) — same tensor the experts consume.
    Output: logits (B, 2), pre-softmax and pre-noise.
    """

    def __init__(self, c_in: int = 258, hidden: int = 64, n_experts: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)  # (B, c_in)
        return self.mlp(pooled)           # (B, n_experts)


class GatedMoE(nn.Module):
    """Learned soft-gate mixture of two CNNExperts, jointly trained.

    Both experts run on every sample; the gate produces per-sample mixture
    weights that blend their outputs. Fixed-eps Gaussian noise is added to the
    gate logits during training only (Shazeer-style exploration), so a lagging
    expert can still receive gradient early in training.
    """

    def __init__(self, c_in: int = 258, gate_hidden: int = 64, gate_noise_eps: float = 0.3):
        super().__init__()
        self.expert_0 = CNNExpert(c_in=c_in)
        self.expert_1 = CNNExpert(c_in=c_in)
        self.gate = Gate(c_in=c_in, hidden=gate_hidden, n_experts=2)
        self.gate_noise_eps = gate_noise_eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)  # (B, 2)
        if self.training and self.gate_noise_eps > 0.0:
            logits = logits + self.gate_noise_eps * torch.randn_like(logits)
        g = torch.softmax(logits, dim=1)  # (B, 2)

        y0 = self.expert_0(x)  # (B, 1, H, W)
        y1 = self.expert_1(x)  # (B, 1, H, W)

        g0 = g[:, 0].view(-1, 1, 1, 1)
        g1 = g[:, 1].view(-1, 1, 1, 1)
        y_hat = g0 * y0 + g1 * y1

        return y_hat, g
