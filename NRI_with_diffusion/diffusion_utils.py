# diffusion_utils.py
# Utilities for 1D DDPM-style diffusion on Euclidean latent vectors.
#
# Notes:
# - Timesteps are treated as 1-indexed (t in {1,...,T}) at the interface level,
#   matching many diffusion derivations and keeping consistency with the NRI codebase.
# - All schedules / buffers are stored as length-T tensors indexed by t-1 internally.

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end (length T)."""
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    return torch.clamp(betas, 1e-8, 0.999)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    Returns betas of length T.
    """
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device) / timesteps
    f = torch.cos((t + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


def get_beta_schedule(
    name: str,
    timesteps: int,
    device: Optional[torch.device] = None,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """
    Returns betas (length T) for the requested schedule.
    Supported: "linear", "cosine".
    """
    name = (name or "cosine").lower()
    if name == "linear":
        return linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end, device=device)
    if name == "cosine":
        return cosine_beta_schedule(timesteps, s=cosine_s, device=device)
    raise ValueError(f"Unknown beta schedule: {name}")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a (shape [T]) at timesteps t (shape [B]) and reshape for broadcast to x_shape.
    We assume t is 1-indexed in [1, T].

    Example:
        a: [T]
        t: [B] with values in {1,...,T}
        x_shape: (B, D) or (B, ...)
        returns: [B, 1, 1, ...] broadcastable to x_shape
    """
    if a.dim() != 1:
        raise ValueError(f"extract expects a to be 1D [T], got shape {tuple(a.shape)}")
    if t.dim() != 1:
        t = t.view(-1)
    # convert 1..T to 0..T-1
    t0 = torch.clamp(t.long() - 1, 0, a.shape[0] - 1)
    out = a.gather(0, t0)
    # broadcast to x_shape: [B, 1, 1, ...]
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for integer timesteps.
    """
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        if dim <= 0:
            raise ValueError("time embedding dim must be positive")
        self.dim = int(dim)
        self.max_period = int(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: int64 or float tensor of shape [B], 1-indexed preferred.
        returns: [B, dim]
        """
        if t.dim() != 1:
            t = t.view(-1)
        t = t.float()
        half = self.dim // 2
        if half == 0:
            # dim==1 case
            return torch.zeros((t.shape[0], 1), device=t.device, dtype=torch.float32)
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / float(half)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=-1)
        return emb
