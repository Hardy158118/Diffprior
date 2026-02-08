# diffusion_prior.py
# One-step residual refinement + random-t diffusion loss for NRI-style logits.
#
# Key design (Scheme B):
#   - Decoder input uses deterministic ONE-STEP refinement at a fixed t_ref (stable train/val/test).
#   - Diffusion loss still uses random t and random noise to train the denoiser.
#
# Note: We DO NOT include the categorical entropy/logq surrogate term, because we treat the encoder
# logits as deterministic continuous latents for the diffusion prior fitting objective.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_utils import SinusoidalTimeEmbedding, get_beta_schedule, extract


@dataclass
class DiffusionLossStats:
    ddpm_mse: torch.Tensor        # unweighted MSE(eps_hat, eps), for logging
    ddpm_weighted: torch.Tensor   # weighted objective actually optimized
    t_mean: torch.Tensor          # mean sampled timestep
    w_mean: torch.Tensor          # mean weight w_t


class _EpsMLP(nn.Module):
    """Simple MLP eps_theta(z_t, t) for edge-wise latents."""
    def __init__(self, latent_dim: int, time_emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        in_dim = int(latent_dim) + int(time_emb_dim)
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(latent_dim)),
        )

    def forward(self, zt: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([zt, t_emb], dim=-1))


class DiffusionPrior(nn.Module):
    """
    Diffusion model over continuous edge logits z0 (shape [B,E,K]).

    - Training: weighted DDPM noise-prediction loss with random t (possibly restricted to small t-range).
    - Refinement: deterministic ONE-STEP residual refinement at fixed t_ref.

    Timesteps convention: t is 1-indexed in [1..T].
    We sample t from {2,..,t_max} for stability (avoid t=1 if posterior variance is used).
    """
    def __init__(
        self,
        latent_dim: int,
        timesteps: int = 100,
        schedule: str = "linear",
        time_emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        # IMPORTANT: entropy/logq term removed; keep arg for backward compat but unused.
        lambda_ent: float = 0.0,
        # If True, multiply by (#timesteps in sampling range) to approximate sum over t.
        scale_by_timesteps: bool = False,
        # Which reverse variance sigma_t^2 to use in the KL->MSE weight w_t.
        variance_type: Literal["beta", "posterior"] = "beta",
        # Training-time t sampling range and multi-t samples (A/B suggestions).
        train_t_max: Optional[int] = None,
        train_num_t: int = 1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.timesteps = int(timesteps)
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        self.schedule = str(schedule)
        self.time_emb_dim = int(time_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lambda_ent = float(lambda_ent)  # unused, kept only for CLI/backward compat
        self.scale_by_timesteps = bool(scale_by_timesteps)
        self.variance_type = str(variance_type)
        self.train_t_max = int(train_t_max) if train_t_max is not None else None
        self.train_num_t = int(train_num_t)
        if self.train_num_t < 1:
            raise ValueError("train_num_t must be >= 1")
        self.eps = float(eps)

        # --- forward process schedule ---
        betas = get_beta_schedule(self.schedule, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)  # [T]
        self.register_buffer("alphas", alphas)  # [T]
        self.register_buffer("alphas_cumprod", alphas_cumprod)  # [T]
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)  # [T]
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # --- posterior variance \tilde{beta}_t ---
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)

        # --- choose sigma_t^2 for KL-derived weight ---
        if self.variance_type == "posterior":
            sigma2 = posterior_variance
        elif self.variance_type == "beta":
            sigma2 = torch.clamp(betas, min=1e-20)
        else:
            raise ValueError(f"Unknown variance_type: {self.variance_type}")
        self.register_buffer("sigma2", sigma2)

        # --- KL->MSE coefficient (your derived weight) ---
        # w_t = beta_t^2 / (2 sigma_t^2 alpha_t (1 - alpha_bar_t))
        loss_weight = (betas ** 2) / (2.0 * sigma2 * alphas * (1.0 - alphas_cumprod) + self.eps)
        self.register_buffer("loss_weight", loss_weight)

        # model
        self.time_embed = SinusoidalTimeEmbedding(self.time_emb_dim)
        self.eps_model = _EpsMLP(self.latent_dim, self.time_emb_dim, self.hidden_dim, dropout=self.dropout)

    # ----------------------- helpers -----------------------
    @staticmethod
    def _flatten_with_shape(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if z.dim() == 2:
            return z, tuple(z.shape)
        if z.dim() == 3:
            b, e, k = z.shape
            return z.reshape(b * e, k), (b, e, k)
        raise ValueError(f"Expected z to be [B,E,K] or [B*E,K], got shape {tuple(z.shape)}")

    @staticmethod
    def _unflatten(z_flat: torch.Tensor, shape_info: Tuple[int, ...]) -> torch.Tensor:
        if len(shape_info) == 2:
            return z_flat
        if len(shape_info) == 3:
            b, e, k = shape_info
            return z_flat.reshape(b, e, k)
        raise ValueError(f"Unexpected shape_info: {shape_info}")

    def _resolve_train_t_max(self, t_max: Optional[int]) -> int:
        if t_max is None:
            t_max = self.train_t_max
        if t_max is None:
            t_max = self.timesteps
        t_max = int(max(2, min(self.timesteps, t_max)))
        return t_max

    # ----------------------- diffusion core -----------------------
    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """z_t = sqrt(alpha_bar_t) z0 + sqrt(1-alpha_bar_t) noise ; t is 1-indexed."""
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        return sqrt_ab * z0 + sqrt_1mab * noise

    def predict_eps(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.eps_model(zt, t_emb)

    def predict_z0_from_eps(self, zt: torch.Tensor, t: torch.Tensor, eps_theta: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, zt.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, zt.shape)
        return (zt - sqrt_1mab * eps_theta) / (sqrt_ab + self.eps)

    # ----------------------- training loss -----------------------
    def ddpm_loss_random_t(
        self,
        z0_logits: torch.Tensor,
        t_max: Optional[int] = None,
        num_t_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, DiffusionLossStats]:
        """
        (A) t sampling range: t ~ Uniform({2,...,t_max})
        (B) multi-t: for each edge sample K timesteps, average (reduces variance).
        """
        z0_flat, _shape_info = self._flatten_with_shape(z0_logits)  # [N,K]
        n, d = z0_flat.shape
        device = z0_flat.device

        t_max = self._resolve_train_t_max(t_max)
        Kt = int(num_t_samples) if num_t_samples is not None else self.train_num_t
        Kt = int(max(1, Kt))

        # sample t: [N, Kt]
        t = torch.randint(2, t_max + 1, (n, Kt), device=device, dtype=torch.long)

        # sample noise: [N, Kt, d]
        noise = torch.randn((n, Kt, d), device=device, dtype=z0_flat.dtype)

        # expand z0: [N, Kt, d] -> flatten to [N*Kt, d]
        z0_rep = z0_flat[:, None, :].expand(n, Kt, d).reshape(n * Kt, d)
        noise_rep = noise.reshape(n * Kt, d)
        t_rep = t.reshape(n * Kt)

        zt = self.q_sample(z0_rep, t_rep, noise_rep)
        pred_noise = self.predict_eps(zt, t_rep)

        # unweighted MSE for logging
        mse_unweighted = F.mse_loss(pred_noise, noise_rep, reduction="mean")

        # weighted objective
        sq_norm = (pred_noise - noise_rep).pow(2).sum(dim=-1)  # [N*Kt]
        wt = extract(self.loss_weight, t_rep, (sq_norm.shape[0], 1)).view(-1)  # [N*Kt]
        loss = (wt * sq_norm).mean()

        if self.scale_by_timesteps:
            # since we sample uniformly over {2,...,t_max}, multiply by (t_max-1) to approximate sum_{t=2..t_max}
            loss = loss * float(t_max - 1)

        stats = DiffusionLossStats(
            ddpm_mse=mse_unweighted.detach(),
            ddpm_weighted=loss.detach(),
            t_mean=t_rep.float().mean().detach(),
            w_mean=wt.mean().detach(),
        )
        return loss, stats

    def loss(
        self,
        z0_logits: torch.Tensor,
        prob: Optional[torch.Tensor] = None,  # kept for backward compat; unused
        return_stats: bool = True,
        t_max: Optional[int] = None,
        num_t_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ddpm_loss, stats_obj = self.ddpm_loss_random_t(z0_logits, t_max=t_max, num_t_samples=num_t_samples)

        stats: Dict[str, torch.Tensor] = {}
        if return_stats:
            stats = {
                "diff_ddpm_mse": stats_obj.ddpm_mse,
                "diff_ddpm_weighted": stats_obj.ddpm_weighted,
                "diff_t_mean": stats_obj.t_mean,
                "diff_w_mean": stats_obj.w_mean,
                "diff_total": ddpm_loss.detach(),
            }
        return ddpm_loss, stats

    # ----------------------- ONE-STEP refinement (for decoder input) -----------------------
    def refine_one_step(
        self,
        z0_logits: torch.Tensor,
        t_ref: int,
        gamma: float = 0.1,
        noise_mode: Literal["zero", "fixed"] = "zero",
        noise_seed: int = 0,
        clip_denoised: Optional[float] = None,
        use_eval: bool = True,
    ) -> torch.Tensor:
        """
        Deterministic ONE-STEP residual refinement:
            zt = q(z_t|z0) at fixed t_ref using deterministic noise (scheme B)
            eps_hat = eps_theta(zt, t_ref)
            z0_hat = (zt - sqrt(1-a_bar)*eps_hat)/sqrt(a_bar)
            refined = z0 + gamma*(z0_hat - z0)

        noise_mode:
          - "zero": use eps=0 => zt is the conditional mean (shape-independent deterministic)
          - "fixed": use a fixed Gaussian noise with a fixed seed (deterministic given tensor shape)

        use_eval=True:
          - Force eval() during refinement to disable dropout, so decoder input is stable even in training.
        """
        was_training = self.training
        if use_eval:
            # Disable dropout for the refinement path (stable decoder input).
            self.eval()

        z0_flat, shape_info = self._flatten_with_shape(z0_logits)
        n, d = z0_flat.shape
        device = z0_flat.device

        t_ref = int(max(1, min(self.timesteps, int(t_ref))))
        t_vec = torch.full((n,), t_ref, device=device, dtype=torch.long)

        if noise_mode == "zero":
            noise = torch.zeros_like(z0_flat)
        elif noise_mode == "fixed":
            g = torch.Generator(device=device)
            g.manual_seed(int(noise_seed))
            noise = torch.randn((n, d), device=device, dtype=z0_flat.dtype, generator=g)
        else:
            raise ValueError(f"Unknown noise_mode: {noise_mode}")

        zt = self.q_sample(z0_flat, t_vec, noise)
        eps_hat = self.predict_eps(zt, t_vec)
        z0_hat = self.predict_z0_from_eps(zt, t_vec, eps_hat)

        if clip_denoised is not None:
            c = float(clip_denoised)
            z0_hat = torch.clamp(z0_hat, -c, c)

        refined_flat = z0_flat + float(gamma) * (z0_hat - z0_flat)

        if use_eval and was_training:
            self.train()

        return self._unflatten(refined_flat, shape_info)
