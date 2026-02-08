from __future__ import annotations

from typing import Optional, Literal

import torch
from torch import Tensor, nn

import config as cfg
from utils.torch_extension import gumbel_softmax, sym_hard


class NRIModel(nn.Module):
    """Auto-encoder (MPM NRI backbone) with optional diffusion prior on encoder logits."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        es: Tensor,
        size: int,
        *,
        diff_prior: Optional[nn.Module] = None,
        use_diff_prior: bool = False,
        # one-step refinement (Scheme B) hyperparams
        diff_t_ref: int = 30,
        diff_refine_gamma: float = 0.1,
        diff_refine_noise: Literal["zero", "fixed"] = "fixed",
        diff_refine_noise_seed: int = 0,
        diff_refine_clip: Optional[float] = None,
    ):
        """
        Args:
            encoder: encoder inferring relations (outputs logits)
            decoder: decoder predicting future states
            es: edge list
            size: number of nodes
            diff_prior: DiffusionPrior module (edge-wise DDPM on logits), or None
            use_diff_prior: whether to enable diffusion refinement path in forward()
            diff_t_ref: fixed timestep used by one-step refinement for decoder input
            diff_refine_gamma: residual scale gamma for refinement
            diff_refine_noise: noise mode used to form z_t for refinement ("zero" or "fixed")
            diff_refine_noise_seed: seed used when diff_refine_noise == "fixed"
            diff_refine_clip: optional clip value applied to z0_hat during refinement
        """
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.es = torch.LongTensor(es)
        self.size = int(size)

        # diffusion prior (optional)
        self.diff_prior = diff_prior
        self.use_diff_prior = bool(use_diff_prior)

        # refinement hyperparams
        self.diff_t_ref = int(diff_t_ref)
        self.diff_refine_gamma = float(diff_refine_gamma)
        self.diff_refine_noise = diff_refine_noise
        self.diff_refine_noise_seed = int(diff_refine_noise_seed)
        self.diff_refine_clip = diff_refine_clip

    def _ensure_es_device(self, device: torch.device):
        if not self.es.is_cuda:
            self.es = self.es.cuda(device)

    def _refine_logits_one_step(self, logits_eBK: Tensor, *, tosym: bool) -> Tensor:
        """
        Apply one-step refinement at fixed t_ref for decoder input (Scheme B).
        Input/Output logits shape: [E, B, K]
        """
        if (not self.use_diff_prior) or (self.diff_prior is None):
            return logits_eBK

        # DiffusionPrior expects [B, E, K] or [B*E, K]; we use [B, E, K].
        logits_bEK = logits_eBK.transpose(0, 1).contiguous()
        logits_ref_bEK = self.diff_prior.refine_one_step(
            logits_bEK,
            t_ref=self.diff_t_ref,
            gamma=self.diff_refine_gamma,
            noise_mode=self.diff_refine_noise,
            noise_seed=self.diff_refine_noise_seed,
            clip_denoised=self.diff_refine_clip,
            use_eval=True,  # training-time also uses eval() for stable decoder input
        )
        logits_ref_eBK = logits_ref_bEK.transpose(0, 1).contiguous()

        # If hard symmetry is requested, enforce it on the final decoder-input logits too.
        if tosym:
            logits_ref_eBK = sym_hard(logits_ref_eBK, self.size)
        return logits_ref_eBK

    def predict_relations(self, states: Tensor, *, tosym: bool = False) -> Tensor:
        """
        Given historical node states, infer interacting relations.

        Args:
            states: [batch, step, node, dim]

        Return:
            prob: [E, batch, K]
        """
        self._ensure_es_device(states.device)
        logits = self.enc(states, self.es)  # [E, B, K]
        if tosym:
            logits = sym_hard(logits, self.size)

        # IMPORTANT: for reporting/inference, match forward() behavior:
        # if diffusion prior is enabled, return probabilities from refined logits.
        logits_used = self._refine_logits_one_step(logits, tosym=tosym)
        prob = logits_used.softmax(-1)
        return prob

    def predict_states(self, states: Tensor, edges: Tensor, M: int = 1) -> Tensor:
        """
        Given historical node states and inferred relations, predict future node states.

        Args:
            states: [batch, step, node, dim]
            edges: [E, batch, K]
            M: number of steps to predict

        Return:
            states: [batch, step_out, node, dim]
        """
        self._ensure_es_device(states.device)
        return self.dec(states, edges, self.es, M)

    def forward(
        self,
        states_enc: Tensor,
        states_dec: Tensor,
        hard: bool = False,
        p: bool = False,
        M: int = 1,
        tosym: bool = False,
        *,
        return_logits_for_diff: bool = False,
    ):
        """
        Args:
            states_enc: [batch, step_enc, node, dim], encoder inputs
            states_dec: [batch, step_dec, node, dim], decoder inputs
            hard: use hard gumbel-softmax sampling or not
            p: return relation distribution or not
            M: number of steps to predict
            tosym: impose hard symmetry constraint or not
            return_logits_for_diff: if True, additionally return z0 logits (pre-refine) in shape [B, E, K],
                                   to be used by diffusion loss in the Instructor.

        Return:
            output: [batch, step_out, node, dim]
            prob (optional): [batch, E, K]
            logits_for_diff (optional): [batch, E, K]  (pre-refine, after sym_hard if tosym=True)
        """
        self._ensure_es_device(states_enc.device)

        logits_raw = self.enc(states_enc, self.es)  # [E, B, K]
        if tosym:
            logits_raw = sym_hard(logits_raw, self.size)

        logits_for_diff = logits_raw.transpose(0, 1).contiguous()  # [B, E, K]

        # Decoder input uses fixed-t one-step refinement (Scheme B) when diffusion prior is enabled.
        logits_used = self._refine_logits_one_step(logits_raw, tosym=tosym)

        edges = gumbel_softmax(logits_used, tau=cfg.temp, hard=hard)
        output = self.dec(states_dec, edges, self.es, M)

        if p:
            prob = logits_used.softmax(-1).transpose(0, 1).contiguous()  # [B, E, K]
            if return_logits_for_diff:
                return output, prob, logits_for_diff
            return output, prob

        if return_logits_for_diff:
            return output, logits_for_diff
        return output
