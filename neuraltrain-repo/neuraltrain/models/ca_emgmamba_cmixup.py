"""CaEmgMamba trained with C-Mixup (CaEmgMambaCMixup).

Same architecture, weights and inference as :class:`CaEmgMambaModel`; only the
training changes. Every training batch is replaced by a convex mix of window
pairs (mixup, Zhang et al. 2018) whose partners are drawn with the C-Mixup
kernel (Yao et al., NeurIPS 2022): window ``i`` is paired with ``j`` with
probability ``P(j | i) ∝ exp(-||ȳ_i - ȳ_j||² / 2σ²)``, where ``ȳ`` is the mean
joint-angle vector of the window. Partners therefore have similar hand poses and
the mixed target ``λ y_i + (1-λ) y_j`` stays a plausible pose; vanilla mixup
(σ → ∞) would average a fist with an open hand.

What is mixed and where
-----------------------
* Targets are mixed frame by frame in ``train_batch_transform``; NaN frames
  propagate and are masked by the loss as usual.
* Inputs are mixed **after** :class:`WindowNorm` (``mix_where="norm"``), inside
  ``forward``, so both windows are z-scored before the convex combination and the
  ``λ`` on the input matches the ``λ`` on the target. Raw EMG of two users can
  differ in amplitude tenfold; mixing before the norm would let the louder user
  dominate. ``mix_where="input"`` mixes the raw EMG instead (vanilla / mixreg).
* ``mix_same_subject_weight`` scales the kernel for pairs from the same user;
  ``0`` pairs across users whenever the batch allows. Stage ids are not in the
  batch, so cross-stage pairing cannot be enforced; with ~128 windows from many
  recordings per batch most partners are from other stages anyway.
* Composes with the time-stretch of the base class: ``train_batch_transform``
  stretches first (if ``time_stretch_prob > 0``), then mixes.

The sampling helpers are shared with :mod:`st_tcn_pose_cmixup` (pure functions
on the targets; nothing architectural). A checkpoint from this class loads into
the plain ``ca_emgmamba_vel`` submission unchanged: no parameters are added.
"""

from __future__ import annotations

import typing as tp

import torch
from torch import nn

from .ca_emgmamba import CaEmgMamba, CaEmgMambaModel
from .st_tcn_pose_cmixup import cmixup_partners, window_label_summary


class CaEmgMambaCMixup(CaEmgMamba):
    """Config: all CaEmgMamba fields plus the C-Mixup ones (training only)."""

    mix_alpha: float = 2.0            # λ ~ Beta(α, α); 2 -> λ around 0.5, 0.2 -> near 0 or 1
    mix_prob: float = 1.0             # fraction of training batches that are mixed
    mix_sigma: float = 0.5            # kernel bandwidth, rad, on ||mean pose_i - mean pose_j||
    mix_same_subject_weight: float = 0.0   # kernel factor for same-user pairs (0 = cross-user only)
    mix_where: tp.Literal["norm", "input"] = "norm"
    mix_label_stat: tp.Literal["mean", "median"] = "mean"

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "CaEmgMambaCMixup requires n_outputs"
        kwargs = {k: getattr(self, k)
                  for k in CaEmgMambaModel.MODEL_FIELDS + CaEmgMambaCMixupModel.MIXUP_FIELDS}
        return CaEmgMambaCMixupModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                                     sfreq=self.sfreq, **kwargs)


class CaEmgMambaCMixupModel(CaEmgMambaModel):
    MIXUP_FIELDS = ("mix_alpha", "mix_prob", "mix_sigma", "mix_same_subject_weight",
                    "mix_where", "mix_label_stat")

    def __init__(self, *args, mix_alpha: float = 2.0, mix_prob: float = 1.0,
                 mix_sigma: float = 0.5, mix_same_subject_weight: float = 0.0,
                 mix_where: str = "norm", mix_label_stat: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if mix_where not in ("norm", "input"):
            raise ValueError(f"mix_where must be 'norm' or 'input', got {mix_where!r}")
        if mix_label_stat not in ("mean", "median"):
            raise ValueError(f"mix_label_stat must be 'mean' or 'median', got {mix_label_stat!r}")
        self.mix_alpha, self.mix_prob, self.mix_sigma = float(mix_alpha), float(mix_prob), float(mix_sigma)
        self.mix_same_subject_weight = float(mix_same_subject_weight)
        self.mix_where, self.mix_label_stat = mix_where, mix_label_stat
        # pairing of the batch in flight: set by train_batch_transform, consumed by
        # _after_norm. A plain attribute, never a buffer -> never in the state dict.
        self._mix_state: tuple[torch.Tensor, torch.Tensor] | None = None

    # ---- NeuralBench training hook -------------------------------------------
    def train_batch_transform(self, neuro, target, subject_ids=None):
        """Time-stretch (if enabled) then C-Mixup the targets; inputs follow in ``forward``."""
        neuro, target = super().train_batch_transform(neuro, target, subject_ids)
        stats = dict(self.last_mix_stats)
        self._mix_state = None
        if (not self.training or self.mix_prob <= 0 or target.shape[0] < 2
                or (self.mix_prob < 1.0 and torch.rand(()).item() >= self.mix_prob)):
            self.last_mix_stats = {**stats, "mix_applied": 0.0}
            return neuro, target

        labels = window_label_summary(target, self.mix_label_stat)                # (B, J)
        perm = cmixup_partners(labels, self.mix_sigma, subject_ids, self.mix_same_subject_weight)
        lam = torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample((target.shape[0],))
        lam = lam.to(device=target.device, dtype=target.dtype)
        y_lam = lam.reshape(-1, 1, 1)
        target = y_lam * target + (1.0 - y_lam) * target[perm]                   # NaN propagates -> masked

        stats.update(mix_applied=1.0, mix_lambda_mean=lam.mean().item(),
                     mix_partner_dist=(labels - labels[perm]).norm(dim=1).mean().item())
        if subject_ids is not None:
            sid = subject_ids.reshape(-1)
            stats["mix_cross_subject_frac"] = (sid != sid[perm]).float().mean().item()
        self.last_mix_stats = stats

        if self.mix_where == "input":
            x_lam = lam.reshape(-1, 1, 1).to(neuro.dtype)
            return x_lam * neuro + (1.0 - x_lam) * neuro[perm], target
        self._mix_state = (perm, lam)
        return neuro, target

    # ---- input-side mix, after the per-window z-score -------------------------
    def _after_norm(self, x: torch.Tensor) -> torch.Tensor:
        state, self._mix_state = self._mix_state, None            # consume exactly once
        if state is None or not self.training:
            return x
        perm, lam = state
        if perm.shape[0] != x.shape[0]:
            return x
        lam = lam.to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        return lam * x + (1.0 - lam) * x[perm.to(x.device)]
