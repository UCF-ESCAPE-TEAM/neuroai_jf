"""StTcnPose trained with C-Mixup (StTcnPoseCMixup).

Same architecture, weights and inference as :class:`StTcnPose`; the only
difference is *how it is trained*. During training every batch is replaced by
a convex mix of pairs of windows (mixup, Zhang et al. ICLR 2018), with the
partner of each window drawn by **C-Mixup** (Yao et al. NeurIPS 2022): the
probability of pairing window ``i`` with window ``j`` decays with the distance
between their *labels*,

    P(j | i)  ∝  exp( -||ȳ_i - ȳ_j||² / (2 σ²) )      (Gaussian kernel on labels)

so a window is mixed with another window whose hand pose is similar, and the
interpolated target ``ỹ = λ y_i + (1-λ) y_j`` stays a plausible pose. Vanilla
mixup (σ → ∞) pairs anything with anything and, for regression, produces
off-manifold targets (the "average" of a fist and an open hand is not a pose
the model will ever have to predict).

What is mixed and where
-----------------------
* ``ȳ_i`` is the mean joint-angle vector of the window (20-d, NaN frames
  ignored) -- the label summary used for the kernel.
* The *targets* are mixed in full, frame by frame:
  ``ỹ[t] = λ y_i[t] + (1-λ) y_j[t]``.
* The *inputs* are mixed **after** the parameter-free front end (FIR +
  per-window scaling), not on the raw EMG (``mix_where="frontend"``). Raw
  windows of two users differ in amplitude by an order of magnitude; mixing
  them before scaling would let the louder user dominate ``x̃`` while the
  target still says ``λ``. After ``maxabs`` scaling both are in ``[-1, 1]``
  and the ``λ`` on the input matches the ``λ`` on the target.
  ``mix_where="input"`` mixes the raw EMG instead (mixreg / vanilla mixup).
* ``same_subject_weight`` scales the kernel for pairs from the same subject.
  ``0`` (default) pairs each window with a *different* user whenever the batch
  contains one -- the EMG analogue of mixreg's "mix observations from
  different training environments": the generalization gap in emg2pose is
  across users, so that is the axis along which to fill in the data manifold.
* ``λ ~ Beta(α, α)``; ``prob`` is the fraction of batches that are mixed.

How it plugs into NeuralBench
-----------------------------
``BrainModule._run_step`` (neuralbench ``pl_module.py``) calls the model's
``train_batch_transform(neuro, target, subject_ids)`` during training. This
model implements it: it draws the pairing and ``λ``, mixes the **targets** and
returns them, and keeps ``(perm, λ)`` so that the next ``forward`` mixes the
**inputs** at the chosen stage with the same ``perm`` and ``λ``. In eval mode
(and at inference / in ``submission.py``) nothing is mixed, so a checkpoint
from this class loads into the plain ``StTcnPose`` submission unchanged.
"""

from __future__ import annotations

import typing as tp

import torch
from torch import nn

from .emg_frontend import frontend_from_config
from .st_tcn_pose import StTcnPose, StTcnPoseModel


class StTcnPoseCMixup(StTcnPose):
    """Config for :class:`StTcnPoseCMixupModel` -- StTcnPose + C-Mixup training."""

    # ---- C-Mixup (training only; none of these change the architecture) ----
    mix_alpha: float = 2.0            # λ ~ Beta(α, α). 2 -> λ around 0.5; 0.2 -> near 0 or 1
    mix_prob: float = 1.0             # fraction of training batches that are mixed
    mix_sigma: float = 0.5            # kernel bandwidth, rad, on ||mean pose_i - mean pose_j||
    mix_same_subject_weight: float = 0.0   # kernel factor for same-subject pairs (0 = cross-user only)
    mix_where: tp.Literal["frontend", "input"] = "frontend"
    mix_label_stat: tp.Literal["mean", "median"] = "mean"

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "StTcnPoseCMixup requires n_outputs"
        frontend = frontend_from_config(n_spatial_locations, self.model_dump())
        kwargs = {k: getattr(self, k)
                  for k in StTcnPoseModel.MODEL_FIELDS + StTcnPoseCMixupModel.MIXUP_FIELDS}
        return StTcnPoseCMixupModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                                    sfreq=self.sfreq, frontend=frontend, **kwargs)


# =============================================================================
# C-Mixup sampling
# =============================================================================
def window_label_summary(target: torch.Tensor, stat: str = "mean") -> torch.Tensor:
    """One label vector per window: ``(B, J, T)`` or ``(B, T, J)`` -> ``(B, J)``.

    Frames whose target is NaN (unlabelled) are ignored. A window with no
    labelled frame gets the batch mean so it is a "neutral" partner.
    ``(B, J)`` targets are returned as they are.
    """
    if target.ndim == 2:
        y = target
    else:
        # NeuralBench feeds the channel-major extracted target (B, J, T); accept
        # the time-major layout too.
        y = target if target.shape[1] <= target.shape[2] else target.transpose(1, 2)
        y = torch.nanmedian(y, dim=-1).values if stat == "median" else torch.nanmean(y, dim=-1)
    bad = y.isnan()
    if bad.any():
        fill = torch.nanmean(y, dim=0, keepdim=True).expand_as(y)
        y = torch.where(bad, fill, y)
        y = torch.nan_to_num(y, nan=0.0)   # an all-NaN column
    return y


def cmixup_partners(
    labels: torch.Tensor,
    sigma: float,
    subject_ids: torch.Tensor | None = None,
    same_subject_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one partner per window with the C-Mixup kernel.

    ``P(j | i) ∝ exp(-||y_i - y_j||² / (2σ²))`` for ``j != i`` (C-Mixup, Yao et
    al. 2022, Eq. 3 with a Gaussian kernel on the label distance), multiplied by
    ``same_subject_weight`` when ``i`` and ``j`` come from the same subject.
    If that leaves a row with no admissible partner (every other window is the
    same subject and the weight is 0), that row falls back to the plain label
    kernel. Returns ``perm`` of shape ``(B,)`` with ``perm[i] = j``.
    """
    n = labels.shape[0]
    device = labels.device
    if n < 2:
        return torch.arange(n, device=device)
    d2 = torch.cdist(labels.float(), labels.float()).pow(2)              # (B, B)
    logits = -d2 / (2.0 * float(sigma) ** 2)
    logits = logits - logits.max(dim=1, keepdim=True).values            # numerical safety
    kernel = logits.exp()
    eye = torch.eye(n, dtype=torch.bool, device=device)
    kernel = kernel.masked_fill(eye, 0.0)
    if subject_ids is not None and same_subject_weight != 1.0:
        sid = subject_ids.reshape(-1).to(device)
        same = (sid[:, None] == sid[None, :]) & ~eye
        weighted = torch.where(same, kernel * float(same_subject_weight), kernel)
        has_partner = weighted.sum(dim=1) > 0
        kernel = torch.where(has_partner[:, None], weighted, kernel)
    kernel = torch.where(kernel.sum(dim=1, keepdim=True) > 0, kernel, (~eye).float())
    return torch.multinomial(kernel, num_samples=1, generator=generator).squeeze(1)


# =============================================================================
# Model
# =============================================================================
class StTcnPoseCMixupModel(StTcnPoseModel):
    """:class:`StTcnPoseModel` with the C-Mixup training hook (see module docstring)."""

    MIXUP_FIELDS = ("mix_alpha", "mix_prob", "mix_sigma", "mix_same_subject_weight",
                    "mix_where", "mix_label_stat")

    def __init__(self, *args, mix_alpha: float = 2.0, mix_prob: float = 1.0,
                 mix_sigma: float = 0.5, mix_same_subject_weight: float = 0.0,
                 mix_where: str = "frontend", mix_label_stat: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if mix_where not in ("frontend", "input"):
            raise ValueError(f"mix_where must be 'frontend' or 'input', got {mix_where!r}")
        if mix_label_stat not in ("mean", "median"):
            raise ValueError(f"mix_label_stat must be 'mean' or 'median', got {mix_label_stat!r}")
        self.mix_alpha = float(mix_alpha)
        self.mix_prob = float(mix_prob)
        self.mix_sigma = float(mix_sigma)
        self.mix_same_subject_weight = float(mix_same_subject_weight)
        self.mix_where = mix_where
        self.mix_label_stat = mix_label_stat
        # pairing of the batch currently in flight: set by train_batch_transform,
        # consumed by forward. Not a buffer -> never in the state dict.
        self._mix_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last_mix_stats: dict[str, float] = {}

    # ---- NeuralBench training hook -------------------------------------------
    def train_batch_transform(
        self,
        neuro: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix the *targets* now; the matching *input* mix happens in ``forward``.

        Called by ``BrainModule._run_step`` with the batch's neuro ``(B, C, T)``,
        target (``(B, J, T)`` radians, NaN = unlabelled frame) and subject ids,
        in training mode only. Returns ``(neuro, target)``: the input is
        returned unchanged when ``mix_where="frontend"`` (it is mixed inside
        ``forward`` after the front end) and mixed here when ``mix_where="input"``.
        """
        self._mix_state = None
        if not self.training or self.mix_prob <= 0 or target.shape[0] < 2:
            self.last_mix_stats = {"mix_applied": 0.0}
            return neuro, target
        if self.mix_prob < 1.0 and torch.rand((), device="cpu").item() >= self.mix_prob:
            self.last_mix_stats = {"mix_applied": 0.0}
            return neuro, target

        labels = window_label_summary(target, self.mix_label_stat)          # (B, J)
        perm = cmixup_partners(labels, self.mix_sigma, subject_ids, self.mix_same_subject_weight)
        lam = torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample((target.shape[0],))
        lam = lam.to(device=target.device, dtype=target.dtype)

        y_lam = lam.reshape(-1, *([1] * (target.ndim - 1)))
        target_mixed = y_lam * target + (1.0 - y_lam) * target[perm]       # NaN propagates -> masked

        stats = {
            "mix_applied": 1.0,
            "mix_lambda_mean": lam.mean().item(),
            "mix_partner_dist": (labels - labels[perm]).norm(dim=1).mean().item(),
        }
        if subject_ids is not None:
            sid = subject_ids.reshape(-1)
            stats["mix_cross_subject_frac"] = (sid != sid[perm]).float().mean().item()
        self.last_mix_stats = stats

        if self.mix_where == "input":
            x_lam = lam.reshape(-1, *([1] * (neuro.ndim - 1))).to(neuro.dtype)
            neuro = x_lam * neuro + (1.0 - x_lam) * neuro[perm]
            return neuro, target_mixed
        self._mix_state = (perm, lam)
        return neuro, target_mixed

    # ---- forward with the input-side mix ---------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``.

        Identical to :meth:`StTcnPoseModel.forward` except that, when a pairing
        is pending from :meth:`train_batch_transform`, the front-end output is
        mixed with it before the spatial block.
        """
        n_samples = x.shape[-1]
        x = self.frontend(x)                                 # (B, C, T) filtered + scaled
        state, self._mix_state = self._mix_state, None       # consume exactly once
        if state is not None and self.training:
            perm, lam = state
            if perm.shape[0] == x.shape[0]:
                lam = lam.to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                x = lam * x + (1.0 - lam) * x[perm.to(x.device)]
        z = self.spatial(x)
        z = self.tcn(z)
        y = self.head_post(self.head(z))
        y = self._upsample(y, n_samples)
        return y.transpose(1, 2)
