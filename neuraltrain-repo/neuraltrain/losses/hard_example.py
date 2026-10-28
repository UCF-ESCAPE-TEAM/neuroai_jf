"""Hard-example-mining losses for dense regression (EMG -> pose).

Two objectives, both drop-in replacements for ``L1Loss`` on ``(N, J)`` /
``(B, T, J)`` predictions in radians:

``GhmRegressionLoss``
    Gradient Harmonizing Mechanism for regression (GHM-R, Li, Liu & Wang,
    AAAI 2019, Sec. "GHM-R Loss"), the variant Lin et al. (Measurement 2026,
    MSST, Eq. 9-15) use for EMG torque / angle estimation. The base loss is the
    Authentic Smooth L1, ``ASL1(d) = sqrt(d^2 + mu^2) - mu``, whose gradient
    norm ``g = |d| / sqrt(d^2 + mu^2)`` lies in ``[0, 1)``. ``[0, 1]`` is cut
    into ``bins`` unit regions, the number of examples per region (optionally
    an EMA over iterations, GHM Eq. 12-13) is the gradient density, and each
    example is weighted by ``beta_i = N / GD(g_i)`` (Eq. 7 / Eq. 10). Examples
    sitting in a crowded region of gradient space are down-weighted; the
    sparse ones (the hard, badly-fitted frames) keep their full gradient.
    Mean weight is 1, so the loss scale matches plain ``ASL1`` / ``L1``.

``OhemRegressionLoss``
    Online Hard Example Mining (Shrivastava, Gupta & Girshick, CVPR 2016,
    Sec. 4.1) rewritten for regression: compute the per-example loss on every
    example of the batch, back-propagate only through the ``keep_fraction``
    hardest ones (and, optionally, through the rest with a reduced weight,
    which is what ``easy_weight`` is for). Mining is done per element
    (frame x joint) or per frame; frames of one 5-s window are strongly
    correlated, the regression analogue of OHEM's overlapping RoIs, so
    ``select="frame"`` is the closer match to their NMS de-duplication.

Both classes are converted to pydantic configs at the bottom of this module so
a task YAML can reference them by class name (``loss: {name: GhmRegressionLoss,
mu: 0.1}``), exactly like the losses in ``losses.py``.
"""

from __future__ import annotations

import torch
from torch import nn

from neuraltrain.utils import convert_to_pydantic


def asl1(d: torch.Tensor, mu: float) -> torch.Tensor:
    """Authentic Smooth L1 (GHM paper Eq. 17): L2 for |d| << mu, L1 for |d| >> mu."""
    return torch.sqrt(d * d + mu * mu) - mu


def asl1_gradient_norm(d: torch.Tensor, mu: float) -> torch.Tensor:
    """|dASL1/dd| = |d| / sqrt(d^2 + mu^2) in [0, 1) (GHM paper Eq. 18)."""
    return d.abs() / torch.sqrt(d * d + mu * mu)


class GhmRegressionLoss(nn.Module):
    """GHM-R / MSST GHM-L loss (see module docstring).

    Parameters
    ----------
    mu : float
        ASL1 transition scale, in the unit of the targets (radians here). It
        is also the scale of the gradient-norm axis: ``g = 0.71`` at
        ``|d| = mu`` and ``g > 0.95`` for ``|d| > 3 mu``. Pick ``mu`` near the
        *median* absolute error of the model you are training (if ``mu`` is far
        below it every example saturates at ``g ~ 1``, lands in one bin and the
        harmonization degenerates to a constant). The detection papers use
        0.02 because their offsets live in [0, 1]; 16 deg MAE is 0.28 rad, so
        0.1 rad is a sensible start for pose.
    bins : int
        Number of unit regions ``M`` on ``[0, 1]``. GHM found 10-30 sufficient
        (their Table 1); the loss is insensitive above ~20.
    momentum : float
        EMA factor ``alpha`` for the per-bin counts (GHM Eq. 12), 0 disables
        the EMA (pure per-batch statistics). 0.75 in the paper.
    normalize_by : {"nonempty_bins", "bins"}
        How the density is normalised. ``"nonempty_bins"`` follows the
        authors' released code (``GHM_Detection``: weights are divided by the
        number of non-empty bins so the mean weight is exactly 1).
        ``"bins"`` follows Eq. 9 literally (``GD = R * M``); the mean weight
        is then ``n_nonempty / M`` <= 1, which rescales the whole loss.
    weight_in_eval : bool
        If False (default), validation / test calls return the plain mean
        ``ASL1`` so ``val/loss`` stays comparable to the ``L1Loss`` runs; the
        EMA statistics are never updated outside training either way.
    eps : float
        Numerical floor for bin counts.
    """

    def __init__(
        self,
        mu: float = 0.1,
        bins: int = 30,
        momentum: float = 0.75,
        normalize_by: str = "nonempty_bins",
        weight_in_eval: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if normalize_by not in ("nonempty_bins", "bins"):
            raise ValueError(f"unknown normalize_by={normalize_by!r}")
        self.mu = float(mu)
        self.bins = int(bins)
        self.momentum = float(momentum)
        self.normalize_by = normalize_by
        self.weight_in_eval = weight_in_eval
        self.eps = float(eps)
        # EMA of the fraction of examples per bin (S_j / N in Eq. 12, kept as a
        # fraction so batches of different size mix). Non-persistent: it is a
        # training statistic, not a weight, and must not be shipped.
        self.register_buffer(
            "ema_density", torch.zeros(self.bins), persistent=False
        )
        self.register_buffer(
            "ema_initialized", torch.tensor(False), persistent=False
        )

    def extra_repr(self) -> str:
        return (
            f"mu={self.mu}, bins={self.bins}, momentum={self.momentum}, "
            f"normalize_by={self.normalize_by!r}"
        )

    @torch.no_grad()
    def _bin_index(self, g: torch.Tensor) -> torch.Tensor:
        # ind(g) of the GHM paper: unit region [(j-1)/M, j/M); g == 1 -> last bin.
        return (g * self.bins).long().clamp_(0, self.bins - 1)

    @torch.no_grad()
    def _density(self, idx: torch.Tensor, n: int) -> torch.Tensor:
        """Per-bin density used for weighting: EMA-smoothed fraction of examples."""
        counts = torch.bincount(idx.reshape(-1), minlength=self.bins).to(
            self.ema_density.dtype
        )
        batch_density = counts / max(n, 1)
        if not self.training or self.momentum == 0.0:
            if self.training or not bool(self.ema_initialized):
                return batch_density
            return self.ema_density
        if bool(self.ema_initialized):
            self.ema_density.mul_(self.momentum).add_(
                batch_density, alpha=1.0 - self.momentum
            )
        else:
            self.ema_density.copy_(batch_density)
            self.ema_initialized.fill_(True)
        return self.ema_density

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(y_pred.shape)} vs true {tuple(y_true.shape)}"
            )
        d = y_pred - y_true
        loss_el = asl1(d, self.mu)  # (..., J), every element is an "example"
        n = loss_el.numel()
        if n == 0 or (not self.training and not self.weight_in_eval):
            return {"total": loss_el.mean() if n else loss_el.sum()}

        with torch.no_grad():
            g = asl1_gradient_norm(d.detach(), self.mu)
            idx = self._bin_index(g)
            density = self._density(idx, n)  # (bins,), fractions
            nonempty = density > self.eps
            # beta_j = 1 / (GD_j / N) with GD_j / N = density_j * M (Eq. 9-10).
            # Dividing by the number of non-empty bins instead of M (the
            # authors' code) keeps the mean weight at exactly 1.
            scale = (
                nonempty.sum().clamp(min=1).to(density.dtype)
                if self.normalize_by == "nonempty_bins"
                else torch.tensor(float(self.bins), device=density.device)
            )
            bin_weight = torch.where(
                nonempty,
                1.0 / (density.clamp(min=self.eps) * scale),
                torch.zeros_like(density),
            )
            weight = bin_weight[idx]  # (..., J)
            # Diagnostics: the weight the hardest (g ~ 1) examples currently
            # get, and the fraction of examples in that saturated bin. If
            # frac_top ~ 1, mu is far too small for this error scale.
            w_top = bin_weight[-1]
            frac_top = (idx == self.bins - 1).float().mean()

        total = (weight * loss_el).sum() / n
        return {"total": total, "ghm_w_top": w_top, "ghm_frac_top": frac_top}


class OhemRegressionLoss(nn.Module):
    """Online hard example mining for dense regression (see module docstring).

    Parameters
    ----------
    keep_fraction : float
        Fraction of examples (per batch) with the largest loss that are
        back-propagated at full weight. OHEM keeps B = 128 of ~4000 RoIs
        (~3 %); for pose, where most frames carry useful signal, 0.25-0.5 is a
        more sensible range, and ``easy_weight > 0`` keeps the rest alive.
    easy_weight : float
        Weight of the non-selected examples. 0 reproduces OHEM (their
        "set the loss of all non-hard RoIs to 0"); values in (0, 1) are a soft
        variant that never fully discards the easy majority.
    select : {"element", "frame"}
        Mining granularity: each (frame, joint) element separately, or each
        frame by its joint-mean loss (all joints of a selected frame are kept).
    base : {"l1", "asl1"}
        Per-element loss; ``asl1`` uses ``mu``.
    mu : float
        ASL1 scale when ``base="asl1"``.
    weight_in_eval : bool
        As in :class:`GhmRegressionLoss`: validation returns the plain mean.
    """

    def __init__(
        self,
        keep_fraction: float = 0.3,
        easy_weight: float = 0.0,
        select: str = "frame",
        base: str = "l1",
        mu: float = 0.1,
        weight_in_eval: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        if not 0.0 <= easy_weight <= 1.0:
            raise ValueError(f"easy_weight must be in [0, 1], got {easy_weight}")
        if select not in ("element", "frame"):
            raise ValueError(f"unknown select={select!r}")
        if base not in ("l1", "asl1"):
            raise ValueError(f"unknown base={base!r}")
        self.keep_fraction = float(keep_fraction)
        self.easy_weight = float(easy_weight)
        self.select = select
        self.base = base
        self.mu = float(mu)
        self.weight_in_eval = weight_in_eval

    def extra_repr(self) -> str:
        return (
            f"keep_fraction={self.keep_fraction}, easy_weight={self.easy_weight}, "
            f"select={self.select!r}, base={self.base!r}"
        )

    def _elementwise(self, d: torch.Tensor) -> torch.Tensor:
        return d.abs() if self.base == "l1" else asl1(d, self.mu)

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(y_pred.shape)} vs true {tuple(y_true.shape)}"
            )
        loss_el = self._elementwise(y_pred - y_true)
        n = loss_el.numel()
        if n == 0 or (not self.training and not self.weight_in_eval):
            return {"total": loss_el.mean() if n else loss_el.sum()}

        if self.select == "frame" and loss_el.ndim >= 2:
            score = loss_el.detach().mean(dim=-1)  # (...,) one score per frame
        else:
            score = loss_el.detach()
        flat = score.reshape(-1)
        k = max(1, int(round(self.keep_fraction * flat.numel())))
        if k >= flat.numel():
            mask = torch.ones_like(flat, dtype=torch.bool)
        else:
            thresh = flat.kthvalue(flat.numel() - k + 1).values  # k-th largest
            mask = flat >= thresh
        mask = mask.reshape(score.shape)
        if self.select == "frame" and loss_el.ndim >= 2:
            mask = mask.unsqueeze(-1).expand_as(loss_el)
        weight = torch.where(
            mask,
            torch.ones_like(loss_el),
            torch.full_like(loss_el, self.easy_weight),
        )
        # Normalise by the total weight so the loss scale does not depend on
        # keep_fraction (mean over the mined set when easy_weight == 0).
        total = (weight * loss_el).sum() / weight.sum().clamp(min=1.0)
        hard_mean = loss_el.detach()[mask].mean()
        return {"total": total, "ohem_hard_mean": hard_mean}


# ---- pydantic configs -----------------------------------------------------
# ``neuraltrain.losses.base`` converts every nn.Module found in ``losses.py``;
# this module lives beside it, so register its classes the same way. The
# resulting config classes carry the nn.Module's constructor arguments as
# fields and discriminate on the class name in YAML.
from .base import BaseLoss  # noqa: E402  (after the classes, avoids a cycle)

GhmRegressionLossConfig = convert_to_pydantic(
    GhmRegressionLoss, "GhmRegressionLoss", parent_class=BaseLoss
)
OhemRegressionLossConfig = convert_to_pydantic(
    OhemRegressionLoss, "OhemRegressionLoss", parent_class=BaseLoss
)
