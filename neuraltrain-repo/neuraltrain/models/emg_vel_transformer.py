"""StVelTransformer with a channel gate that reads the raw EMG window (EmgVelTransformer).

Identical to :class:`StVelTransformerModel` (front end, multi-scale spatial block,
Conformer encoder, velocity decoder) except for ``self.gate``: the SE-style
:class:`ChannelGate` summarised every electrode by two hand-picked statistics
(log-RMS, log-MAV). :class:`RawChannelGate` instead *learns* the squeeze from
the ``(B, C, T)`` window itself, in the spirit of SE (Hu et al. 2018) and ECA
(Wang et al. 2020) applied to a signal whose global average is ~0:

    squeeze   x (B, C, T) -> per-electrode descriptor (B, C, n_feat)
              a bank of ``n_feat`` temporal filters shared across electrodes
              (Conv1d over time, stride ``feat_stride``), squared (energy),
              averaged over time, log.  With n_feat = 1 and a flat kernel this
              reduces to log-RMS; with a learned bank it can measure energy in
              whichever temporal patterns help the pose loss.
    excite    descriptor -> one gain per electrode in (0, 2). Four options, which
              are the rows of Table 2 in the ECA paper (Wang et al., CVPR 2020)
              applied to electrodes:
              "se"        SE block, Eq. 2: W2 SiLU(W1 y) with a bottleneck of width
                          ``ca_hidden`` (dimensionality reduction; every electrode
                          sees every other).            params ~ 2 * C*n_feat*hidden
              "fc"        SE-Var3, Eq. 3/4: one full linear layer W y, no
                          reduction, every electrode sees every other.
                                                        params = C*n_feat * C
              "eca"       ECA, Eq. 9: the n_feat descriptors are averaged into ONE
                          scalar per electrode (y in R^C, as in the paper), then a
                          1-D conv of kernel ``gate_eca_k`` across the electrode
                          axis, no bias, circular padding (the 16 electrodes form
                          a ring). ``gate_eca_k: 0`` picks k adaptively from C
                          with the paper's Eq. 12 (gamma 2, b 1 -> k 3 for C 16).
                                                        params = k
              "eca_multi" ECA on the full descriptor: the 1-D conv over the
                          electrode axis has n_feat input channels, so it mixes
                          the n_feat energies instead of averaging them first.
                                                        params = k * n_feat (+1)
              The last layer is zero-initialised in every option, so the gate
              starts as the identity; the paper's sigma is replaced by 2*sigmoid
              so that identity is the mid-point of the gain range rather than its
              upper end (this is the one departure from Eq. 9, shared by all four
              options so the comparison isolates the excitation structure).
    scale     x * gate

``causal=True``: the filter bank is left-padded, the energy average becomes a
running mean at the stride-``feat_stride`` frame rate, the excitation is applied
per frame and the gains are held (zero-order) back to the sample rate, so the
gain at sample t depends on x[.. t] only.
"""

import math
import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .emg_frontend import frontend_from_config
from .st_pose_transformer import _same_pad
from .st_vel_transformer import StVelTransformer, StVelTransformerModel


# =============================================================================
# Config
# =============================================================================
class EmgVelTransformer(StVelTransformer):
    """All StVelTransformer fields, plus the raw-signal gate's own."""

    channel_attention: bool = True            # False -> nn.Identity (then == StVelTransformer)
    gate_n_feat: int = 8                      # temporal filters in the learned squeeze
    gate_k: int = 63                          # their length in samples (63 = 31.5 ms at 2 kHz)
    gate_stride: int = 8                      # conv stride: energy is pooled over T / stride frames
    gate_excite: tp.Literal["se", "fc", "eca", "eca_multi", "mlp"] = "se"   # "mlp" = alias of "se"
    gate_eca_k: int = 3                       # ECA kernel over the electrode ring (odd); 0 = adaptive (Eq. 12)
    # ``ca_hidden`` (inherited) is the bottleneck width for gate_excite: se

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "EmgVelTransformer requires n_outputs"
        frontend = frontend_from_config(n_spatial_locations, self.model_dump())
        model_kwargs = {k: getattr(self, k) for k in EmgVelTransformerModel.MODEL_FIELDS}
        return EmgVelTransformerModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                                      sfreq=self.sfreq, frontend=frontend, **model_kwargs)


# =============================================================================
# Raw-signal channel gate
# =============================================================================
def eca_kernel_size(n_ch: int, gamma: int = 2, b: int = 1) -> int:
    """ECA Eq. 12: k = |log2(C)/gamma + b/gamma|_odd (nearest odd number)."""
    k = int(abs((math.log2(n_ch) + b) / gamma))
    return k if k % 2 else k + 1


class RawChannelGate(nn.Module):
    EXCITES = ("se", "fc", "eca", "eca_multi")

    def __init__(self, n_ch: int, n_feat: int = 8, k: int = 63, stride: int = 8,
                 excite: str = "se", hidden: int = 32, eca_k: int = 3,
                 causal: bool = False, eps: float = 1e-6):
        super().__init__()
        if excite == "mlp":
            excite = "se"
        if excite not in self.EXCITES:
            raise ValueError(f"gate_excite must be one of {self.EXCITES}, got {excite!r}")
        if eca_k == 0:
            eca_k = eca_kernel_size(n_ch)
        if eca_k % 2 == 0:
            raise ValueError("gate_eca_k must be odd (or 0 for adaptive)")
        self.n_ch, self.n_feat, self.stride = n_ch, n_feat, stride
        self.excite_kind, self.causal, self.eps = excite, causal, eps
        self.eca_k = eca_k
        # squeeze: one filter bank shared by all electrodes (applied per electrode)
        self.pad = _same_pad(k, causal)
        self.feat = nn.Conv1d(1, n_feat, k, stride=stride, bias=False)
        # excitation
        if excite == "se":                                              # Eq. 2
            self.excite = nn.Sequential(nn.Linear(n_ch * n_feat, hidden), nn.SiLU(),
                                        nn.Linear(hidden, n_ch))
            last = self.excite[-1]
        elif excite == "fc":                                            # Eq. 3, W full (SE-Var3)
            self.excite = nn.Linear(n_ch * n_feat, n_ch)
            last = self.excite
        elif excite == "eca":                                           # Eq. 9, C1D_k on y in R^C
            self.excite = nn.Conv1d(1, 1, eca_k, bias=False)
            last = self.excite
        else:                                                           # eca_multi
            self.excite = nn.Conv1d(n_feat, 1, eca_k, bias=True)
            last = self.excite
        self.eca_pad = eca_k // 2
        with torch.no_grad():                                           # identity at init
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()

    # ---- squeeze: (B, C, T) -> log-energy descriptors (B, C, n_feat[, T'])
    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        y = self.feat(F.pad(x.reshape(B * C, 1, T), self.pad))          # (B*C, n_feat, T')
        e = y.pow(2)
        if not self.causal:
            d = e.mean(-1)                                               # (B*C, n_feat)
            return d.add(self.eps).log().view(B, C, self.n_feat)
        n = torch.arange(1, e.shape[-1] + 1, device=e.device, dtype=e.dtype)
        d = e.cumsum(-1) / n                                             # running mean (B*C, n_feat, T')
        return d.add(self.eps).log().view(B, C, self.n_feat, -1)

    # ---- excite: descriptors (N, C, n_feat) -> gains (N, C)
    def _gains(self, d: torch.Tensor) -> torch.Tensor:
        if self.excite_kind in ("se", "fc"):
            s = self.excite(d.flatten(1))                                # (N, C)
        else:
            z = d.mean(-1, keepdim=True) if self.excite_kind == "eca" else d   # (N, C, 1) | (N, C, n_feat)
            z = F.pad(z.transpose(1, 2), (self.eca_pad, self.eca_pad), mode="circular")   # (N, f, C + 2p)
            s = self.excite(z).squeeze(1)                                # (N, C)
        return 2.0 * torch.sigmoid(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                 # (B, C, T)
        B, C, T = x.shape
        d = self._descriptor(x)
        if not self.causal:
            return x * self._gains(d).unsqueeze(-1)                      # one gain per electrode per window
        Tp = d.shape[-1]
        g = self._gains(d.permute(0, 3, 1, 2).reshape(B * Tp, C, self.n_feat))   # (B*T', C)
        g = g.view(B, Tp, C).transpose(1, 2)                             # (B, C, T')
        # frame j of the strided conv ends at sample j*stride + (k-1) - left_pad = j*stride
        # (causal pad puts all k-1 on the left), so it is complete at sample j*stride;
        # sample t may use frame t // stride.
        idx = (torch.arange(T, device=x.device) // self.stride).clamp_(max=Tp - 1)
        return x * g[..., idx]


# =============================================================================
# Model
# =============================================================================
class EmgVelTransformerModel(StVelTransformerModel):
    MODEL_FIELDS = (
        *StVelTransformerModel.MODEL_FIELDS,
        "gate_n_feat", "gate_k", "gate_stride", "gate_excite", "gate_eca_k",
    )

    def __init__(self, n_in_channels: int, n_outputs: int, sfreq: float = 2000.0,
                 gate_n_feat: int = 8, gate_k: int = 63, gate_stride: int = 8,
                 gate_excite: str = "se", gate_eca_k: int = 3, **kwargs):
        super().__init__(n_in_channels, n_outputs, sfreq=sfreq, **kwargs)
        if kwargs.get("channel_attention", False):
            self.gate = RawChannelGate(
                n_in_channels, gate_n_feat, gate_k, gate_stride, gate_excite,
                kwargs.get("ca_hidden", 32), gate_eca_k, causal=self.causal,
            )
