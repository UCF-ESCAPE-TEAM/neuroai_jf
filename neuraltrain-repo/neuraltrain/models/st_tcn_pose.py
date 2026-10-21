"""Spatial-CNN + TCN EMG-to-pose model (StTcnPose).

    EMG (B, C, T) --> EmgFrontEnd (no params) --> SpatialBlock (EEGNet-style 2-D
    CNN over the electrode ring, pools to ~50 Hz) --> TCN (EEG-TCNet residual
    dilated blocks) --> 1x1 head --> upsample --> joint angles (B, T, n_outputs)

``causal`` switches the whole model between

* **non-causal** (default, offline): zero-phase front end, centred "same"
  convolutions, window-level z-score, linear upsampling. Every output sample
  may use the whole window; this is what the competition allows.
* **causal** (online): left-only padding everywhere, expanding-window z-score,
  forward-only TCN, zero-order-hold upsampling from the last *completed*
  feature frame. The output at sample ``t`` depends on ``x[.. t]`` only
  (except the first ``pool1 * pool2`` samples of a window, which use frame 0).

Both variants share module names, so checkpoints load across them.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig
from .emg_frontend import EmgFrontEnd


class StTcnPose(BaseBrainModelConfig):
    """Config for :class:`StTcnPoseModel`.

    Front-end fields mirror :class:`EmgFrontEnd`; ``sfreq`` is the training
    sampling rate and fixes the FIR design. Model fields are listed in
    :attr:`StTcnPoseModel.MODEL_FIELDS`.
    """

    # front end (EmgFrontEnd); filtering and normalisation are off by default
    sfreq: float = 2000.0
    highpass: float | None = None
    lowpass: float | None = None
    notch: float | None = None
    notch_width: float = 6.0
    fir_taps: int = 2001
    normalize: bool = False
    clip: float | None = None
    # spatial 2-D CNN
    f1: int = 16
    k_e: int = 65
    spatial_mode: tp.Literal["conv2d", "none"] = "conv2d"
    spatial_layers: int = 2
    k_s: int = 9
    depth_mult: int = 2
    f2: int = 64
    k_sep: int = 16
    pool1: int = 5
    pool2: int = 8
    spatial_dropout: float = 0.2
    # TCN
    tcn_modules: int = 2
    tcn_blocks: int = 3
    tcn_filters: int = 64
    tcn_kernel: int = 4
    tcn_residual: bool = True
    tcn_dropout: float = 0.2
    # output head (see PoseHead)
    head_kind: tp.Literal["linear", "smooth", "velocity"] = "linear"
    head_smooth_k: int = 5
    head_position_steps: int = 12
    head_vel_scale: float = 0.02
    # causality of the whole model (front end, spatial block, TCN, head, upsampling)
    causal: bool = False

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "StTcnPose requires n_outputs"
        frontend = EmgFrontEnd(
            n_channels=n_spatial_locations, sfreq=self.sfreq, highpass=self.highpass,
            lowpass=self.lowpass, notch=self.notch, notch_width=self.notch_width,
            numtaps=self.fir_taps, normalize=self.normalize, clip=self.clip,
            causal=self.causal,
        )
        model_kwargs = {k: getattr(self, k) for k in StTcnPoseModel.MODEL_FIELDS}
        return StTcnPoseModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                              sfreq=self.sfreq, frontend=frontend, **model_kwargs)


# =============================================================================
# padding helpers
# =============================================================================
def _same_pad(k: int, causal: bool) -> tuple[int, int]:
    """(left, right) zero padding so a stride-1 conv of kernel ``k`` keeps length.

    Non-causal: centred (``k // 2`` each side for odd ``k``; for even ``k`` the
    extra sample goes left, matching PyTorch's symmetric ``padding=k//2`` on
    the kept samples). Causal: all ``k - 1`` on the left.
    """
    if causal:
        return k - 1, 0
    left = k // 2
    return left, k - 1 - left


# =============================================================================
# Spatial block: 2-D CNN over (channels x time)
# =============================================================================
class _RingConv2d(nn.Module):
    """Conv2d over (C, T) with circular padding along C and 'same' padding along T."""

    def __init__(self, ch: int, k_c: int, k_t: int, causal: bool):
        super().__init__()
        self.pad_c = k_c // 2
        self.pad_t = _same_pad(k_t, causal)
        self.conv = nn.Conv2d(ch, ch, (k_c, k_t), padding=0, bias=False)

    def forward(self, x):                                   # (B, F, C, T)
        x = F.pad(x, (0, 0, self.pad_c, self.pad_c), mode="circular")  # electrode ring
        x = F.pad(x, (*self.pad_t, 0, 0))                                # time (zeros)
        return self.conv(x)


class SpatialBlock(nn.Module):
    """EEGNet-style front end, optionally with 2-D spatial convs over the electrode ring."""

    def __init__(self, n_ch, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
                 f2, k_sep, pool1, pool2, dropout, causal):
        super().__init__()
        self.pad_e = _same_pad(k_e, causal)
        self.pad_sep = _same_pad(k_sep, causal)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, k_e), padding=0, bias=False),
            nn.BatchNorm2d(f1),
        )
        layers = []
        if spatial_mode == "conv2d":
            for _ in range(spatial_layers):
                layers += [_RingConv2d(f1, 3, k_s, causal), nn.BatchNorm2d(f1), nn.ELU()]
        self.spatial2d = nn.Sequential(*layers)
        # depthwise (C, 1): one spatial filter per feature map (x depth_mult), as in EEGNet
        self.collapse = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_mult, (n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_mult),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),   # bin j = samples [j*p, (j+1)*p): end-aligned, causal
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * depth_mult, f1 * depth_mult, (1, k_sep), padding=0,
                      groups=f1 * depth_mult, bias=False),
            nn.Conv2d(f1 * depth_mult, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        self.out_channels = f2
        self.downsample = pool1 * pool2

    def forward(self, x):                      # (B, C, T)
        x = F.pad(x.unsqueeze(1), (*self.pad_e, 0, 0))   # (B, 1, C, T + k_e - 1)
        x = self.temporal(x)                   # (B, F1, C, T)
        x = self.spatial2d(x)                  # (B, F1, C, T)
        x = self.collapse(x)                   # (B, F1*D, 1, T/p1)
        x = F.pad(x, (*self.pad_sep, 0, 0))
        x = self.separable(x)                  # (B, F2, 1, T/(p1*p2))
        return x.squeeze(2)                    # (B, F2, T')


# =============================================================================
# Temporal block: EEG-TCNet style TCN
# =============================================================================
class _DilatedConv1d(nn.Module):
    """Dilated conv with 'same' length output; causal (left pad) or centred."""

    def __init__(self, c_in, c_out, k, dilation, causal):
        super().__init__()
        total = (k - 1) * dilation
        self.pad = (total, 0) if causal else (total // 2, total - total // 2)
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


class TcnResidualBlock(nn.Module):
    """Two [dilated conv -> BN -> ELU -> dropout] layers, optional residual (EEG-TCNet Fig. 2)."""

    def __init__(self, c_in, c_out, k, dilation, dropout, residual, causal):
        super().__init__()
        self.net = nn.Sequential(
            _DilatedConv1d(c_in, c_out, k, dilation, causal), nn.BatchNorm1d(c_out), nn.ELU(), nn.Dropout(dropout),
            _DilatedConv1d(c_out, c_out, k, dilation, causal), nn.BatchNorm1d(c_out), nn.ELU(), nn.Dropout(dropout),
        )
        self.residual = residual
        if residual:
            # 1x1 projection only when the width changes (the "(Optional)" box in the paper)
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ELU()

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            y = self.act(y + self.skip(x))
        return y


class TcnModule(nn.Module):
    """``n_blocks`` residual blocks with dilations 1, 2, 4, ... (one EEG-TCNet TCN)."""

    def __init__(self, c_in, filters, k, n_blocks, dropout, residual, causal):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(TcnResidualBlock(c_in if i == 0 else filters, filters, k, 2 ** i,
                                           dropout, residual, causal))
        self.blocks = nn.Sequential(*blocks)
        # RFS = 1 + 2 (K_T - 1)(2^L - 1)   (paper, eq. 1)
        self.receptive_field = 1 + 2 * (k - 1) * (2 ** n_blocks - 1)

    def forward(self, x):
        return self.blocks(x)


# =============================================================================
# Output head
# =============================================================================
class PoseHead(nn.Module):
    """TCN features ``(B, F, T')`` -> joint angles ``(B, J, T')`` at the feature rate.

    kind="linear"   : 1x1 conv readout, one pose per frame, no continuity prior
                      (state-dict compatible with the original ``head`` conv).
    kind="smooth"   : the same readout followed by a learnable depthwise
                      ``smooth_k``-tap filter per joint, initialised to a moving
                      average (centred, or left-aligned when causal).
    kind="velocity" : the readout emits a position AND a velocity per joint
                      (2J channels). The pose is the position for the first
                      ``position_steps`` frames and from there on the previous
                      pose plus ``vel_scale`` x velocity -- VEMG2Pose's hybrid
                      parameterisation, without the LSTM. Integration is a
                      cumulative sum, so it is causal by construction.
    """

    def __init__(self, c_in: int, n_outputs: int, kind: str = "linear",
                 smooth_k: int = 5, position_steps: int = 12, vel_scale: float = 0.02,
                 causal: bool = False):
        super().__init__()
        if kind not in ("linear", "smooth", "velocity"):
            raise ValueError(f"head_kind must be linear | smooth | velocity, got {kind!r}")
        self.kind = kind
        self.n_outputs = n_outputs
        self.position_steps = position_steps
        self.vel_scale = vel_scale
        n_out = 2 * n_outputs if kind == "velocity" else n_outputs
        # named ``proj`` inside the head; StTcnPoseModel aliases it so that the
        # original key ``head.weight`` keeps loading for kind="linear"
        self.proj = nn.Conv1d(c_in, n_out, 1)
        if kind == "smooth":
            self.pad = _same_pad(smooth_k, causal)
            self.smooth = nn.Conv1d(n_outputs, n_outputs, smooth_k, groups=n_outputs, bias=False)
            with torch.no_grad():
                self.smooth.weight.fill_(1.0 / smooth_k)   # moving average at init

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.proj(z)
        if self.kind == "linear":
            return y
        if self.kind == "smooth":
            return self.smooth(F.pad(y, self.pad))
        pos, vel = y.chunk(2, dim=1)                       # (B, J, T') each
        k = min(self.position_steps, pos.shape[-1])
        if k == pos.shape[-1]:
            return pos
        integrated = pos[..., k - 1:k] + torch.cumsum(vel[..., k:] * self.vel_scale, dim=-1)
        return torch.cat([pos[..., :k], integrated], dim=-1)


# =============================================================================
# Full model
# =============================================================================
class StTcnPoseModel(nn.Module):
    # architecture kwargs the config forwards (keeps build() and __init__ in sync)
    MODEL_FIELDS = (
        "f1", "k_e", "spatial_mode", "spatial_layers", "k_s", "depth_mult",
        "f2", "k_sep", "pool1", "pool2", "spatial_dropout",
        "tcn_modules", "tcn_blocks", "tcn_filters", "tcn_kernel",
        "tcn_residual", "tcn_dropout",
        "head_kind", "head_smooth_k", "head_position_steps", "head_vel_scale",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # spatial
        f1=16, k_e=65, spatial_mode="conv2d", spatial_layers=2, k_s=9, depth_mult=2,
        f2=64, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # tcn
        tcn_modules=2, tcn_blocks=3, tcn_filters=64, tcn_kernel=4,
        tcn_residual=True, tcn_dropout=0.2,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        self.causal = causal
        # parameter-free preprocessing applied to every EMG window, train and test
        self.frontend = frontend if frontend is not None else EmgFrontEnd(
            n_in_channels, sfreq=sfreq, causal=causal)
        self.spatial = SpatialBlock(
            n_in_channels, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        modules, c_in = [], self.spatial.out_channels
        for _ in range(tcn_modules):
            modules.append(TcnModule(c_in, tcn_filters, tcn_kernel, tcn_blocks,
                                     tcn_dropout, tcn_residual, causal))
            c_in = tcn_filters
        self.tcn = nn.Sequential(*modules)
        self.head = nn.Conv1d(tcn_filters, n_outputs, 1)

        # bookkeeping for the docstring / logs
        self.feature_rate = sfreq / self.spatial.downsample
        self.tcn_receptive_field_s = (
            sum(m.receptive_field - 1 for m in modules) + 1
        ) / self.feature_rate

    def _upsample(self, y: torch.Tensor, n_samples: int) -> torch.Tensor:
        """(B, J, T') at the feature rate -> (B, J, T) at the input rate."""
        if not self.causal:
            return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        # Zero-order hold from the last completed frame. Frame j summarises
        # samples [j*P, (j+1)*P) and is complete at sample (j+1)*P - 1, so
        # sample t may use frame (t+1)//P - 1; samples before the first
        # completed frame fall back to frame 0.
        P = self.spatial.downsample
        t = torch.arange(n_samples, device=y.device)
        idx = ((t + 1) // P - 1).clamp_(0, y.shape[-1] - 1)
        return y[..., idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.frontend(x)            # (B, C, T)   filtered + normalised, no parameters
        z = self.spatial(x)             # (B, F2, T')  spatial features at ~50 Hz
        z = self.tcn(z)                 # (B, F_T, T') temporal context
        y = self.head(z)                # (B, n_outputs, T')
        y = self._upsample(y, n_samples)
        return y.transpose(1, 2)        # (B, T, n_outputs)