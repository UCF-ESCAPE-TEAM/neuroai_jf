"""Channel-attention CNN + bidirectional Mamba EMG-to-pose model (CaEmgMamba).

    EMG (B, C, T)
      --> WindowNorm            z-score of every electrode over the window (no parameters)
      --> ConvStem              3 strided Conv1d stages, 2 kHz -> 50 Hz; a channel-attention
                                gate (SE or ECA) after the stages listed in ``ca_stages``
      --> MambaEncoder          n_layers bidirectional Mamba blocks at the 50 Hz frame rate
      --> head                  "pose": one absolute joint-angle vector per frame
                                "vel" : position + velocity readouts, integrated and fused
      --> linear upsample       (B, T, n_outputs), time-major as NeuralBench expects

Deliberately *not* in this model (see tutorial/CA_EMGMAMBA_PROGRESS.md for the discussion):

* no EmgPreproc front end (no FIR band-pass / notch, no bad-channel rejection);
  the first convolution has to learn its own band-pass.
* no spatial block. The 16 electrodes enter the first ``Conv1d`` as ordinary input
  channels, exactly like the colour channels of an image. Nothing here is taken
  from EEGNet (no temporal-bank -> depthwise-spatial -> separable sequence) and
  this file imports nothing from the st_* models.

Channel attention
-----------------
Applied to the *feature* channels of the stem, after the non-linearity, where a
global average over time is a meaningful descriptor (raw EMG averages to ~0;
post-GELU features do not).  ``ca_kind``:

    "se"   Squeeze-and-Excitation (Hu et al. 2018): mean over time -> bottleneck
           MLP (C -> C/ca_reduction -> C) -> one gain per channel.
    "eca"  Efficient Channel Attention (Wang et al. 2020): mean over time ->
           1-D conv of kernel k across the channel axis, no bias. ``ca_eca_k: 0``
           picks k from the channel count with the paper's Eq. 12.
    "none" no gate.

Every gate is ``2 * sigmoid(.)`` with a zero-initialised last layer, so it is
exactly 1 at initialisation and the model starts as if the gate were absent.

Mamba encoder
-------------
Original Mamba block (Gu & Dao 2023): in_proj -> depthwise causal Conv1d -> SiLU
-> selective state-space scan -> gate by SiLU(z) -> out_proj. Parameter names and
shapes follow ``mamba_ssm.Mamba`` so the two are interchangeable. One layer runs
a block forward and a second block on the time-reversed frames and averages them
(``bidirectional: false`` keeps the forward one only).

``scan_impl``:
    "torch"  pure-PyTorch scan, one Python step per frame. Runs anywhere (this is
             what the Codabench submission uses) but autograd keeps every step,
             so training memory grows with batch * frames * d_inner * d_state.
             ``scan_checkpoint: true`` re-runs each scan during backward instead
             of storing it (same gradients; about 1/8 of the memory with 8 scans).
    "auto"   use ``mamba_ssm``'s CUDA selective-scan kernel when the package is
             importable and the tensors are on a GPU, otherwise fall back to
             "torch". Same parameters, same result up to float error.
"""

import math
import typing as tp

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import BaseBrainModelConfig

try:  # optional fast kernel; the model never requires it
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _fast_scan
except Exception:  # noqa: BLE001  (ImportError, or a broken CUDA build)
    _fast_scan = None


# =============================================================================
# Config
# =============================================================================
class CaEmgMamba(BaseBrainModelConfig):
    """Config for :class:`CaEmgMambaModel`; fields are listed in ``MODEL_FIELDS``."""

    sfreq: float = 2000.0
    # ---- per-window normalisation ----
    norm_eps: float = 1e-12
    # ---- 1-D CNN downsampler (one entry per stage) ----
    stem_channels: tuple[int, ...] = (64, 128, 192)
    stem_kernels: tuple[int, ...] = (31, 9, 5)        # samples at each stage's own input rate
    stem_strides: tuple[int, ...] = (5, 4, 2)         # 2000 -> 400 -> 100 -> 50 Hz
    stem_groups: int = 8                              # GroupNorm groups
    stem_dropout: float = 0.1
    # ---- channel attention ----
    ca_kind: tp.Literal["se", "eca", "none"] = "se"
    ca_stages: tuple[int, ...] = (1, 2, 3)            # 1-based stem stages that get a gate
    ca_reduction: int = 8                             # SE bottleneck = C / ca_reduction
    ca_eca_k: int = 0                                 # ECA kernel (odd); 0 = adaptive (Eq. 12)
    # ---- Mamba temporal encoder ----
    d_model: int = 192
    n_layers: int = 4
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    dt_rank: int = 0                                  # 0 = ceil(d_model / 16)
    bidirectional: bool = True
    enc_dropout: float = 0.1
    scan_impl: tp.Literal["torch", "auto"] = "auto"
    scan_checkpoint: bool = True                      # pure-PyTorch scan only, when training
    # ---- head ----
    head: tp.Literal["pose", "vel"] = "pose"
    position_steps: int = 12                          # vel: frames read as position first (240 ms)
    vel_scale: float = 0.02                           # vel: rad per frame per unit readout
    fuse_gain: float = 0.1                            # vel: pull towards the position readout

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "CaEmgMamba requires n_outputs"
        kwargs = {k: getattr(self, k) for k in CaEmgMambaModel.MODEL_FIELDS}
        return CaEmgMambaModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                               sfreq=self.sfreq, **kwargs)


# =============================================================================
# Per-window normalisation
# =============================================================================
class WindowNorm(nn.Module):
    """Z-score of every electrode over the time axis of its own window.

    This is ``LayerNorm`` over time without the affine weights (which would be
    one weight per sample and would fix the window length). The epsilon is added
    to the standard deviation, not the variance, and is far below any real signal
    (1e-12 vs a std of ~1e-6 for EMG in volts), so the result does not depend on
    the amplitude unit the server feeds (uV, V, or NeuralBench's scale). A dead
    electrode (std 0) comes out as all zeros.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.std(dim=-1, keepdim=True) + self.eps)


# =============================================================================
# Channel attention
# =============================================================================
def eca_kernel_size(n_ch: int, gamma: int = 2, b: int = 1) -> int:
    """ECA Eq. 12: k = |log2(C)/gamma + b/gamma|_odd."""
    k = int(abs((math.log2(n_ch) + b) / gamma))
    return k if k % 2 else k + 1


class ChannelAttention(nn.Module):
    """One gain in (0, 2) per feature channel and per window; identity at init."""

    def __init__(self, n_ch: int, kind: str = "se", reduction: int = 8, eca_k: int = 0):
        super().__init__()
        if kind not in ("se", "eca"):
            raise ValueError(f"ca_kind must be 'se', 'eca' or 'none', got {kind!r}")
        self.kind = kind
        if kind == "se":
            hidden = max(n_ch // reduction, 4)
            self.excite = nn.Sequential(nn.Linear(n_ch, hidden), nn.SiLU(), nn.Linear(hidden, n_ch))
            last = self.excite[-1]
        else:
            k = eca_k if eca_k else eca_kernel_size(n_ch)
            if k % 2 == 0:
                raise ValueError("ca_eca_k must be odd (or 0 for adaptive)")
            self.excite = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
            last = self.excite
        with torch.no_grad():
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()

    def gains(self, x: torch.Tensor) -> torch.Tensor:            # (B, C, T) -> (B, C)
        s = x.mean(dim=-1)                                       # squeeze
        s = self.excite(s) if self.kind == "se" else self.excite(s.unsqueeze(1)).squeeze(1)
        return 2.0 * torch.sigmoid(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gains(x).unsqueeze(-1)


# =============================================================================
# 1-D CNN downsampler
# =============================================================================
class StemStage(nn.Module):
    """Conv1d (strided) -> GroupNorm -> GELU -> Dropout -> [channel attention]."""

    def __init__(self, c_in: int, c_out: int, k: int, stride: int, groups: int, dropout: float,
                 ca_kind: str, ca_reduction: int, ca_eca_k: int):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("stem_kernels must be odd")
        self.conv = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
        self.norm = nn.GroupNorm(groups, c_out)
        self.drop = nn.Dropout(dropout)
        self.ca = (nn.Identity() if ca_kind == "none"
                   else ChannelAttention(c_out, ca_kind, ca_reduction, ca_eca_k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(self.drop(F.gelu(self.norm(self.conv(x)))))


class ConvStem(nn.Module):
    def __init__(self, n_in: int, channels, kernels, strides, groups, dropout,
                 ca_kind, ca_stages, ca_reduction, ca_eca_k):
        super().__init__()
        if not len(channels) == len(kernels) == len(strides):
            raise ValueError("stem_channels, stem_kernels and stem_strides must have the same length")
        bad = [s for s in ca_stages if not 1 <= s <= len(channels)]
        if bad:
            raise ValueError(f"ca_stages {bad} outside 1..{len(channels)}")
        dims = (n_in, *channels)
        self.stages = nn.ModuleList(
            StemStage(dims[i], dims[i + 1], kernels[i], strides[i], groups, dropout,
                      ca_kind if (i + 1) in ca_stages else "none", ca_reduction, ca_eca_k)
            for i in range(len(channels))
        )
        self.out_channels = channels[-1]
        self.downsample = math.prod(strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T) -> (B, F, T')
        for stage in self.stages:
            x = stage(x)
        return x


# =============================================================================
# Mamba
# =============================================================================
def selective_scan_torch(u, delta, A, B, C, D):
    """Reference selective scan.

    u, delta : (N, L, d_inner)      input and (already softplus-ed) step size
    A        : (d_inner, d_state)   negative decay rates
    B, C     : (N, L, d_state)      input-dependent input / output maps
    D        : (d_inner,)           skip gain
    returns  : (N, L, d_inner)

        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        y_t = <h_t, C_t> + D * u_t
    """
    N, L, d_inner = u.shape
    h = u.new_zeros(N, d_inner, A.shape[1])
    ys = []
    for t in range(L):
        dt = delta[:, t, :, None]                                 # (N, d_inner, 1)
        h = torch.exp(dt * A) * h + dt * B[:, t, None, :] * u[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))                 # (N, d_inner)
    return torch.stack(ys, dim=1) + u * D


class MambaBlock(nn.Module):
    """Mamba mixer (same parameter names and shapes as ``mamba_ssm.Mamba``)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int = 0, dt_min: float = 1e-3, dt_max: float = 0.1,
                 scan_impl: str = "auto", scan_checkpoint: bool = False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank else math.ceil(d_model / 16)
        self.scan_impl = scan_impl
        self.scan_checkpoint = scan_checkpoint

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        # step size: softplus(bias) log-uniform in [dt_min, dt_max] at init
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))     # inverse softplus
        # decay rates 1..d_state for every inner channel (stored as log, used as -exp)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (N, L, d_model)
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)                  # (N, L, d_inner) each
        u = self.conv1d(u.transpose(1, 2))[..., :L]              # causal depthwise conv
        u = F.silu(u).transpose(1, 2)                            # (N, L, d_inner)
        dt, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        if self.scan_impl == "auto" and _fast_scan is not None and x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                y = _fast_scan(
                    u.float().transpose(1, 2).contiguous(),                       # (N, d_inner, L)
                    (dt.float() @ self.dt_proj.weight.float().t()).transpose(1, 2).contiguous(),
                    A,
                    B.float().transpose(1, 2).contiguous(),                       # (N, d_state, L)
                    C.float().transpose(1, 2).contiguous(),
                    self.D.float(),
                    z=z.float().transpose(1, 2).contiguous(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                ).transpose(1, 2)                                                 # gated by SiLU(z)
            return self.out_proj(y.to(x.dtype))
        delta = F.softplus(self.dt_proj(dt))
        args = (u.float(), delta.float(), A, B.float(), C.float(), self.D.float())
        if self.scan_checkpoint and self.training and torch.is_grad_enabled():
            y = checkpoint(selective_scan_torch, *args, use_reentrant=False)
        else:
            y = selective_scan_torch(*args)
        return self.out_proj((y * F.silu(z.float())).to(x.dtype))


class MambaLayer(nn.Module):
    """Pre-norm residual layer; bidirectional = mean of a forward and a reversed block."""

    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, bidirectional, dropout, scan_impl,
                 scan_checkpoint=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        block = dict(scan_impl=scan_impl, scan_checkpoint=scan_checkpoint)
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand, dt_rank, **block)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand, dt_rank, **block) if bidirectional else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, L, d)
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = 0.5 * (y + self.bwd(h.flip(1)).flip(1))
        return x + self.drop(y)


class MambaEncoder(nn.Module):
    """Frame features ``(B, F, T')`` -> contextual features ``(B, d_model, T')``."""

    def __init__(self, in_ch, d_model, n_layers, d_state, d_conv, expand, dt_rank,
                 bidirectional, dropout, scan_impl, scan_checkpoint=False):
        super().__init__()
        self.proj = nn.Identity() if in_ch == d_model else nn.Conv1d(in_ch, d_model, 1)
        self.layers = nn.ModuleList(
            MambaLayer(d_model, d_state, d_conv, expand, dt_rank, bidirectional, dropout, scan_impl,
                       scan_checkpoint)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).transpose(1, 2)                         # (B, T', d)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).transpose(1, 2)                      # (B, d, T')


# =============================================================================
# Heads
# =============================================================================
class PoseHead(nn.Module):
    """One absolute joint-angle vector per frame (no continuity prior, no drift)."""

    def __init__(self, d_model: int, n_outputs: int):
        super().__init__()
        self.pos = nn.Conv1d(d_model, n_outputs, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, d, T') -> (B, J, T')
        return self.pos(z)


class VelocityHead(nn.Module):
    """Position + velocity readouts, integrated and fused (VEMG2Pose-style).

        pose_t = position_t                                          t <  position_steps
        pose_t = (1-g) * (pose_{t-1} + vel_scale * velocity_t) + g * position_t   otherwise

    g = ``fuse_gain``: 0 is pure integration (smooth, drifts), 1 is the pose
    head, values in between let the velocity shape fast motion while the
    position readout removes drift. The velocity readout starts at zero, so the
    trajectory begins as "hold the position".
    """

    def __init__(self, d_model: int, n_outputs: int, position_steps: int = 12,
                 vel_scale: float = 0.02, fuse_gain: float = 0.1):
        super().__init__()
        if not 0.0 <= fuse_gain <= 1.0:
            raise ValueError(f"fuse_gain must be in [0, 1], got {fuse_gain}")
        self.position_steps, self.vel_scale, self.fuse_gain = position_steps, vel_scale, fuse_gain
        self.pos = nn.Conv1d(d_model, n_outputs, 1)
        self.vel = nn.Conv1d(d_model, n_outputs, 1)
        with torch.no_grad():
            self.vel.weight.zero_()
            self.vel.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, d, T') -> (B, J, T')
        pos, vel = self.pos(z), self.vel(z)
        T = pos.shape[-1]
        k = min(self.position_steps, T)
        if k == T:
            return pos
        step = vel[..., k:] * self.vel_scale
        if self.fuse_gain == 0.0:
            return torch.cat([pos[..., :k], pos[..., k - 1:k] + torch.cumsum(step, dim=-1)], dim=-1)
        g = self.fuse_gain
        prev, out = pos[..., k - 1], []
        for t in range(T - k):
            prev = (1.0 - g) * (prev + step[..., t]) + g * pos[..., k + t]
            out.append(prev)
        return torch.cat([pos[..., :k], torch.stack(out, dim=-1)], dim=-1)


# =============================================================================
# Full model
# =============================================================================
class CaEmgMambaModel(nn.Module):
    MODEL_FIELDS = (
        "norm_eps",
        "stem_channels", "stem_kernels", "stem_strides", "stem_groups", "stem_dropout",
        "ca_kind", "ca_stages", "ca_reduction", "ca_eca_k",
        "d_model", "n_layers", "d_state", "expand", "d_conv", "dt_rank",
        "bidirectional", "enc_dropout", "scan_impl", "scan_checkpoint",
        "head", "position_steps", "vel_scale", "fuse_gain",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        norm_eps: float = 1e-12,
        stem_channels=(64, 128, 192), stem_kernels=(31, 9, 5), stem_strides=(5, 4, 2),
        stem_groups: int = 8, stem_dropout: float = 0.1,
        ca_kind: str = "se", ca_stages=(1, 2, 3), ca_reduction: int = 8, ca_eca_k: int = 0,
        d_model: int = 192, n_layers: int = 4, d_state: int = 16, expand: int = 2,
        d_conv: int = 4, dt_rank: int = 0, bidirectional: bool = True,
        enc_dropout: float = 0.1, scan_impl: str = "auto", scan_checkpoint: bool = True,
        head: str = "pose", position_steps: int = 12, vel_scale: float = 0.02,
        fuse_gain: float = 0.1,
    ):
        super().__init__()
        if head not in ("pose", "vel"):
            raise ValueError(f"head must be 'pose' or 'vel', got {head!r}")
        self.norm = WindowNorm(norm_eps)
        self.stem = ConvStem(n_in_channels, tuple(stem_channels), tuple(stem_kernels),
                             tuple(stem_strides), stem_groups, stem_dropout,
                             ca_kind, tuple(ca_stages), ca_reduction, ca_eca_k)
        self.encoder = MambaEncoder(self.stem.out_channels, d_model, n_layers, d_state, d_conv,
                                    expand, dt_rank, bidirectional, enc_dropout, scan_impl,
                                    scan_checkpoint)
        self.head = (PoseHead(d_model, n_outputs) if head == "pose"
                     else VelocityHead(d_model, n_outputs, position_steps, vel_scale, fuse_gain))
        self.feature_rate = sfreq / self.stem.downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        z = self.stem(self.norm(x))        # (B, F, T')  features at ~50 Hz
        z = self.encoder(z)                # (B, d, T')  contextual features
        y = self.head(z)                   # (B, J, T')  one pose per frame
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)           # (B, T, J)
