"""Spatial-CNN + transformer EMG-to-pose model with an absolute-angle head (StPoseTransformer).

    EMG (B, C, T) --> EmgPreproc (no params) --> [ChannelGate] --> SpatialBlock
    (EEGNet-style 2-D CNN over the electrode ring, optionally with a multi-scale
    temporal filter bank; pools to ~50 Hz) --> SequenceEncoder (pre-norm
    Conformer / transformer blocks with rotary positions) --> 1x1 readout
    (one joint-angle vector per frame) --> upsample --> (B, T, n_outputs)

This file also holds every block that :mod:`st_vel_transformer` shares with it
(``ChannelGate``, ``MsSpatialBlock``, ``SequenceEncoder``, ``upsample_frames``).
The two models differ only after the encoder: this one reads the *angle* out of
every frame independently (no continuity prior, no drift); the velocity model
reads a position and a velocity and integrates. See
``tutorial/transformer_pose_models.md`` for the design rationale.

Why the transformer sits *after* a convolutional subsampler
-----------------------------------------------------------
Attention over raw 2 kHz samples is pointless (a 5-s window is 10 000 tokens,
almost all redundant) and quadratic in length. The spatial block turns the
window into ~250 tokens at 50 Hz, each already a 128-d summary of 20 ms of all
16 electrodes, so the encoder models *muscle-activation dynamics at the tempo of
hand motion* rather than the EMG waveform. This is the conv-subsampling pattern
of speech Conformers, and it is what makes the encoder cheap (250^2 attention).

``causal`` keeps the same meaning as in :class:`StTcnPose`: with ``True`` every
convolution is left-padded, the scaling is expanding-window, attention is
masked to the past (and to ``attn_window`` frames), and the upsampling is a
zero-order hold. Checkpoints load across both settings.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig
from .emg_frontend import EmgPreproc, frontend_from_config
from .st_tcn_pose import PosePost, SpatialBlock, _same_pad


# =============================================================================
# Config
# =============================================================================
class StPoseTransformer(BaseBrainModelConfig):
    """Config for :class:`StPoseTransformerModel`.

    Front-end fields mirror :class:`EmgPreproc` (identical to StTcnPose).
    Model fields are listed in :attr:`StPoseTransformerModel.MODEL_FIELDS`.
    """

    # ---- front end (EmgPreproc): FIR band-pass + notch, per-window scaling ----
    sfreq: float = 2000.0
    l_freq: float | None = 20.0
    h_freq: float | None = 450.0
    notch_freqs: tuple[float, ...] | None = (60.0, 120.0, 180.0)
    notch_width: float = 6.0
    bp_taps: int | None = None
    notch_taps: int | None = None
    norm: tp.Literal["maxabs", "minmax", "robust", "zscore", "none"] = "zscore"
    scope: tp.Literal["channel", "window", "batch"] = "channel"
    signed: bool = True
    robust_q: float = 0.005
    eps: float = 1e-6
    clip: float | None = None
    reject_channels: bool = False
    reject_rms_nstd: float = 3.0
    reject_mains: float | None = 60.0
    reject_mains_nstd: float = 3.0
    reject_mains_bw: float = 2.0
    reject_segment: int = 400
    reject_max_frac: float = 0.5
    reject_method: tp.Literal["std", "mad"] = "std"
    # ---- channel attention (SE-style gate over the 16 electrodes) ----
    channel_attention: bool = False
    ca_hidden: int = 32
    # ---- spatial 2-D CNN (as StTcnPose) + optional multi-scale temporal bank ----
    f1: int = 32
    k_e: int = 65
    ms_kernels: tuple[int, ...] | None = None   # e.g. (15, 31, 63, 127); None -> single k_e bank
    spatial_mode: tp.Literal["conv2d", "none"] = "conv2d"
    spatial_layers: int = 2
    k_s: int = 9
    depth_mult: int = 2
    f2: int = 128
    k_sep: int = 16
    pool1: int = 5
    pool2: int = 8
    spatial_dropout: float = 0.2
    # ---- sequence encoder (Conformer / transformer at the 50 Hz frame rate) ----
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    enc_dropout: float = 0.1
    attn_dropout: float = 0.1
    block_style: tp.Literal["conformer", "transformer"] = "conformer"
    conv_kernel: int = 15                 # conformer conv module; frames (15 = 300 ms)
    pos_enc: tp.Literal["rope", "sinusoid", "none"] = "rope"
    attn_window: int | None = None        # frames of attention context per side; None = full window
    # ---- output head ----
    head_kind: tp.Literal["linear", "smooth"] = "linear"
    head_smooth_k: int = 5
    # ---- causality of the whole model ----
    causal: bool = False

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "StPoseTransformer requires n_outputs"
        frontend = frontend_from_config(n_spatial_locations, self.model_dump())
        model_kwargs = {k: getattr(self, k) for k in StPoseTransformerModel.MODEL_FIELDS}
        return StPoseTransformerModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                                      sfreq=self.sfreq, frontend=frontend, **model_kwargs)


# =============================================================================
# Channel attention: a gate over the electrodes from per-window statistics
# =============================================================================
class ChannelGate(nn.Module):
    """Squeeze-and-excitation over the C input electrodes.

    Band placement and rotation differ between users, so the informative
    electrodes differ from window to window. The gate reads two scale-free
    statistics per channel (log RMS and log mean |x|, both of the already
    normalised signal), passes them through a tiny MLP and multiplies every
    channel by ``2 * sigmoid(.)``. The last layer starts at zero, so the gate is
    exactly 1 (identity) at initialisation and the model begins as if the gate
    were absent.

    Non-causal: statistics over the whole window -> one gate per window.
    Causal: expanding-window statistics -> one gate per sample; the gate at
    sample ``t`` uses ``x[.. t]`` only.
    """

    def __init__(self, n_ch: int, hidden: int = 32, causal: bool = False, eps: float = 1e-6):
        super().__init__()
        self.causal = causal
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(2 * n_ch, hidden), nn.SiLU(), nn.Linear(hidden, n_ch))
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        if not self.causal:
            rms = x.pow(2).mean(-1).add(self.eps).sqrt()          # (B, C)
            mav = x.abs().mean(-1).add(self.eps)
            stats = torch.cat([rms.log(), mav.log()], dim=-1)    # (B, 2C)
            gate = 2.0 * torch.sigmoid(self.mlp(stats))          # (B, C)
            return x * gate.unsqueeze(-1)
        n = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        rms = (x.pow(2).cumsum(-1) / n).add(self.eps).sqrt()     # (B, C, T)
        mav = (x.abs().cumsum(-1) / n).add(self.eps)
        stats = torch.cat([rms.log(), mav.log()], dim=1).transpose(1, 2)   # (B, T, 2C)
        gate = 2.0 * torch.sigmoid(self.mlp(stats)).transpose(1, 2)        # (B, C, T)
        return x * gate


# =============================================================================
# Multi-scale temporal filter bank inside the spatial block
# =============================================================================
class _MultiScaleTemporal(nn.Module):
    """Parallel per-channel temporal convs with different kernel lengths.

    The ``f1`` filters are split evenly over the kernel sizes; each branch pads
    its own input ("same" length, left-only when causal) and the outputs are
    concatenated along the filter axis, followed by one BatchNorm. This is the
    MSST idea (Lin et al. 2026, Measurement 257:118953) with kernel lengths
    chosen for 2 kHz: 15/31/63/127 samples = 7.5/15/31/63 ms. A single kernel
    (``k_e = 65``) sees one window length; the bank sees short bursts and slow
    envelopes at once.
    """

    def __init__(self, f1: int, kernels: tp.Sequence[int], causal: bool):
        super().__init__()
        if f1 % len(kernels):
            raise ValueError(f"f1={f1} must be divisible by the number of ms_kernels ({len(kernels)})")
        per = f1 // len(kernels)
        self.pads = [_same_pad(k, causal) for k in kernels]
        self.branches = nn.ModuleList(nn.Conv2d(1, per, (1, k), bias=False) for k in kernels)
        self.bn = nn.BatchNorm2d(f1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, 1, C, T)
        outs = [conv(F.pad(x, (*pad, 0, 0))) for conv, pad in zip(self.branches, self.pads)]
        return self.bn(torch.cat(outs, dim=1))                   # (B, f1, C, T)


class MsSpatialBlock(SpatialBlock):
    """:class:`SpatialBlock` whose first temporal conv is a multi-scale bank."""

    def __init__(self, n_ch, f1, kernels, spatial_mode, spatial_layers, k_s, depth_mult,
                 f2, k_sep, pool1, pool2, dropout, causal):
        super().__init__(n_ch, f1, max(kernels), spatial_mode, spatial_layers, k_s, depth_mult,
                         f2, k_sep, pool1, pool2, dropout, causal)
        self.temporal = _MultiScaleTemporal(f1, kernels, causal)
        self.pad_e = (0, 0)                      # each branch pads itself


def make_spatial_block(n_ch, f1, k_e, ms_kernels, spatial_mode, spatial_layers, k_s, depth_mult,
                       f2, k_sep, pool1, pool2, dropout, causal) -> SpatialBlock:
    if ms_kernels:
        return MsSpatialBlock(n_ch, f1, tuple(ms_kernels), spatial_mode, spatial_layers, k_s,
                              depth_mult, f2, k_sep, pool1, pool2, dropout, causal)
    return SpatialBlock(n_ch, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
                        f2, k_sep, pool1, pool2, dropout, causal)


# =============================================================================
# Sequence encoder: pre-norm Conformer / transformer blocks at the frame rate
# =============================================================================
def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on the last axis of ``x`` (B, H, T, D)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


def _rotary_tables(T: int, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None]   # (T, dim/2)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _sinusoid(T: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pe = torch.zeros(T, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe.to(dtype)


def attention_mask(T: int, causal: bool, window: int | None, device) -> torch.Tensor | None:
    """Boolean (T, T) mask, True = may attend; None when unrestricted."""
    if not causal and window is None:
        return None
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    allowed = torch.ones(T, T, dtype=torch.bool, device=device)
    if causal:
        allowed &= j <= i
    if window is not None:
        allowed &= (i - j).abs() <= window
    return allowed


class _SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float, rope: bool):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model={d} must be divisible by n_heads={n_heads}")
        self.h = n_heads
        self.hd = d // n_heads
        if rope and self.hd % 2:
            raise ValueError("rotary positions need an even head dimension")
        self.rope = rope
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:   # (B, T, d)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)   # 3 x (B, H, T, hd)
        if self.rope:
            cos, sin = _rotary_tables(T, self.hd, x.device, q.dtype)
            q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class _FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mult * d), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(mult * d, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class _ConvModule(nn.Module):
    """Conformer convolution module: LN -> 1x1 (GLU) -> depthwise k -> BN -> SiLU -> 1x1."""

    def __init__(self, d: int, k: int, dropout: float, causal: bool):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.pad = _same_pad(k, causal)
        self.dw = nn.Conv1d(d, d, k, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                        # (B, T, d)
        y = self.ln(x).transpose(1, 2)                           # (B, d, T)
        y = F.glu(self.pw1(y), dim=1)
        y = self.dw(F.pad(y, self.pad))
        y = self.pw2(F.silu(self.bn(y)))
        return self.drop(y).transpose(1, 2)


class EncoderBlock(nn.Module):
    """Pre-norm block.

    conformer  : x + 1/2 FFN -> x + MHSA -> x + Conv -> x + 1/2 FFN -> LN  (Gulati et al. 2020)
    transformer: x + MHSA -> x + FFN                               (Vaswani et al. 2017, pre-norm)
    """

    def __init__(self, d, n_heads, ff_mult, dropout, attn_dropout, style, conv_kernel, rope, causal):
        super().__init__()
        self.style = style
        if style == "conformer":
            self.ff1 = _FeedForward(d, ff_mult, dropout)
            self.conv = _ConvModule(d, conv_kernel, dropout, causal)
            self.ln_out = nn.LayerNorm(d)
        self.ln_attn = nn.LayerNorm(d)
        self.attn = _SelfAttention(d, n_heads, attn_dropout, rope)
        self.attn_drop = nn.Dropout(dropout)
        self.ff2 = _FeedForward(d, ff_mult, dropout)

    def forward(self, x, mask):
        if self.style == "conformer":
            x = x + 0.5 * self.ff1(x)
        x = x + self.attn_drop(self.attn(self.ln_attn(x), mask))
        if self.style == "conformer":
            x = x + self.conv(x)
            x = x + 0.5 * self.ff2(x)
            return self.ln_out(x)
        return x + self.ff2(x)


class SequenceEncoder(nn.Module):
    """Frame features ``(B, F, T')`` -> contextual features ``(B, d_model, T')``."""

    def __init__(self, in_ch, d_model, n_heads, n_layers, ff_mult, dropout, attn_dropout,
                 block_style, conv_kernel, pos_enc, attn_window, causal):
        super().__init__()
        self.pos_enc = pos_enc
        self.attn_window = attn_window
        self.causal = causal
        self.d_model = d_model
        self.proj = nn.Conv1d(in_ch, d_model, 1)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, ff_mult, dropout, attn_dropout, block_style,
                         conv_kernel, pos_enc == "rope", causal)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, F, T')
        x = self.proj(z).transpose(1, 2)                         # (B, T', d)
        if self.pos_enc == "sinusoid":
            x = x + _sinusoid(x.shape[1], self.d_model, x.device, x.dtype)
        x = self.in_drop(x)
        mask = attention_mask(x.shape[1], self.causal, self.attn_window, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x).transpose(1, 2)                      # (B, d, T')


# =============================================================================
# Shared helpers
# =============================================================================
def upsample_frames(y: torch.Tensor, n_samples: int, causal: bool, P: int) -> torch.Tensor:
    """(B, J, T') at the frame rate -> (B, J, T) at the input rate (as StTcnPose)."""
    if not causal:
        return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
    t = torch.arange(n_samples, device=y.device)
    idx = ((t + 1) // P - 1).clamp_(0, y.shape[-1] - 1)         # last completed frame
    return y[..., idx]


SPATIAL_FIELDS = (
    "f1", "k_e", "ms_kernels", "spatial_mode", "spatial_layers", "k_s", "depth_mult",
    "f2", "k_sep", "pool1", "pool2", "spatial_dropout",
)
ENCODER_FIELDS = (
    "d_model", "n_heads", "n_layers", "ff_mult", "enc_dropout", "attn_dropout",
    "block_style", "conv_kernel", "pos_enc", "attn_window",
)


# =============================================================================
# Full model
# =============================================================================
class StPoseTransformerModel(nn.Module):
    MODEL_FIELDS = (
        "channel_attention", "ca_hidden",
        *SPATIAL_FIELDS, *ENCODER_FIELDS,
        "head_kind", "head_smooth_k",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # channel attention
        channel_attention=False, ca_hidden=32,
        # spatial
        f1=32, k_e=65, ms_kernels=None, spatial_mode="conv2d", spatial_layers=2, k_s=9,
        depth_mult=2, f2=128, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # encoder
        d_model=128, n_heads=4, n_layers=4, ff_mult=4, enc_dropout=0.1, attn_dropout=0.1,
        block_style="conformer", conv_kernel=15, pos_enc="rope", attn_window=None,
        # head
        head_kind="linear", head_smooth_k=5,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        self.causal = causal
        self.frontend = frontend if frontend is not None else EmgPreproc(
            n_in_channels, sfreq=sfreq, norm="zscore", causal=causal)
        self.gate = ChannelGate(n_in_channels, ca_hidden, causal) if channel_attention else nn.Identity()
        self.spatial = make_spatial_block(
            n_in_channels, f1, k_e, ms_kernels, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        self.encoder = SequenceEncoder(
            self.spatial.out_channels, d_model, n_heads, n_layers, ff_mult, enc_dropout,
            attn_dropout, block_style, conv_kernel, pos_enc, attn_window, causal,
        )
        self.head = nn.Conv1d(d_model, n_outputs, 1)
        self.head_post = PosePost(n_outputs, head_kind, head_smooth_k, causal=causal)
        self.feature_rate = sfreq / self.spatial.downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.gate(self.frontend(x))    # (B, C, T)   filtered, scaled, gated
        z = self.spatial(x)                # (B, f2, T') spatial features at ~50 Hz
        z = self.encoder(z)                # (B, d, T')  contextual features
        y = self.head_post(self.head(z))   # (B, J, T')  one angle vector per frame
        y = upsample_frames(y, n_samples, self.causal, self.spatial.downsample)
        return y.transpose(1, 2)           # (B, T, J)
