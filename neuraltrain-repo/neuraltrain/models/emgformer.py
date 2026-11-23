"""EMGFormer (Xi et al. 2026, "EgoEMG", Sec. 4.2, Fig. 3, App. B.4 / D.2, Table S12).

    EMG (B, C, T)
      --> ZeroPhaseFilter       OPTIONAL (``filter: true|false``): 20 Hz high-pass, 450 Hz
                                low-pass, notches at the mains harmonics, zero phase
      --> WindowNorm            per-channel z-score over the window (``norm``)
      --> TdsFeaturizer         4 TDS blocks, 2 kHz -> 40 Hz, 256 features per token
      --> Transformer           L post-norm layers ("Add & Norm"), self-attention with RoPE
      --> head                  Linear -> GELU -> Linear, one pose per token
      --> linear upsample       (B, T, n_outputs), time-major as NeuralBench expects

Sizes (Table S12) are set in the YAMLs:

    variant   n_layers  d_model  n_heads  d_ffn   paper params (8 ch, 22 angles)
    S         3         256      4        512     3.5M
    M         6         256      8        1024    6.6M
    L         8         384      12       1536    16.3M

What the paper specifies, and what is a choice made here
--------------------------------------------------------
Specified: the filter band and notch frequencies and that it is zero-phase; the four
TDS blocks (kernels, strides, widths, squeeze-and-excitation after the TDS blocks);
RoPE; L / d / heads / d_ffn; the Linear-GELU-Linear head; AdamW 5e-4 -> 5e-6 cosine,
weight decay 1e-2, gradient clipping 1.0.

Not specified (defaults here, all exposed as config fields):
* filter family / order / notch width -> Butterworth order 4, 2 Hz wide notches
  (``filter_order``, ``notch_width``). The paper filters whole sessions offline; here each
  window is filtered on its own after an odd-reflection pad of ``filter_pad`` samples.
* mains frequency: EgoEMG is a 50 Hz recording, emg2pose a 60 Hz one -> ``notch_freq``.
* padding, normalisation and activation inside the TDS blocks -> centred "same"
  padding, ReLU, and layer normalisation over features and time as in the original
  TDS paper (``tds_norm: time``). Normalising each time step on its own
  (``tds_norm: step``) removes the EMG envelope and the model then predicts a constant.
* 256 -> 384 bridge for the L size -> one Linear (identity when d_model == 256).
* dropout, FFN activation, head hidden width -> 0.1, GELU, d_model.
* input normalisation: the paper uses training-set statistics; the Codabench server
  feeds another amplitude scale, so a per-window z-score is used instead.
The paper's training augmentations and its causal variants are not implemented.
"""

import math
import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig


# =============================================================================
# Config
# =============================================================================
class EmgFormer(BaseBrainModelConfig):
    """Config for :class:`EmgFormerModel`; fields are listed in ``MODEL_FIELDS``."""

    sfreq: float = 2000.0
    # ---- optional zero-phase filter (App. B.4) ----
    filter: bool = True
    l_freq: float | None = 20.0                       # high-pass corner (Hz); None = off
    h_freq: float | None = 450.0                      # low-pass corner (Hz); None = off
    notch_freq: float | None = 60.0                   # mains base (paper: 50); None = no notches
    notch_max_freq: float = 450.0                     # notch every harmonic up to here
    notch_width: float = 2.0                          # -3 dB width of every notch (Hz), per pass
    filter_order: int = 4                             # Butterworth order of each pass
    filter_pad: int = 1000                            # odd-reflection pad per side (samples)
    # ---- per-window normalisation ----
    norm: tp.Literal["zscore", "none"] = "zscore"
    norm_eps: float = 1e-12
    # ---- TDS featurizer (App. D.2) ----
    tds_features: int = 256                           # = tds_channels * tds_width
    tds_channels: int = 8
    conv_kernels: tuple[int, ...] = (11, 5, 9, 3)     # strided input conv of each block
    conv_strides: tuple[int, ...] = (5, 2, 5, 1)      # 2000 -> 400 -> 200 -> 40 -> 40 Hz
    tds_kernels: tuple[int, ...] = (0, 0, 5, 3)       # TDS conv kernel per block; 0 = no TDS part
    se_reduction: int = 16
    tds_norm: tp.Literal["time", "batch", "step"] = "time"   # see ChannelLayerNorm
    # ---- Transformer (Table S12) ----
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    d_ffn: int = 512
    dropout: float = 0.1
    rope_base: float = 10000.0
    # ---- head ----
    head_hidden: int = 0                              # 0 = d_model

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "EmgFormer requires n_outputs"
        kwargs = {k: getattr(self, k) for k in EmgFormerModel.MODEL_FIELDS}
        return EmgFormerModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                              sfreq=self.sfreq, **kwargs)


# =============================================================================
# Optional zero-phase filter (no parameters, no buffers in the state dict)
# =============================================================================
def zero_phase_gain(freqs: torch.Tensor, sfreq: float, l_freq: float | None,
                    h_freq: float | None, notch_freqs: tp.Sequence[float], notch_width: float,
                    order: int) -> torch.Tensor:
    """Real gain per frequency of a forward-backward (``filtfilt``) cascade.

    Running a filter forward and backward gives zero phase and the squared
    magnitude |H(f)|^2. For digital (bilinear) Butterworth passes and the standard
    second-order notch (``scipy.signal.iirnotch``) that magnitude has a closed form,
    so no filter design routine is needed.
    """
    w = math.pi * freqs.double() / sfreq                          # half the digital frequency
    tan_w = torch.tan(w.clamp(max=math.pi / 2 - 1e-9))
    gain = torch.ones_like(tan_w)
    if l_freq is not None:                                        # Butterworth high-pass
        ratio = math.tan(math.pi * l_freq / sfreq) / tan_w.clamp(min=1e-30)
        gain = gain / (1.0 + ratio ** (2 * order))
    if h_freq is not None:                                        # Butterworth low-pass
        ratio = tan_w / math.tan(math.pi * h_freq / sfreq)
        gain = gain / (1.0 + ratio ** (2 * order))
    for f0 in notch_freqs:                                        # second-order notches
        w0 = 2 * math.pi * f0 / sfreq
        beta = math.tan(math.pi * notch_width / sfreq)            # tan(bandwidth / 2)
        d = torch.cos(2 * w) - math.cos(w0)
        gain = gain * d ** 2 / (d ** 2 + (beta * torch.sin(2 * w)) ** 2)
    return gain.float()


class ZeroPhaseFilter(nn.Module):
    """Band-pass + mains notches applied in the frequency domain, per window.

    The window is extended by odd reflection (as ``scipy.signal.filtfilt`` does),
    multiplied by the zero-phase gain in the FFT domain, and cropped back.
    """

    def __init__(self, sfreq: float = 2000.0, l_freq: float | None = 20.0,
                 h_freq: float | None = 450.0, notch_freq: float | None = 60.0,
                 notch_max_freq: float = 450.0, notch_width: float = 2.0, order: int = 4,
                 pad: int = 1000):
        super().__init__()
        nyq = sfreq / 2.0
        for name, f in (("l_freq", l_freq), ("h_freq", h_freq)):
            if f is not None and not 0.0 < f < nyq:
                raise ValueError(f"{name} must be in (0, {nyq}), got {f}")
        self.sfreq, self.l_freq, self.h_freq = float(sfreq), l_freq, h_freq
        self.notch_width, self.order, self.pad = notch_width, order, pad
        self.notch_freqs: tuple[float, ...] = ()
        if notch_freq is not None:
            n = int(math.floor(min(notch_max_freq, nyq - 1e-6) / notch_freq + 1e-9))
            self.notch_freqs = tuple(notch_freq * k for k in range(1, n + 1))
        self._gain: dict[tuple[int, str], torch.Tensor] = {}      # plain cache, not a buffer

    def gain(self, n: int, device: torch.device) -> torch.Tensor:
        key = (n, str(device))
        if key not in self._gain:
            freqs = torch.fft.rfftfreq(n, d=1.0 / self.sfreq)
            self._gain[key] = zero_phase_gain(freqs, self.sfreq, self.l_freq, self.h_freq,
                                              self.notch_freqs, self.notch_width, self.order).to(device)
        return self._gain[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, T) -> (B, C, T)
        T = x.shape[-1]
        p = min(self.pad, T - 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            if p:
                left = 2 * xf[..., :1] - xf[..., 1:p + 1].flip(-1)
                right = 2 * xf[..., -1:] - xf[..., -p - 1:-1].flip(-1)
                xf = torch.cat([left, xf, right], dim=-1)
            n = xf.shape[-1]
            y = torch.fft.irfft(torch.fft.rfft(xf) * self.gain(n, xf.device), n=n)
        return y[..., p:p + T].to(x.dtype)

    def extra_repr(self) -> str:
        return (f"band=({self.l_freq}, {self.h_freq}) Hz, order={self.order}, "
                f"notches={self.notch_freqs}, width={self.notch_width} Hz, pad={self.pad}")


class WindowNorm(nn.Module):
    """Z-score of every electrode over the time axis of its own window."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, T)
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.std(dim=-1, keepdim=True) + self.eps)


# =============================================================================
# TDS featurizer
# =============================================================================
class ChannelLayerNorm(nn.Module):
    """Normalisation of a ``(B, F, T)`` feature map; ``kind`` sets what is pooled.

    "time"   one mean / std per window, over features AND time (Hannun et al. 2019:
             "over all dimensions for a given example including time"). Keeps how the
             signal strength changes along the window, which is where EMG carries its
             information. Default.
    "batch"  BatchNorm1d: statistics per feature over batch and time. Also keeps the
             amplitude; adds running statistics to the checkpoint.
    "step"   LayerNorm over the features of each time step on its own. This rescales
             every time step to unit variance and so removes the EMG envelope: with it
             the model collapsed to a constant pose (val R2 ~ 0) on emg2pose. Kept only
             to reproduce that run.
    """

    def __init__(self, n_features: int, kind: str = "time"):
        super().__init__()
        if kind not in ("time", "batch", "step"):
            raise ValueError(f"tds_norm must be 'time', 'batch' or 'step', got {kind!r}")
        self.kind = kind
        self.norm = {"time": lambda: nn.GroupNorm(1, n_features),
                     "batch": lambda: nn.BatchNorm1d(n_features),
                     "step": lambda: nn.LayerNorm(n_features)}[kind]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "step":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class TdsConvBlock(nn.Module):
    """Time-depth separable convolution (Hannun et al. 2019).

    The F = channels * width features are viewed as ``channels`` groups of
    ``width``; a ``(1, k)`` 2-D convolution mixes the groups along time only.
    """

    def __init__(self, channels: int, width: int, kernel: int, norm: str = "time"):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("tds_kernels must be odd")
        self.channels, self.width = channels, width
        self.conv = nn.Conv2d(channels, channels, (1, kernel), padding=(0, kernel // 2))
        self.norm = ChannelLayerNorm(channels * width, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, F, T)
        B, n_feat, T = x.shape
        y = self.conv(x.reshape(B, self.channels, self.width, T))
        return self.norm(x + F.relu(y).reshape(B, n_feat, T))


class TdsFullyConnectedBlock(nn.Module):
    """Channel-wise (per time step) two-layer MLP with a residual connection."""

    def __init__(self, n_features: int, norm: str = "time"):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU(),
                                nn.Linear(n_features, n_features))
        self.norm = ChannelLayerNorm(n_features, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, F, T)
        y = self.fc(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation (Hu et al. 2018): one gain per feature and window."""

    def __init__(self, n_features: int, reduction: int = 16):
        super().__init__()
        hidden = max(n_features // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                nn.Linear(hidden, n_features), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, F, T)
        return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)


class TdsStage(nn.Module):
    """Strided Conv1d -> LayerNorm -> ReLU, then (``tds_kernel`` > 0) TDS conv -> FC -> SE."""

    def __init__(self, c_in: int, n_features: int, kernel: int, stride: int,
                 tds_channels: int, tds_kernel: int, se_reduction: int, norm: str = "time"):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("conv_kernels must be odd")
        self.conv = nn.Conv1d(c_in, n_features, kernel, stride=stride, padding=kernel // 2)
        self.norm = ChannelLayerNorm(n_features, norm)
        if tds_kernel:
            self.tds = nn.Sequential(
                TdsConvBlock(tds_channels, n_features // tds_channels, tds_kernel, norm),
                TdsFullyConnectedBlock(n_features, norm),
                SqueezeExcite(n_features, se_reduction),
            )
        else:
            self.tds = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tds(F.relu(self.norm(self.conv(x))))


class TdsFeaturizer(nn.Module):
    def __init__(self, n_in: int, n_features: int, tds_channels: int, conv_kernels,
                 conv_strides, tds_kernels, se_reduction: int, norm: str = "time"):
        super().__init__()
        if not len(conv_kernels) == len(conv_strides) == len(tds_kernels):
            raise ValueError("conv_kernels, conv_strides and tds_kernels must have the same length")
        if n_features % tds_channels:
            raise ValueError("tds_features must be a multiple of tds_channels")
        self.stages = nn.ModuleList(
            TdsStage(n_in if i == 0 else n_features, n_features, conv_kernels[i],
                     conv_strides[i], tds_channels, tds_kernels[i], se_reduction, norm)
            for i in range(len(conv_kernels))
        )
        self.downsample = math.prod(conv_strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, C, T) -> (B, F, T')
        for stage in self.stages:
            x = stage(x)
        return x


# =============================================================================
# Transformer with rotary position embedding
# =============================================================================
class RotaryEmbedding(nn.Module):
    """RoPE (Su et al. 2021): rotate query/key pairs by an angle proportional to position."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("d_model / n_heads must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):          # (B, H, L, head_dim)
        L = q.shape[-2]
        ang = torch.arange(L, device=q.device, dtype=self.inv_freq.dtype)[:, None] * self.inv_freq
        cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)   # (L, head_dim / 2)

        def rotate(t: torch.Tensor) -> torch.Tensor:
            t1, t2 = t[..., 0::2], t[..., 1::2]
            return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class RopeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_base: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be a multiple of n_heads")
        self.n_heads, self.dropout = n_heads, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, L, d)
        B, L, d = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, L, d))


class TransformerLayer(nn.Module):
    """Post-norm layer as drawn in Fig. 3: attention, Add & Norm, FFN, Add & Norm."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float, rope_base: float):
        super().__init__()
        self.attn = RopeSelfAttention(d_model, n_heads, dropout, rope_base)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ffn, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ffn(x)))


# =============================================================================
# Full model
# =============================================================================
class EmgFormerModel(nn.Module):
    MODEL_FIELDS = (
        "filter", "l_freq", "h_freq", "notch_freq", "notch_max_freq", "notch_width",
        "filter_order", "filter_pad",
        "norm", "norm_eps",
        "tds_features", "tds_channels", "conv_kernels", "conv_strides", "tds_kernels",
        "se_reduction", "tds_norm",
        "d_model", "n_layers", "n_heads", "d_ffn", "dropout", "rope_base",
        "head_hidden",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        filter: bool = True,  # noqa: A002  (config field name)
        l_freq: float | None = 20.0, h_freq: float | None = 450.0,
        notch_freq: float | None = 60.0, notch_max_freq: float = 450.0, notch_width: float = 2.0,
        filter_order: int = 4, filter_pad: int = 1000,
        norm: str = "zscore", norm_eps: float = 1e-12,
        tds_features: int = 256, tds_channels: int = 8,
        conv_kernels=(11, 5, 9, 3), conv_strides=(5, 2, 5, 1), tds_kernels=(0, 0, 5, 3),
        se_reduction: int = 16, tds_norm: str = "time",
        d_model: int = 256, n_layers: int = 3, n_heads: int = 4, d_ffn: int = 512,
        dropout: float = 0.1, rope_base: float = 10000.0,
        head_hidden: int = 0,
    ):
        super().__init__()
        if norm not in ("zscore", "none"):
            raise ValueError(f"norm must be 'zscore' or 'none', got {norm!r}")
        self.filter = (ZeroPhaseFilter(sfreq, l_freq, h_freq, notch_freq, notch_max_freq,
                                       notch_width, filter_order, filter_pad)
                       if filter else nn.Identity())
        self.norm = WindowNorm(norm_eps) if norm == "zscore" else nn.Identity()
        self.featurizer = TdsFeaturizer(n_in_channels, tds_features, tds_channels,
                                        tuple(conv_kernels), tuple(conv_strides),
                                        tuple(tds_kernels), se_reduction, tds_norm)
        self.proj = nn.Identity() if tds_features == d_model else nn.Linear(tds_features, d_model)
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, d_ffn, dropout, rope_base) for _ in range(n_layers)
        )
        hidden = head_hidden or d_model
        self.head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_outputs))
        self.feature_rate = sfreq / self.featurizer.downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.norm(self.filter(x))                  # (B, C, T)   filtered, unit-free
        z = self.featurizer(x).transpose(1, 2)         # (B, T', 256) tokens at 40 Hz
        z = self.proj(z)                               # (B, T', d)
        for layer in self.layers:
            z = layer(z)
        y = self.head(z).transpose(1, 2)               # (B, J, T')  one pose per token
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)                       # (B, T, J)
