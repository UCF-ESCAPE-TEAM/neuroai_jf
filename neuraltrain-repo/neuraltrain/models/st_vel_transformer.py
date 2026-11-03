"""Spatial-CNN + transformer EMG-to-pose model with a velocity decoder (StVelTransformer).

    EMG (B, C, T) --> EmgPreproc --> [ChannelGate] --> SpatialBlock (~50 Hz)
    --> SequenceEncoder (Conformer / transformer) --> VelocityDecoder
    --> upsample --> joint angles (B, T, n_outputs)

Same trunk as :class:`StPoseTransformerModel`; only the readout differs. Where
the pose model regresses the angle of every frame independently, this decoder
emits a *position* and a *velocity* per joint and per frame and turns them into
a trajectory by integration (VEMG2Pose's hybrid parameterisation):

    pose_t = position_t                             for t <  position_steps
    pose_t = pose_{t-1} + vel_scale * velocity_t    for t >= position_steps

Why a dedicated velocity decoder, and what the knobs are for
-------------------------------------------------------------
* **Integration smooths.** The integrated trajectory is continuous by
  construction; the per-frame angle readout is not. Fast transients are
  handled well because the velocity head only has to get the *change* right.
* **Integration drifts.** A memoryless velocity head cannot know it has
  drifted, so over a 5-s window the integrated angle walks away from the truth
  even if each step is decent. VEMG2Pose counters this with an LSTM whose state
  implicitly tracks the pose. Two mechanisms here:

  - ``vel_gru_hidden > 0``: a small GRU runs over the encoder frames before the
    velocity readout, giving the velocity head a recurrent state (one
    vectorised ``nn.GRU`` call, no per-frame Python loop, no pose feedback).
  - ``fuse_gain > 0``: a complementary filter. The position head is read out
    at every frame and the integrated pose is pulled towards it:

        pose_t = (1 - g) * (pose_{t-1} + vel_scale * velocity_t) + g * position_t

    so the velocity head governs the high frequencies (smooth, responsive) and
    the position head anchors the low frequencies (no drift). ``g = 0`` is the
    pure VEMG2Pose hybrid, ``g = 1`` reduces to the pose model. Both heads
    are trained through the fused output, so no auxiliary loss is needed.

The integration is causal by construction; the encoder in front of it is
centred unless ``causal=True``.
"""

import typing as tp

import torch
from torch import nn

from .base import BaseBrainModelConfig
from .emg_frontend import EmgPreproc, frontend_from_config
from .st_pose_transformer import (
    ENCODER_FIELDS,
    SPATIAL_FIELDS,
    ChannelGate,
    SequenceEncoder,
    make_spatial_block,
    upsample_frames,
)


# =============================================================================
# Config
# =============================================================================
class StVelTransformer(BaseBrainModelConfig):
    """Config for :class:`StVelTransformerModel` (front end / trunk fields as StPoseTransformer)."""

    # ---- front end (EmgPreproc) ----
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
    # ---- channel attention ----
    channel_attention: bool = False
    ca_hidden: int = 32
    # ---- spatial 2-D CNN ----
    f1: int = 32
    k_e: int = 65
    ms_kernels: tuple[int, ...] | None = None
    spatial_mode: tp.Literal["conv2d", "none"] = "conv2d"
    spatial_layers: int = 2
    k_s: int = 9
    depth_mult: int = 2
    f2: int = 128
    k_sep: int = 16
    pool1: int = 5
    pool2: int = 8
    spatial_dropout: float = 0.2
    # ---- sequence encoder ----
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    enc_dropout: float = 0.1
    attn_dropout: float = 0.1
    block_style: tp.Literal["conformer", "transformer"] = "conformer"
    conv_kernel: int = 15
    pos_enc: tp.Literal["rope", "sinusoid", "none"] = "rope"
    attn_window: int | None = None
    # ---- velocity decoder ----
    position_steps: int = 12          # frames read as position before integrating (240 ms)
    vel_scale: float = 0.02           # rad per frame per unit velocity readout
    fuse_gain: float = 0.0            # complementary-filter gain towards the position head
    vel_gru_hidden: int = 0           # >0: GRU state before the velocity readout
    vel_gru_layers: int = 1
    # ---- causality ----
    causal: bool = False

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "StVelTransformer requires n_outputs"
        frontend = frontend_from_config(n_spatial_locations, self.model_dump())
        model_kwargs = {k: getattr(self, k) for k in StVelTransformerModel.MODEL_FIELDS}
        return StVelTransformerModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                                     sfreq=self.sfreq, frontend=frontend, **model_kwargs)


# =============================================================================
# Velocity decoder
# =============================================================================
class VelocityDecoder(nn.Module):
    """Encoder frames ``(B, d, T')`` -> pose trajectory ``(B, J, T')``.

    Two readouts: ``pos`` (1x1 conv, an absolute angle per frame) and ``vel``
    (1x1 conv on the encoder frames, or a linear map on a GRU run over them).
    The velocity readout starts at zero so the trajectory begins as "hold the
    position" instead of a random walk. ``fuse_gain`` blends the integrated
    pose with the position readout at every step (see module docstring).
    """

    def __init__(self, d_model: int, n_outputs: int, position_steps: int = 12,
                 vel_scale: float = 0.02, fuse_gain: float = 0.0,
                 gru_hidden: int = 0, gru_layers: int = 1):
        super().__init__()
        if not 0.0 <= fuse_gain <= 1.0:
            raise ValueError(f"fuse_gain must be in [0, 1], got {fuse_gain}")
        self.n_outputs = n_outputs
        self.position_steps = position_steps
        self.vel_scale = vel_scale
        self.fuse_gain = fuse_gain
        self.pos = nn.Conv1d(d_model, n_outputs, 1)
        if gru_hidden > 0:
            self.gru = nn.GRU(d_model, gru_hidden, num_layers=gru_layers, batch_first=True)
            self.vel = nn.Linear(gru_hidden, n_outputs)
            for name, p in self.gru.named_parameters():
                nn.init.normal_(p, std=0.02) if "weight" in name else nn.init.zeros_(p)
        else:
            self.gru = None
            self.vel = nn.Conv1d(d_model, n_outputs, 1)
        with torch.no_grad():                                   # "hold the position" at init
            self.vel.weight.zero_()
            self.vel.bias.zero_()

    def readouts(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``z`` (B, d, T') -> position (B, J, T'), velocity (B, J, T')."""
        pos = self.pos(z)
        if self.gru is None:
            return pos, self.vel(z)
        h, _ = self.gru(z.transpose(1, 2))                      # (B, T', H), h0 = 0
        return pos, self.vel(h).transpose(1, 2)

    def integrate(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        T = pos.shape[-1]
        k = min(self.position_steps, T)
        if k == T:
            return pos
        step = vel[..., k:] * self.vel_scale                    # (B, J, T'-k)
        if self.fuse_gain == 0.0:                               # pure integration: one cumsum
            integrated = pos[..., k - 1:k] + torch.cumsum(step, dim=-1)
            return torch.cat([pos[..., :k], integrated], dim=-1)
        g = self.fuse_gain
        prev, out = pos[..., k - 1], []
        for t in range(T - k):                                  # first-order IIR, T'-k cheap steps
            prev = (1.0 - g) * (prev + step[..., t]) + g * pos[..., k + t]
            out.append(prev)
        return torch.cat([pos[..., :k], torch.stack(out, dim=-1)], dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pos, vel = self.readouts(z)
        return self.integrate(pos, vel)


# =============================================================================
# Full model
# =============================================================================
class StVelTransformerModel(nn.Module):
    MODEL_FIELDS = (
        "channel_attention", "ca_hidden",
        *SPATIAL_FIELDS, *ENCODER_FIELDS,
        "position_steps", "vel_scale", "fuse_gain", "vel_gru_hidden", "vel_gru_layers",
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
        # velocity decoder
        position_steps=12, vel_scale=0.02, fuse_gain=0.0, vel_gru_hidden=0, vel_gru_layers=1,
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
        self.decoder = VelocityDecoder(d_model, n_outputs, position_steps, vel_scale,
                                       fuse_gain, vel_gru_hidden, vel_gru_layers)
        self.feature_rate = sfreq / self.spatial.downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.gate(self.frontend(x))    # (B, C, T)
        z = self.spatial(x)                # (B, f2, T')  ~50 Hz
        z = self.encoder(z)                # (B, d, T')
        y = self.decoder(z)                # (B, J, T')   integrated trajectory
        y = upsample_frames(y, n_samples, self.causal, self.spatial.downsample)
        return y.transpose(1, 2)           # (B, T, J)
