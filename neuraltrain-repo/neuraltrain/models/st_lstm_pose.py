"""Spatial-CNN + LSTM EMG-to-pose model (StLstmPose).

    EMG (B, C, T) --> EmgPreproc (no params) --> SpatialBlock (EEGNet-style 2-D
    CNN over the electrode ring, pools to ~50 Hz) --> LstmPoseDecoder (VEMG2Pose-
    style recurrent rollout) --> upsample --> joint angles (B, T, n_outputs)

Same front end and spatial block as :class:`StTcnPose`; only the temporal part
differs. Where StTcnPose runs a stack of dilated convolutions and reads every
frame out independently, this model *walks* through the frames one by one with
an LSTM, carrying a hidden state and (optionally) its own previous pose
prediction from frame to frame -- the decoder VEMG2Pose uses.

Reading guide for the LSTM part (see also lstm_tutorial.md):

* ``nn.LSTM(input_size, hidden_size, num_layers)`` holds the weights of the
  recurrent cell(s). Calling it on a sequence ``(B, L, input_size)`` with a
  state ``(h, c)`` returns the outputs ``(B, L, hidden_size)`` and the new
  state. ``h`` ("hidden") is what the cell exposes at each step, ``c``
  ("cell") is its long-term memory; both have shape
  ``(num_layers, B, hidden_size)`` and start at zeros for every window.
* We call it with ``L = 1`` inside a Python loop over the 50 Hz frames,
  because the input of step t contains the pose we predicted at step t-1
  (state conditioning). That feedback is what makes the trajectory
  continuous -- and it is why the rollout cannot be done in one vectorised
  call. With ``condition_on_pose=False`` the loop disappears and the whole
  sequence goes through ``nn.LSTM`` at once (much faster, no feedback).
* The LSTM is causal by construction: step t only ever sees frames <= t.
  The spatial block in front of it is centred unless ``causal=True``.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig
from .emg_frontend import EmgPreproc, frontend_from_config
from .st_tcn_pose import SpatialBlock


class StLstmPose(BaseBrainModelConfig):
    """Config for :class:`StLstmPoseModel` (front end + spatial fields as StTcnPose)."""

    # front end (EmgPreproc) -- identical to StTcnPose
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
    # spatial 2-D CNN -- identical to StTcnPose
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
    # LSTM decoder (VEMG2Pose defaults: feature_dim 64, hidden 512, 2 layers,
    # hybrid parameterisation, 250 ms of position steps, output_scalar 0.01)
    feature_dim: int = 64
    hidden_size: int = 256
    lstm_layers: int = 2
    condition_on_pose: bool = True
    bidirectional: bool = False
    parameterization: tp.Literal["hybrid", "velocity", "position"] = "hybrid"
    position_steps: int = 12
    output_scalar: float = 0.01
    decoder_dropout: float = 0.0
    # causality of the front end / spatial block (the LSTM rollout is always causal)
    causal: bool = False

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "StLstmPose requires n_outputs"
        frontend = frontend_from_config(n_spatial_locations, self.model_dump())
        model_kwargs = {k: getattr(self, k) for k in StLstmPoseModel.MODEL_FIELDS}
        return StLstmPoseModel(n_in_channels=n_spatial_locations, n_outputs=n_outputs,
                               sfreq=self.sfreq, frontend=frontend, **model_kwargs)


# =============================================================================
# LSTM decoder
# =============================================================================
class LstmPoseDecoder(nn.Module):
    """Features ``(B, F, T')`` at the frame rate -> joint angles ``(B, J, T')``.

    A re-implementation of VEMG2Pose's recurrent decoder (Salter et al. 2024;
    braindecode ``VEMG2Pose``), without the TDS encoder in front of it.

    Two ways to run it
    ------------------
    ``condition_on_pose=True`` (VEMG2Pose, default)
        A Python loop over the T' frames. At step t the LSTM input is
        ``[feature_t ; pose_{t-1}]`` -- the current EMG feature *and the pose
        the decoder itself predicted one frame earlier*. The LSTM updates its
        state, a linear head turns the new hidden vector into a position and
        a velocity for every joint, and the pose is advanced from them (see
        ``parameterization``). Because each step feeds on the previous
        output, the steps are sequential: 250 LSTM calls per 5 s window.
    ``condition_on_pose=False``
        No feedback. The whole feature sequence is passed to ``nn.LSTM`` in
        one call (cuDNN runs the recurrence internally, ~20x faster), then
        the head is applied to every frame at once. May be ``bidirectional``
        (a second LSTM reads the sequence backwards; outputs are
        concatenated) when the model is not meant to be causal.

    Parameterization (how the head output becomes a pose; VEMG2Pose names)
    ---------------------------------------------------------------------
    ``"position"``  pose_t = head_t                      (read the angle out)
    ``"velocity"``  pose_t = pose_{t-1} + head_t          (integrate a velocity)
    ``"hybrid"``    position for the first ``position_steps`` frames, velocity
                    integration afterwards -- what ``regression_vemg2pose``
                    was trained as. The head emits 2J numbers in this mode.
    All head outputs are multiplied by ``output_scalar`` (0.01 in VEMG2Pose)
    so that the trajectory starts near zero and moves in small steps.
    """

    def __init__(self, in_features: int, n_outputs: int, hidden_size: int = 256,
                 num_layers: int = 2, condition_on_pose: bool = True,
                 bidirectional: bool = False, parameterization: str = "hybrid",
                 position_steps: int = 12, output_scalar: float = 0.01,
                 dropout: float = 0.0):
        super().__init__()
        if parameterization not in ("hybrid", "velocity", "position"):
            raise ValueError(f"parameterization must be hybrid|velocity|position, got {parameterization!r}")
        if bidirectional and condition_on_pose:
            raise ValueError("bidirectional=True needs condition_on_pose=False (no feedback loop to run backwards)")
        self.n_outputs = n_outputs
        self.condition_on_pose = condition_on_pose
        self.bidirectional = bidirectional
        self.parameterization = parameterization
        self.position_steps = position_steps
        self.output_scalar = output_scalar

        # The recurrent core. Input = feature (+ previous pose when conditioning).
        #   input_size  : length of the vector fed in at every step
        #   hidden_size : length of h and c; the "width" of the decoder
        #   num_layers  : stacked LSTMs -- layer 2 reads layer 1's h sequence
        #   batch_first : tensors are (B, L, features) rather than (L, B, features)
        #   dropout     : between stacked layers only (PyTorch applies none after the last)
        lstm_in = in_features + (n_outputs if condition_on_pose else 0)
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.activation = nn.LeakyReLU()
        head_in = hidden_size * (2 if bidirectional else 1)
        head_out = n_outputs * (2 if parameterization == "hybrid" else 1)
        self.head = nn.Linear(head_in, head_out)

        # VEMG2Pose / braindecode initialisation: small truncated-normal weights,
        # zero biases. Small LSTM weights keep the gates near 0.5 at the start,
        # so gradients flow through the 250-step recurrence instead of saturating.
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.zeros_(p)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------ one step
    def _advance(self, output: torch.Tensor, prev_pose: torch.Tensor, step: int) -> torch.Tensor:
        """Head output (B, head_out) + previous pose (B, J) -> pose at this step (B, J)."""
        output = output * self.output_scalar
        if self.parameterization == "hybrid":
            position, velocity = torch.split(output, self.n_outputs, dim=-1)
            return position if step < self.position_steps else prev_pose + velocity
        if self.parameterization == "velocity":
            return prev_pose + output
        return output                                              # position

    # ------------------------------------------------------------ forward
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` (B, F, T') -> poses (B, J, T')."""
        B, _, n_steps = z.shape
        seq = z.transpose(1, 2)                                     # (B, T', F): time-major for the LSTM

        if not self.condition_on_pose:
            # -------- fast path: whole sequence through the LSTM at once.
            # h0/c0 default to zeros. ``out`` holds the top layer's h at every step.
            out, _ = self.lstm(seq)                                 # (B, T', hidden * dirs)
            head = self.head(self.activation(out))                  # (B, T', head_out)
            poses, pose = [], z.new_zeros(B, self.n_outputs)
            for t in range(n_steps):                                # integration only (cheap)
                pose = self._advance(head[:, t], pose, t)
                poses.append(pose)
            return torch.stack(poses, dim=-1)                       # (B, J, T')

        # -------- VEMG2Pose path: step-by-step rollout with pose feedback.
        # State for every layer: (num_layers, B, hidden). Zeros = "nothing seen yet".
        n_layers = self.lstm.num_layers
        h = z.new_zeros(n_layers, B, self.lstm.hidden_size)
        c = torch.zeros_like(h)
        pose = z.new_zeros(B, self.n_outputs)                       # regression mode: start from zero pose
        poses = []
        for t in range(n_steps):
            # 1. build this step's input: current EMG feature + the pose we just predicted
            x_t = torch.cat([seq[:, t], pose], dim=-1).unsqueeze(1)  # (B, 1, F + J)
            # 2. one recurrent step: the LSTM reads x_t and its own memory (h, c),
            #    decides what to forget / store / expose, and returns the new (h, c).
            out, (h, c) = self.lstm(x_t, (h, c))                    # out: (B, 1, hidden)
            # 3. read the pose out of the exposed state and advance the trajectory
            pose = self._advance(self.head(self.activation(out[:, 0])), pose, t)
            poses.append(pose)
        return torch.stack(poses, dim=-1)                           # (B, J, T')


# =============================================================================
# Full model
# =============================================================================
class StLstmPoseModel(nn.Module):
    MODEL_FIELDS = (
        "f1", "k_e", "spatial_mode", "spatial_layers", "k_s", "depth_mult",
        "f2", "k_sep", "pool1", "pool2", "spatial_dropout",
        "feature_dim", "hidden_size", "lstm_layers", "condition_on_pose", "bidirectional",
        "parameterization", "position_steps", "output_scalar", "decoder_dropout",
        "causal",
    )

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sfreq: float = 2000.0,
        # spatial (as StTcnPose)
        f1=16, k_e=65, spatial_mode="conv2d", spatial_layers=2, k_s=9, depth_mult=2,
        f2=64, k_sep=16, pool1=5, pool2=8, spatial_dropout=0.2,
        # LSTM decoder
        feature_dim=64, hidden_size=256, lstm_layers=2, condition_on_pose=True,
        bidirectional=False, parameterization="hybrid", position_steps=12,
        output_scalar=0.01, decoder_dropout=0.0,
        # causality
        causal: bool = False,
        # preprocessing
        frontend: nn.Module | None = None,
    ):
        super().__init__()
        if causal and bidirectional:
            raise ValueError("a bidirectional LSTM reads the future: not allowed with causal=True")
        self.causal = causal
        self.frontend = frontend if frontend is not None else EmgPreproc(
            n_in_channels, sfreq=sfreq, norm="zscore", causal=causal)
        # Same spatial block as StTcnPose -> (B, f2, T') at sfreq / (pool1 * pool2)
        self.spatial = SpatialBlock(
            n_in_channels, f1, k_e, spatial_mode, spatial_layers, k_s, depth_mult,
            f2, k_sep, pool1, pool2, spatial_dropout, causal,
        )
        # VEMG2Pose projects its encoder output to a small feature vector before
        # the LSTM; a 1x1 conv is that linear map applied at every frame.
        self.project = nn.Conv1d(f2, feature_dim, 1)
        self.decoder = LstmPoseDecoder(
            feature_dim, n_outputs, hidden_size, lstm_layers, condition_on_pose,
            bidirectional, parameterization, position_steps, output_scalar, decoder_dropout,
        )
        self.feature_rate = sfreq / self.spatial.downsample

    def _upsample(self, y: torch.Tensor, n_samples: int) -> torch.Tensor:
        """(B, J, T') at the frame rate -> (B, J, T) at the input rate (as StTcnPose)."""
        if not self.causal:
            return F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        P = self.spatial.downsample
        t = torch.arange(n_samples, device=y.device)
        idx = ((t + 1) // P - 1).clamp_(0, y.shape[-1] - 1)         # last completed frame
        return y[..., idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)``."""
        n_samples = x.shape[-1]
        x = self.frontend(x)              # (B, C, T)    filtered + scaled, no parameters
        z = self.spatial(x)               # (B, f2, T')  spatial features at ~50 Hz
        z = self.project(z)               # (B, feature_dim, T')
        y = self.decoder(z)               # (B, J, T')   LSTM rollout, one pose per frame
        y = self._upsample(y, n_samples)  # (B, J, T)
        return y.transpose(1, 2)          # (B, T, J)
