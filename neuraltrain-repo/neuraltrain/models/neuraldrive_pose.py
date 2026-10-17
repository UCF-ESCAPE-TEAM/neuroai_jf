"""Neural-drive-style EMG-to-pose model in three stages.

    EMG (B, C, T) --> [1] NeuralDriveDecoder --> event trains e (B, K, T) @ 2 kHz
                      [2] DriveFeatureExtractor --> features (B, F, T/40) @ 50 Hz
                      [3] KinematicsDecoder --> joint angles (B, T, n_outputs)

[1] is an unrolled convolutive BSS (Negro et al. 2016): band-pass front-end ->
mean/gain removal -> extension (R delays) -> whitening -> separation vectors ->
spike detection. It is trained with the ICA *objective* (non-Gaussianity,
decorrelation, sparsity) returned by :meth:`auxiliary_losses`, never with BSS
outputs as labels, so it can also be pretrained on unlabeled EMG and frozen
(``freeze_drive``).

[2] starts with a depthwise "twitch" filter per source (Hann init, so at init
it is the population firing-rate pooling) followed by conv blocks that mix
across sources: the spike-train -> muscle-force stage of neural-drive models.

[3] is a temporal conv stack (or a bidirectional GRU) with a linear head.

``forward``'s output goes to the task loss as usual; :meth:`auxiliary_losses`
returns the decomposition losses for the patched ``BrainModule.training_step``
to add, exactly as :class:`UlhmPoseModel` does.
"""

import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBrainModelConfig


class NeuralDrivePose(BaseBrainModelConfig):
    """Config for the three-stage neural-drive EMG-to-pose model.

    Parameters
    ----------
    sample_rate : int
        Input sampling rate in Hz (2000 for emg2pose).
    band_low_hz, band_high_hz : float
        Band-pass edges of the EMG front-end (Negro et al. 2016 use 10-900 Hz
        at 10 kHz; 20-500 Hz is the usual sEMG decomposition band at 2 kHz).
        ``band_high_hz`` <= 0 disables the filter.
    filter_taps : int
        Length of the zero-phase FIR (odd); 201 taps = 100 ms at 2 kHz.
    learnable_filter : bool
        Fine-tune the FIR from its band-pass initialisation.
    n_delays : int
        Extension factor R. Negro et al. use R ~ 1000 / n_channels (64 for
        16 channels); 32 (16 ms at 2 kHz) covers a MUAP at a quarter of the
        memory.
    n_sources : int
        Number of separation vectors K.
    threshold_init : float
        Initial spike-detection threshold on mean-normalised source power.
    ica_weight, decorr_weight, sparsity_weight : float
        Weights of the self-supervised decomposition losses.
    freeze_drive : bool
        Freeze stage [1] (e.g. after pretraining it on unlabeled EMG).
    twitch_ms : float
        Kernel of the per-source twitch/rate filter in stage [2].
    rate_stride_ms : float
        Stride of that filter: 20 ms -> features at 50 Hz.
    feature_hidden, feature_depth, feature_kernel : int
        Conv blocks that mix across sources in stage [2]. ``feature_depth=0``
        passes the K per-source rates straight to the decoder.
    decoder_kind : {"conv", "gru"}
        Stage [3] architecture.
    decoder_hidden, decoder_depth, decoder_kernel : int
        Size of stage [3] (``decoder_kernel`` is used by "conv" only).
    dropout : float
        Dropout in stages [2] and [3].
    augment : bool
        Per-channel gain jitter and +-1 channel rotation during training.
    gain_jitter : float
        Half-range of the multiplicative gain jitter.
    """

    sample_rate: int = 2000
    # [1] neural-drive decoder
    band_low_hz: float = 20.0
    band_high_hz: float = 500.0
    filter_taps: int = 201
    learnable_filter: bool = False
    n_delays: int = 32
    n_sources: int = 32
    threshold_init: float = 2.0
    ica_weight: float = 0.1
    decorr_weight: float = 0.1
    sparsity_weight: float = 0.01
    freeze_drive: bool = False
    # [2] drive feature extractor
    twitch_ms: float = 100.0
    rate_stride_ms: float = 20.0
    feature_hidden: int = 128
    feature_depth: int = 2
    feature_kernel: int = 9
    # [3] kinematics decoder
    decoder_kind: tp.Literal["conv", "gru"] = "conv"
    decoder_hidden: int = 128
    decoder_depth: int = 2
    decoder_kernel: int = 9
    dropout: float = 0.1
    # augmentation
    augment: bool = True
    gain_jitter: float = 0.3

    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None, "NeuralDrivePose requires n_outputs"
        return NeuralDrivePoseModel(
            n_in_channels=n_spatial_locations,
            n_outputs=n_outputs,
            sample_rate=self.sample_rate,
            band_low_hz=self.band_low_hz,
            band_high_hz=self.band_high_hz,
            filter_taps=self.filter_taps,
            learnable_filter=self.learnable_filter,
            n_delays=self.n_delays,
            n_sources=self.n_sources,
            threshold_init=self.threshold_init,
            ica_weight=self.ica_weight,
            decorr_weight=self.decorr_weight,
            sparsity_weight=self.sparsity_weight,
            freeze_drive=self.freeze_drive,
            twitch_ms=self.twitch_ms,
            rate_stride_ms=self.rate_stride_ms,
            feature_hidden=self.feature_hidden,
            feature_depth=self.feature_depth,
            feature_kernel=self.feature_kernel,
            decoder_kind=self.decoder_kind,
            decoder_hidden=self.decoder_hidden,
            decoder_depth=self.decoder_depth,
            decoder_kernel=self.decoder_kernel,
            dropout=self.dropout,
            augment=self.augment,
            gain_jitter=self.gain_jitter,
        )


def _bandpass_fir(low_hz: float, high_hz: float, sample_rate: float, n_taps: int) -> torch.Tensor:
    """Windowed-sinc (Hamming) band-pass FIR, symmetric -> zero phase with centred padding."""
    assert n_taps % 2 == 1, "filter_taps must be odd for zero phase"
    n = torch.arange(n_taps, dtype=torch.float64) - (n_taps - 1) / 2
    window = torch.hamming_window(n_taps, periodic=False, dtype=torch.float64)

    def lowpass(fc_hz: float) -> torch.Tensor:
        fc = fc_hz / sample_rate  # cycles / sample
        h = 2 * fc * torch.sinc(2 * fc * n) * window
        return h / h.sum()  # unit DC gain

    high_hz = min(high_hz, 0.499 * sample_rate)
    h = lowpass(high_hz)
    if low_hz > 0:
        h = h - lowpass(low_hz)
    return h.to(torch.float32)


class EmgFrontEnd(nn.Module):
    """Negro et al. (2016) conditioning: band-pass + mean subtraction.

    The band-pass is a depthwise zero-phase FIR; the mean subtraction (and
    per-channel gain removal) is done by the InstanceNorm that follows.
    """

    def __init__(
        self,
        n_channels: int,
        sample_rate: float,
        low_hz: float = 20.0,
        high_hz: float = 500.0,
        n_taps: int = 201,
        learnable: bool = False,
    ):
        super().__init__()
        self.enabled = high_hz > 0
        self.norm = nn.InstanceNorm1d(n_channels, affine=False)
        if not self.enabled:
            return
        h = _bandpass_fir(low_hz, high_hz, sample_rate, n_taps)
        self.filt = nn.Conv1d(n_channels, n_channels, n_taps, padding=n_taps // 2,
                              groups=n_channels, bias=False)
        with torch.no_grad():
            self.filt.weight.copy_(h.expand(n_channels, 1, n_taps))
        self.filt.weight.requires_grad_(learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            x = self.filt(x)
        return self.norm(x)


def _extend(x: torch.Tensor, n_delays: int) -> torch.Tensor:
    """Extension step of CKC/FastICA: (B, C, T) -> (B, C*R, T), delay r in 0..R-1."""
    T = x.shape[-1]
    shifted = [F.pad(x, (r, 0))[..., :T] for r in range(n_delays)]
    return torch.cat(shifted, dim=1)


def _block(c_in: int, c_out: int, kernel: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


# ============================================================ 1. neural drive
class NeuralDriveDecoder(nn.Module):
    """EMG ``(B, C, T)`` -> sparse event trains ``(B, K, T)`` at the input rate.

    Unrolled convolutive BSS: front-end -> extension -> whitening -> separation
    -> spike detection. Carries the self-supervised decomposition losses, so it
    can be trained (or pretrained) without any labels.
    """

    def __init__(
        self,
        n_in_channels: int,
        sample_rate: int,
        band_low_hz: float,
        band_high_hz: float,
        filter_taps: int,
        learnable_filter: bool,
        n_delays: int,
        n_sources: int,
        threshold_init: float,
        ica_weight: float,
        decorr_weight: float,
        sparsity_weight: float,
    ):
        super().__init__()
        self.n_delays = n_delays
        n_ext = n_in_channels * n_delays
        self.front_end = EmgFrontEnd(n_in_channels, sample_rate, band_low_hz,
                                     band_high_hz, filter_taps, learnable_filter)
        self.whiten = nn.Conv1d(n_ext, n_ext, 1, bias=False)
        nn.init.eye_(self.whiten.weight[:, :, 0])
        self.separate = nn.Conv1d(n_ext, n_sources, 1, bias=False)
        nn.init.orthogonal_(self.separate.weight[:, :, 0])
        self.threshold = nn.Parameter(torch.full((1, n_sources, 1), threshold_init))
        self.ica_weight = ica_weight
        self.decorr_weight = decorr_weight
        self.sparsity_weight = sparsity_weight

    @torch.no_grad()
    def fit_whitening(self, x: torch.Tensor, eps: float | None = None) -> None:
        """ZCA init of the whitening layer from a batch of raw EMG (B, C, T).

        ``eps`` regularises the eigenvalues; ``None`` uses Negro et al.'s rule
        (mean of the smallest half of the eigenvalues, a noise-variance estimate).
        """
        xe = _extend(self.front_end(x), self.n_delays)
        xe = xe.transpose(0, 1).reshape(xe.shape[1], -1)
        cov = (xe @ xe.T / xe.shape[1]).double()
        evals, evecs = torch.linalg.eigh(cov)
        if eps is None:
            eps = evals[: evals.numel() // 2].mean().clamp_min(1e-8).item()
        w = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
        self.whiten.weight.copy_(w.float().unsqueeze(-1))

    def sources(self, x: torch.Tensor) -> torch.Tensor:
        """Separated sources (B, K, T) before spike detection."""
        x = self.front_end(x)
        return self.separate(self.whiten(_extend(x, self.n_delays)))

    def events(self, s: torch.Tensor) -> torch.Tensor:
        """Sign-free, amplitude-invariant event trains (B, K, T)."""
        p = s.pow(2)
        p = p / (p.mean(dim=-1, keepdim=True) + 1e-6)
        return F.relu(p - F.softplus(self.threshold))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.sources(x)
        return s, self.events(s)

    def losses(self, s: torch.Tensor, e: torch.Tensor) -> dict[str, torch.Tensor]:
        z = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)
        kurt = z.pow(4).mean(-1) - 3.0  # (B, K): super-Gaussian = spiky
        ica = -torch.log1p(F.relu(kurt)).mean()
        gram = z @ z.transpose(1, 2) / z.shape[-1]  # (B, K, K)
        off = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        decorr = off.pow(2).mean()
        sparsity = e.mean()
        return {
            "ica": self.ica_weight * ica,
            "decorr": self.decorr_weight * decorr,
            "sparsity": self.sparsity_weight * sparsity,
        }


# ======================================================= 2. drive -> features
class DriveFeatureExtractor(nn.Module):
    """Event trains ``(B, K, T)`` -> features ``(B, F, T / stride)``.

    Stage A is a depthwise "twitch" filter per source (kernel ``twitch_ms``,
    stride ``rate_stride_ms``), initialised to a Hann window so that at init it
    equals the population firing-rate pooling; training turns it into a learned
    spike-to-force kernel per source. Stage B mixes across sources with conv
    blocks (synergy-level features) at the pooled rate.
    """

    def __init__(
        self,
        n_sources: int,
        sample_rate: int,
        twitch_ms: float,
        rate_stride_ms: float,
        hidden: int,
        depth: int,
        kernel: int,
        dropout: float,
    ):
        super().__init__()
        taps = int(sample_rate * twitch_ms / 1000) | 1  # odd
        stride = int(sample_rate * rate_stride_ms / 1000)
        self.twitch = nn.Conv1d(n_sources, n_sources, taps, stride=stride,
                                padding=taps // 2, groups=n_sources, bias=False)
        with torch.no_grad():
            w = torch.hann_window(taps, periodic=False)
            self.twitch.weight.copy_((w / w.sum()).expand(n_sources, 1, taps))
        self.norm = nn.BatchNorm1d(n_sources)
        layers, c_in = [], n_sources
        for _ in range(depth):
            layers.append(_block(c_in, hidden, kernel, dropout))
            c_in = hidden
        self.mix = nn.Sequential(*layers)
        self.n_features = c_in

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        r = self.norm(torch.log1p(F.relu(self.twitch(e))))
        return self.mix(r)


# ====================================================== 3. features -> pose
class KinematicsDecoder(nn.Module):
    """Features ``(B, F, T')`` -> joint angles ``(B, n_outputs, T')``.

    ``kind="conv"``: temporal conv blocks + 1x1 head.
    ``kind="gru"``: bidirectional GRU + linear head (smoother dynamics).
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden: int,
        depth: int,
        kernel: int,
        dropout: float,
        kind: str = "conv",
    ):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(n_features, hidden // 2, num_layers=max(depth, 1),
                              batch_first=True, bidirectional=True,
                              dropout=dropout if depth > 1 else 0.0)
            self.head = nn.Linear(hidden, n_outputs)
        else:
            layers, c_in = [], n_features
            for _ in range(depth):
                layers.append(_block(c_in, hidden, kernel, dropout))
                c_in = hidden
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(c_in, n_outputs, kernel_size=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.kind == "gru":
            h, _ = self.rnn(f.transpose(1, 2))
            return self.head(h).transpose(1, 2)
        return self.head(self.body(f))


# ================================================================ full model
class NeuralDrivePoseModel(nn.Module):
    """``nn.Module`` implementation of :class:`NeuralDrivePose`."""

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        sample_rate: int = 2000,
        # neural-drive decoder
        band_low_hz: float = 20.0,
        band_high_hz: float = 500.0,
        filter_taps: int = 201,
        learnable_filter: bool = False,
        n_delays: int = 32,
        n_sources: int = 32,
        threshold_init: float = 2.0,
        ica_weight: float = 0.1,
        decorr_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        freeze_drive: bool = False,
        # feature extractor
        twitch_ms: float = 100.0,
        rate_stride_ms: float = 20.0,
        feature_hidden: int = 128,
        feature_depth: int = 2,
        feature_kernel: int = 9,
        # kinematics decoder
        decoder_kind: str = "conv",
        decoder_hidden: int = 128,
        decoder_depth: int = 2,
        decoder_kernel: int = 9,
        dropout: float = 0.1,
        # augmentation
        augment: bool = True,
        gain_jitter: float = 0.3,
    ):
        super().__init__()
        self.drive = NeuralDriveDecoder(
            n_in_channels, sample_rate, band_low_hz, band_high_hz, filter_taps,
            learnable_filter, n_delays, n_sources, threshold_init,
            ica_weight, decorr_weight, sparsity_weight,
        )
        self.features = DriveFeatureExtractor(
            n_sources, sample_rate, twitch_ms, rate_stride_ms,
            feature_hidden, feature_depth, feature_kernel, dropout,
        )
        self.decoder = KinematicsDecoder(
            self.features.n_features, n_outputs, decoder_hidden, decoder_depth,
            decoder_kernel, dropout, decoder_kind,
        )
        self.freeze_drive = freeze_drive
        if freeze_drive:
            for p in self.drive.parameters():
                p.requires_grad_(False)
        self.augment = augment
        self.gain_jitter = gain_jitter
        self._reg: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------ helpers
    def fit_whitening(self, x: torch.Tensor, eps: float | None = None) -> None:
        self.drive.fit_whitening(x, eps)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.augment):
            return x
        B, C, _ = x.shape
        gain = 1.0 + self.gain_jitter * (2 * torch.rand(B, C, 1, device=x.device) - 1)
        x = x * gain
        shift = int(torch.randint(-1, 2, (1,)).item())  # wristband electrode shift
        if shift:
            x = torch.roll(x, shifts=shift, dims=1)
        return x

    def neural_drive(self, x: torch.Tensor) -> torch.Tensor:
        """Event trains (B, K, T): the decoded neural drive, for inspection."""
        return self.drive(x)[1]

    # ------------------------------------------------------------ forward
    def forward(self, x, channel_positions=None):
        """EMG ``(B, C, T)`` -> joint angles ``(B, T, n_outputs)`` (time-major)."""
        n_samples = x.shape[-1]
        x = self._augment(x)
        if self.freeze_drive:
            with torch.no_grad():
                s, e = self.drive(x)
            self._reg = {}
        else:
            s, e = self.drive(x)
            self._reg = self.drive.losses(s, e) if self.training else {}
        y = self.decoder(self.features(e))
        y = F.interpolate(y, size=n_samples, mode="linear", align_corners=False)
        return y.transpose(1, 2)

    def auxiliary_losses(self, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Self-supervised decomposition losses from the last training forward().

        Same signature as ``UlhmPoseModel.auxiliary_losses``; ``target`` is
        accepted for compatibility and ignored (these losses need no labels).
        """
        reg, self._reg = self._reg, {}
        return reg
