"""Parameter-free EMG conditioning: bad-channel rejection + interpolation,
FIR band-pass / notch, standardisation.

Non-causal (default): zero-phase centred FIR and per-window z-score, as in
offline processing (``filtfilt``-style).  Causal: the same FIR applied with
left-only padding (output at ``t`` uses ``x[t-K+1 .. t]``) and an
expanding-window z-score (running mean / SD up to ``t``), so nothing in the
output depends on samples after ``t``.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn


def _design_fir(sfreq: float, highpass: float | None, lowpass: float | None,
                notch: float | None, notch_width: float, numtaps: int) -> np.ndarray | None:
    """One multi-band linear-phase FIR of ``numtaps`` taps: band-pass edges plus a
    stop band around the notch frequency and each harmonic inside the pass band.
    Returns None when nothing is requested."""
    nyq = sfreq / 2.0
    lo = highpass if highpass is not None else 0.0
    hi = lowpass if lowpass is not None else nyq
    edges: list[float] = []
    if lo > 0:
        edges.append(lo)
    if notch is not None:
        f0 = notch
        while f0 + notch_width / 2 < hi:
            if f0 - notch_width / 2 > lo:
                edges += [f0 - notch_width / 2, f0 + notch_width / 2]
            f0 += notch
    if hi < nyq:
        edges.append(hi)
    if not edges:
        return None
    # pass_zero: DC is passed only when there is no high-pass edge
    h = signal.firwin(numtaps, edges, pass_zero=(lo <= 0), fs=sfreq)
    return h.astype(np.float32)


class EmgFrontEnd(nn.Module):
    """Band-pass / notch filter and per-channel standardization.

    Parameters
    ----------
    n_channels : int
        EMG channels (16 for emg2pose).
    sfreq : float
        Sampling rate in Hz (2000 for emg2pose).
    highpass, lowpass : float or None
        Band-pass cutoffs in Hz. ``None`` disables that side.
    notch : float or None
        Mains frequency; this and every harmonic below Nyquist are removed.
    notch_width : float
        Full stop-band width of each notch in Hz. With the default 2001 taps a
        6 Hz notch attenuates 60 Hz by ~55 dB; 801 taps need ~10 Hz for the same.
    numtaps : int
        FIR length (odd). 2001 taps at 2 kHz = 1.0 s kernel, transition ~ 4 Hz;
        use 801 for a cheaper filter when no notch is needed.
    normalize : bool
        Zero mean / unit SD per channel: over the whole window (non-causal) or
        over the samples seen so far (causal).
    clip : float or None
        Clamp |x| after normalization (in SD units) to guard against spikes.
    causal : bool
        Left-only FIR padding and expanding-window statistics. The symmetric
        FIR then has a group delay of ``(numtaps - 1) / 2`` samples (0.5 s for
        2001 taps at 2 kHz); use fewer taps (e.g. 401 -> 0.1 s) for a causal
        model.
    reject_channels : bool
        Per-window bad-channel rejection + ring interpolation, applied to the
        raw signal before the FIR. A channel is bad when, within the window,
        its RMS or its mains-band energy is an outlier across channels
        (``mean + n * std``). Bad channels are replaced by the mean of their
        two ring neighbours (circular, so channel 0 and C-1 are neighbours).
    reject_rms_nstd : float
        ``n`` for the RMS criterion.
    reject_mains : float or None
        Mains frequency for the power-line criterion (energy within
        ``+-reject_mains_bw`` Hz of the fundamental and 2nd harmonic);
        ``None`` disables that criterion.
    reject_mains_nstd, reject_mains_bw : float
        ``n`` and half-bandwidth (Hz) for the power-line criterion.
    reject_segment : int
        Causal mode only: the statistics are computed on the first
        ``reject_segment`` samples of the window and the decision applied to
        the whole window (so the first ``reject_segment`` samples are not
        strictly causal). Non-causal mode uses the whole window.
    reject_max_frac : float
        Safety valve: if more than this fraction of channels is flagged in a
        window, nothing is interpolated there (the statistics are then
        unreliable, e.g. a window of pure noise).
    reject_method : {"std", "mad"}
        Outlier rule across channels: ``mean + n*std`` (original) or the robust
        ``median + n*1.4826*MAD`` (recommended with 16 channels: it still flags
        two simultaneously bad channels, which mask each other under "std").
    """

    def __init__(self, n_channels: int = 16, sfreq: float = 2000.0,
                 highpass: float | None = 20.0, lowpass: float | None = None,
                 notch: float | None = None, notch_width: float = 6.0,
                 numtaps: int = 2001, normalize: bool = True, clip: float | None = 20.0,
                 causal: bool = False,
                 reject_channels: bool = False, reject_rms_nstd: float = 3.0,
                 reject_mains: float | None = 60.0, reject_mains_nstd: float = 3.0,
                 reject_mains_bw: float = 2.0, reject_segment: int = 400,
                 reject_max_frac: float = 0.5, reject_method: str = "std"):
        super().__init__()
        self.n_channels = n_channels
        self.sfreq = float(sfreq)
        self.normalize = normalize
        self.clip = clip
        self.causal = causal
        self.reject_channels = reject_channels
        self.reject_rms_nstd = reject_rms_nstd
        self.reject_mains = reject_mains
        self.reject_mains_nstd = reject_mains_nstd
        self.reject_mains_bw = reject_mains_bw
        self.reject_segment = reject_segment
        self.reject_max_frac = reject_max_frac
        if reject_method not in ("std", "mad"):
            raise ValueError(f"reject_method must be 'std' or 'mad', got {reject_method!r}")
        self.reject_method = reject_method
        # fraction of (window, channel) pairs interpolated in the last forward;
        # for monitoring, not part of the state
        self.last_reject_fraction: float = 0.0
        h = _design_fir(sfreq, highpass, lowpass, notch, notch_width, numtaps)
        if h is None:
            self.kernel = None
        else:
            # FIR taps (K,); not a parameter and not persisted: rebuilt from the
            # config at construction, so weights.pt stays architecture-only.
            self.register_buffer("kernel", torch.from_numpy(h), persistent=False)

    # ------------------------------------------------------------ rejection
    def _outliers(self, v: torch.Tensor, n: float) -> torch.Tensor:
        """Outliers across channels, per window. (B, C) -> bool (B, C).

        ``reject_method="std"``: ``v > mean + n * std`` (the original rule).
        ``reject_method="mad"``: ``v > median + n * 1.4826 * MAD`` -- the robust
        equivalent; with 16 channels two loud channels inflate the std enough
        to hide each other from the "std" rule, the MAD rule still flags both.
        """
        if self.reject_method == "mad":
            med = v.median(-1, keepdim=True).values
            mad = (v - med).abs().median(-1, keepdim=True).values
            return v > med + n * 1.4826 * mad
        thr = v.mean(-1, keepdim=True) + n * v.std(-1, keepdim=True, unbiased=False)
        return v > thr

    def bad_channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Bool mask (B, C) of channels to interpolate in each window of ``x`` (B, C, T)."""
        seg = x[..., : self.reject_segment] if self.causal else x
        seg = seg.float()
        rms = seg.pow(2).mean(-1).sqrt()                                 # (B, C)
        bad = self._outliers(rms, self.reject_rms_nstd)
        if self.reject_mains is not None:
            spec = torch.fft.rfft(seg, dim=-1).abs()                     # (B, C, n_freq)
            freqs = torch.fft.rfftfreq(seg.shape[-1], d=1.0 / self.sfreq).to(seg.device)
            band = torch.zeros_like(freqs, dtype=torch.bool)
            for harmonic in (1, 2):
                f0 = self.reject_mains * harmonic
                if f0 < self.sfreq / 2:
                    band |= (freqs - f0).abs() <= self.reject_mains_bw
            energy = spec[..., band].sum(-1)                             # (B, C)
            bad |= self._outliers(energy, self.reject_mains_nstd)
        # too many flagged channels -> statistics meaningless, leave the window alone
        too_many = bad.float().mean(-1, keepdim=True) > self.reject_max_frac
        return bad & ~too_many

    @staticmethod
    def _ring_interpolate(x: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
        """Replace flagged channels by the mean of their two ring neighbours.

        (B, C, T), bool (B, C) -> (B, C, T). Neighbours wrap around (the arm
        band is a ring). Neighbour values are taken from the *input*, so two
        adjacent bad channels each get a blend of one good and one bad
        neighbour -- the same as applying the per-channel rule in one pass.
        """
        neighbours = (x.roll(1, dims=1) + x.roll(-1, dims=1)) / 2
        return torch.where(bad.unsqueeze(-1), neighbours, x)

    def _reject(self, x: torch.Tensor) -> torch.Tensor:
        bad = self.bad_channel_mask(x)
        self.last_reject_fraction = bad.float().mean().item()
        if not bool(bad.any()):
            return x
        return self._ring_interpolate(x, bad)

    # ------------------------------------------------------------- filtering
    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        """'Same'-length convolution with the FIR, via FFT.

        Centred (zero delay) when non-causal, left-aligned when causal. FFT
        convolution is O(T log T) regardless of kernel length; a direct
        depthwise conv1d with a 2001-tap kernel is ~50x slower on CPU.
        """
        k = self.kernel.shape[-1]
        T = x.shape[-1]
        left, right = (k - 1, 0) if self.causal else (k // 2, k // 2)
        mode = "reflect" if T > max(left, right) else "replicate"
        xp = F.pad(x, (left, right), mode=mode)  # reflect pad: no edge transient
        n = xp.shape[-1] + k - 1
        X = torch.fft.rfft(xp.float(), n=n)
        H = torch.fft.rfft(self.kernel.float(), n=n)
        y = torch.fft.irfft(X * H, n=n)
        # full convolution -> output j uses xp[j .. j+k-1]; with the padding
        # above that is x[j-k//2 .. j+k//2] (centred) or x[j-k+1 .. j] (causal)
        y = y[..., k - 1: k - 1 + T]
        return y.to(x.dtype)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.causal:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            return (x - mean) / (std + 1e-5)
        # expanding-window statistics: mean / SD of x[.. t] at every t
        # (float64 cumsums: sums of squares over 10^4 samples exceed float32)
        xd = x.double()
        n = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=torch.float64)
        mean = xd.cumsum(-1) / n
        var = (xd * xd).cumsum(-1) / n - mean * mean
        y = (xd - mean) / (var.clamp_min(0).sqrt() + 1e-5)
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        if self.reject_channels:
            x = self._reject(x)          # on the raw signal, before any filtering
        if self.kernel is not None:
            x = self._filter(x)
        if self.normalize:
            x = self._normalize(x)
        if self.clip is not None:
            x = x.clamp(-self.clip, self.clip)
        return x
