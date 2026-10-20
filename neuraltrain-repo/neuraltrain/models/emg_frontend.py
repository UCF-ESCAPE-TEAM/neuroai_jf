"""Parameter-free EMG conditioning: FIR band-pass / notch + standardisation.

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
    """

    def __init__(self, n_channels: int = 16, sfreq: float = 2000.0,
                 highpass: float | None = 20.0, lowpass: float | None = None,
                 notch: float | None = None, notch_width: float = 6.0,
                 numtaps: int = 2001, normalize: bool = True, clip: float | None = 20.0,
                 causal: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.normalize = normalize
        self.clip = clip
        self.causal = causal
        h = _design_fir(sfreq, highpass, lowpass, notch, notch_width, numtaps)
        if h is None:
            self.kernel = None
        else:
            # FIR taps (K,); not a parameter and not persisted: rebuilt from the
            # config at construction, so weights.pt stays architecture-only.
            self.register_buffer("kernel", torch.from_numpy(h), persistent=False)

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
        if self.kernel is not None:
            x = self._filter(x)
        if self.normalize:
            x = self._normalize(x)
        if self.clip is not None:
            x = x.clamp(-self.clip, self.clip)
        return x
