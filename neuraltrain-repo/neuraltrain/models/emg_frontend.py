"""Parameter-free EMG front end: band-pass + notch (FIR) and per-window normalization.

Lives inside the model so training, ``check_model`` and the Codabench harness all
see identical inputs. All filtering is one depthwise ``conv1d`` with a symmetric
(linear-phase) FIR kernel designed once with scipy at construction time, so the
output has no group delay and nothing here is trained or stateful.

Input / output: ``(B, C, T)`` float, same shape.
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
    """Band-pass / notch filter and per-window, per-channel standardization.

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
        Per-window, per-channel zero mean / unit SD (what InstanceNorm1d does).
    clip : float or None
        Clamp |x| after normalization (in SD units) to guard against spikes.
    """

    def __init__(self, n_channels: int = 16, sfreq: float = 2000.0,
                 highpass: float | None = 20.0, lowpass: float | None = None,
                 notch: float | None = None, notch_width: float = 6.0,
                 numtaps: int = 2001, normalize: bool = True, clip: float | None = 20.0):
        super().__init__()
        self.n_channels = n_channels
        self.normalize = normalize
        self.clip = clip
        h = _design_fir(sfreq, highpass, lowpass, notch, notch_width, numtaps)
        if h is None:
            self.kernel = None
        else:
            # FIR taps (K,); not a parameter and not persisted: rebuilt from the
            # config at construction, so weights.pt stays architecture-only.
            self.register_buffer("kernel", torch.from_numpy(h), persistent=False)

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-delay 'same' convolution with the symmetric FIR, via FFT.

        FFT convolution is O(T log T) regardless of kernel length; a direct
        depthwise conv1d with a 2001-tap kernel is ~50x slower on CPU.
        """
        k = self.kernel.shape[-1]
        pad = k // 2
        mode = "reflect" if x.shape[-1] > pad else "replicate"
        xp = F.pad(x, (pad, pad), mode=mode)  # reflect pad: no edge transient
        n = xp.shape[-1] + k - 1
        X = torch.fft.rfft(xp.float(), n=n)
        H = torch.fft.rfft(self.kernel.float(), n=n)
        y = torch.fft.irfft(X * H, n=n)
        # full convolution -> keep the central T samples (linear phase => no delay)
        y = y[..., k - 1: k - 1 + x.shape[-1]]
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        if self.kernel is not None:
            x = self._filter(x)
        if self.normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / (std + 1e-5)
        if self.clip is not None:
            x = x.clamp(-self.clip, self.clip)
        return x
