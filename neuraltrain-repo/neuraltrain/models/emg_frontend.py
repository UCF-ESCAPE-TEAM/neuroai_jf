"""Parameter-free EMG front end: band-pass + notch (one scipy FIR) + per-window scaling.

Drop it in as the first layer of the model (``self.frontend = EmgPreproc(...)``)
so train and test see exactly the same preprocessing and it ships inside
``submission.py`` with nothing to install: only torch + scipy (scipy comes
with scikit-learn / mne in the Codabench worker image).

    EMG (B, C, T) --> [bad-channel rejection] --> FIR band-pass+notch (FFT conv)
                  --> scaling --> [clip] --> (B, C, T)

The FIR is designed ONCE in ``__init__`` with ``scipy.signal.firwin`` from the
training ``sfreq`` and kept as a non-persistent buffer, so ``weights.pt`` stays
architecture-only and the filter is rebuilt from the config at load time.

Scaling ("norm") options, all computed per window (never across the batch):
    "maxabs"  x / max|x|                 -> [-1, 1], keeps zero at zero  (recommended)
    "minmax"  (x - min) / (max - min)    -> [0, 1]   (or [-1, 1] with signed=True)
    "robust"  like minmax but with the q / 1-q quantiles instead of min / max,
              then clamped: one artifact sample no longer sets the scale
    "zscore"  (x - mean) / std
    "none"
``scope="channel"`` scales each channel on its own (removes electrode-gain and
impedance differences between users); ``scope="window"`` uses one scale for all
16 channels of the window (keeps the relative amplitude pattern across the ring).
``scope="batch"`` exists only for experiments: it makes a prediction depend on
which other windows happen to be in the batch, which the server does not
guarantee to be reproducible (batch size 64 during scoring, any size allowed).

``causal=True`` (the model's online mode): the FIR is applied with left-only
padding (group delay ``(taps - 1) / 2`` samples) and the scaling uses
expanding-window statistics (running max / min / mean / std up to ``t``), so
the output at ``t`` never depends on samples after ``t``. ``robust`` has no
cheap running form and is refused in causal mode.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn


def design_fir(sfreq: float, l_freq: float | None, h_freq: float | None,
               notch_freqs: tuple[float, ...] | None = None, notch_width: float = 6.0,
               bp_taps: int | None = None, notch_taps: int | None = None,
               window: str = "hamming") -> np.ndarray | None:
    """Band-pass + notch as ONE symmetric (zero-phase) FIR, designed with scipy.

    Tap counts follow MNE's rule of thumb for a Hamming window
    (``3.3 * sfreq / transition_bandwidth``, rounded to odd); pass ``bp_taps`` /
    ``notch_taps`` to override. The cascade is obtained by convolving the kernels.
    """
    nyq = sfreq / 2.0

    def odd(n: float) -> int:
        n = int(np.ceil(n))
        return n if n % 2 else n + 1

    h = None
    if l_freq is not None or h_freq is not None:
        lo = l_freq if l_freq is not None else 0.0
        hi = h_freq if h_freq is not None else nyq
        trans = min(max(lo * 0.25, 2.0), max(0.25 * (nyq - hi), 2.0))  # MNE-like auto transition
        n = bp_taps or odd(3.3 * sfreq / trans)
        if lo > 0 and hi < nyq:
            h = signal.firwin(n, [lo, hi], pass_zero=False, window=window, fs=sfreq)
        elif lo > 0:
            h = signal.firwin(n, lo, pass_zero=False, window=window, fs=sfreq)
        else:
            h = signal.firwin(n, hi, pass_zero=True, window=window, fs=sfreq)
    for f0 in (notch_freqs or ()):
        if f0 + notch_width / 2 >= nyq:
            continue
        n = notch_taps or odd(3.3 * sfreq / (notch_width / 2))
        hn = signal.firwin(n, [f0 - notch_width / 2, f0 + notch_width / 2],
                           pass_zero=True, window=window, fs=sfreq)  # band-stop
        h = hn if h is None else np.convolve(h, hn)
    return None if h is None else h.astype(np.float32)


class EmgPreproc(nn.Module):
    NORMS = ("maxabs", "minmax", "robust", "zscore", "none")
    SCOPES = ("channel", "window", "batch")

    def __init__(self, n_channels: int = 16, sfreq: float = 2000.0,
                 l_freq: float | None = 20.0, h_freq: float | None = 450.0,
                 notch_freqs: tuple[float, ...] | None = (60.0, 120.0, 180.0),
                 notch_width: float = 6.0, bp_taps: int | None = None,
                 notch_taps: int | None = None,
                 norm: str = "maxabs", scope: str = "channel", signed: bool = True,
                 robust_q: float = 0.005, eps: float = 1e-6, clip: float | None = None,
                 causal: bool = False,
                 reject_channels: bool = False, reject_rms_nstd: float = 3.0,
                 reject_mains: float | None = 60.0, reject_mains_nstd: float = 3.0,
                 reject_mains_bw: float = 2.0, reject_segment: int = 400,
                 reject_max_frac: float = 0.5, reject_method: str = "std",
                 kernel: np.ndarray | None = None):
        super().__init__()
        if norm not in self.NORMS:
            raise ValueError(f"norm must be maxabs|minmax|robust|zscore|none, got {norm!r}")
        if scope not in self.SCOPES:
            raise ValueError(f"scope must be channel|window|batch, got {scope!r}")
        if causal and norm == "robust":
            raise ValueError("norm='robust' has no running form; use maxabs/minmax/zscore when causal")
        if causal and scope == "batch":
            raise ValueError("scope='batch' is not causal")
        if reject_method not in ("std", "mad"):
            raise ValueError(f"reject_method must be 'std' or 'mad', got {reject_method!r}")
        self.n_channels, self.sfreq = n_channels, float(sfreq)
        self.norm, self.scope, self.signed = norm, scope, signed
        self.robust_q, self.eps, self.clip = robust_q, eps, clip
        self.causal = causal
        self.reject_channels = reject_channels
        self.reject_rms_nstd, self.reject_mains = reject_rms_nstd, reject_mains
        self.reject_mains_nstd, self.reject_mains_bw = reject_mains_nstd, reject_mains_bw
        self.reject_segment, self.reject_max_frac = reject_segment, reject_max_frac
        self.reject_method = reject_method
        self.last_reject_fraction: float = 0.0
        if kernel is None:
            kernel = design_fir(self.sfreq, l_freq, h_freq, notch_freqs, notch_width, bp_taps, notch_taps)
        if kernel is None:
            self.kernel = None
        else:
            # not a parameter, not persisted: rebuilt from the config, never from weights.pt
            self.register_buffer("kernel", torch.as_tensor(np.asarray(kernel, dtype=np.float32)),
                                 persistent=False)
        self.n_taps = 0 if kernel is None else int(len(kernel))

    # ------------------------------------------------------------ rejection
    def _outliers(self, v: torch.Tensor, n: float) -> torch.Tensor:
        """Outliers across channels, per window. (B, C) -> bool (B, C).

        "std": v > mean + n*std.  "mad": v > median + n*1.4826*MAD (robust: two
        loud channels cannot hide each other; use n ~ 5).
        """
        if self.reject_method == "mad":
            med = v.median(-1, keepdim=True).values
            mad = (v - med).abs().median(-1, keepdim=True).values
            return v > med + n * 1.4826 * mad
        return v > v.mean(-1, keepdim=True) + n * v.std(-1, keepdim=True, unbiased=False)

    def bad_channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Bool mask (B, C): channels whose RMS or mains-band energy (±reject_mains_bw
        Hz around reject_mains and its 2nd harmonic) is an outlier in that window.
        Causal: statistics from the first ``reject_segment`` samples. Windows with
        more than ``reject_max_frac`` of the channels flagged are left untouched."""
        seg = (x[..., : self.reject_segment] if self.causal else x).float()
        bad = self._outliers(seg.pow(2).mean(-1).sqrt(), self.reject_rms_nstd)
        if self.reject_mains is not None:
            spec = torch.fft.rfft(seg, dim=-1).abs()
            freqs = torch.fft.rfftfreq(seg.shape[-1], d=1.0 / self.sfreq).to(seg.device)
            band = torch.zeros_like(freqs, dtype=torch.bool)
            for harmonic in (1, 2):
                f0 = self.reject_mains * harmonic
                if f0 < self.sfreq / 2:
                    band |= (freqs - f0).abs() <= self.reject_mains_bw
            bad |= self._outliers(spec[..., band].sum(-1), self.reject_mains_nstd)
        too_many = bad.float().mean(-1, keepdim=True) > self.reject_max_frac
        return bad & ~too_many

    def reject(self, x: torch.Tensor) -> torch.Tensor:
        """Replace flagged channels by the mean of their two ring neighbours (circular)."""
        bad = self.bad_channel_mask(x)
        self.last_reject_fraction = bad.float().mean().item()
        if not bool(bad.any()):
            return x
        neighbours = (x.roll(1, dims=1) + x.roll(-1, dims=1)) / 2
        return torch.where(bad.unsqueeze(-1), neighbours, x)

    # ------------------------------------------------------------ filtering
    @staticmethod
    def _pad_reflect(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
        """Odd reflection about the end samples (MNE 'reflect_limited'), zeros beyond."""
        T = x.shape[-1]
        ml, mr = min(left, T - 1), min(right, T - 1)
        parts = []
        if left > ml:
            parts.append(x.new_zeros(x.shape[:-1] + (left - ml,)))
        if ml:
            parts.append(2 * x[..., :1] - x[..., 1:ml + 1].flip(-1))
        parts.append(x)
        if mr:
            parts.append(2 * x[..., -1:] - x[..., -mr - 1:-1].flip(-1))
        if right > mr:
            parts.append(x.new_zeros(x.shape[:-1] + (right - mr,)))
        return torch.cat(parts, dim=-1)

    def filter(self, x: torch.Tensor) -> torch.Tensor:
        """'Same'-length FIR via FFT: O(T log T) whatever the tap count.

        Centred / zero phase (output t uses x[t-k//2 .. t+k//2]) or, when causal,
        left-aligned (output t uses x[t-k+1 .. t]).
        """
        k = self.n_taps
        T = x.shape[-1]
        left, right = (k - 1, 0) if self.causal else (k // 2, k // 2)
        xp = self._pad_reflect(x.float(), left, right)
        n = xp.shape[-1] + k - 1
        y = torch.fft.irfft(torch.fft.rfft(xp, n=n) * torch.fft.rfft(self.kernel, n=n), n=n)
        return y[..., k - 1: k - 1 + T].to(x.dtype)

    # ---------------------------------------------------------- normalising
    def _reduce_dims(self) -> tuple[int, ...]:
        return {"channel": (-1,), "window": (-2, -1), "batch": (0, -2, -1)}[self.scope]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "none":
            return x
        if self.causal:
            return self._normalize_causal(x)
        dims = self._reduce_dims()
        if self.norm == "zscore":
            mean = x.mean(dims, keepdim=True)
            std = x.std(dims, keepdim=True, unbiased=False)
            return (x - mean) / (std + self.eps)
        if self.norm == "maxabs":
            return x / (x.abs().amax(dims, keepdim=True) + self.eps)
        if self.norm == "minmax":
            lo, hi = x.amin(dims, keepdim=True), x.amax(dims, keepdim=True)
        else:  # robust: quantiles instead of extremes, via kthvalue (no size limit)
            B, C, _ = x.shape
            flat = {"channel": x, "window": x.reshape(B, 1, -1),
                    "batch": x.reshape(1, 1, -1)}[self.scope]
            n = flat.shape[-1]
            k_lo = max(1, int(round(self.robust_q * n)))
            k_hi = max(1, int(round((1 - self.robust_q) * n)))
            lo = flat.kthvalue(k_lo, dim=-1, keepdim=True).values   # (B,C,1) / (B,1,1) / (1,1,1)
            hi = flat.kthvalue(k_hi, dim=-1, keepdim=True).values
        y = (x - lo) / (hi - lo + self.eps)
        if self.norm == "robust":
            y = y.clamp(0.0, 1.0)
        return y * 2 - 1 if self.signed else y

    def _normalize_causal(self, x: torch.Tensor) -> torch.Tensor:
        """Same scalings with running statistics over x[.. t] (per channel, or over
        all channels of the window so far when scope='window')."""
        xd = x.double()
        if self.scope == "window":                       # running stats over (C, t)
            B, C, T = xd.shape
            flat = xd.transpose(1, 2).reshape(B, 1, T * C)   # time-major: all channels of t before t+1
            y = self._running(flat)
            return y.reshape(B, T, C).transpose(1, 2).to(x.dtype)
        return self._running(xd).to(x.dtype)

    def _running(self, xd: torch.Tensor) -> torch.Tensor:
        if self.norm == "zscore":
            n = torch.arange(1, xd.shape[-1] + 1, device=xd.device, dtype=torch.float64)
            mean = xd.cumsum(-1) / n
            var = (xd * xd).cumsum(-1) / n - mean * mean
            return (xd - mean) / (var.clamp_min(0).sqrt() + self.eps)
        if self.norm == "maxabs":
            return xd / (xd.abs().cummax(-1).values + self.eps)
        lo, hi = xd.cummin(-1).values, xd.cummax(-1).values   # minmax
        y = (xd - lo) / (hi - lo + self.eps)
        return y * 2 - 1 if self.signed else y

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        if self.reject_channels:
            x = self.reject(x)         # on the raw signal, before the filter
        if self.kernel is not None:
            x = self.filter(x)         # filter FIRST: removes the DC offset min-max would otherwise keep
        x = self.normalize(x)
        if self.clip is not None:
            x = x.clamp(-self.clip, self.clip)
        return x

    def extra_repr(self) -> str:
        return (f"sfreq={self.sfreq}, taps={self.n_taps}, norm={self.norm}, "
                f"scope={self.scope}, signed={self.signed}, causal={self.causal}, "
                f"reject={self.reject_channels}")


EmgFrontEnd = EmgPreproc   # name used by older code


# ---------------------------------------------------------------------------
# Building from a config dict, with support for checkpoints trained before this
# front end (raw / R0 / R1): their configs carry highpass / lowpass / notch /
# notch_width / fir_taps / normalize and used one multi-band firwin kernel plus
# a per-channel z-score (eps 1e-5). ``frontend_from_config`` rebuilds that
# exactly so those weights evaluate as trained.
# ---------------------------------------------------------------------------
FILTER_KEYS = ("l_freq", "h_freq", "notch_freqs", "notch_width", "bp_taps", "notch_taps")
SCALE_KEYS = ("norm", "scope", "signed", "robust_q", "eps", "clip")
COMMON_KEYS = ("sfreq", "causal", "reject_channels", "reject_rms_nstd", "reject_mains",
               "reject_mains_nstd", "reject_mains_bw", "reject_segment", "reject_max_frac",
               "reject_method")
LEGACY_KEYS = ("highpass", "lowpass", "notch", "fir_taps", "normalize")
FRONTEND_KEYS = FILTER_KEYS + SCALE_KEYS + COMMON_KEYS


def legacy_fir_kernel(sfreq: float, highpass: float | None, lowpass: float | None,
                      notch: float | None, notch_width: float, numtaps: int) -> np.ndarray | None:
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
    return signal.firwin(numtaps, edges, pass_zero=(lo <= 0), fs=sfreq).astype(np.float32)


def merge_config(base: dict, ckpt: dict | None) -> dict:
    """``base`` (submission CONFIG) updated by a checkpoint's config. A legacy
    checkpoint config replaces the new filter/scaling keys instead of mixing."""
    if not ckpt:
        return dict(base)
    merged = {**base, **ckpt}
    if "fir_taps" in ckpt:
        for k in FILTER_KEYS + SCALE_KEYS:
            if k not in ckpt:
                merged.pop(k, None)
    return merged


def frontend_from_config(n_channels: int, cfg: dict) -> EmgPreproc:
    common = {k: cfg[k] for k in COMMON_KEYS if k in cfg}
    if "fir_taps" in cfg:                      # legacy checkpoint
        warnings.warn("legacy front end (multi-band firwin + z-score) rebuilt from the checkpoint config")
        kernel = legacy_fir_kernel(cfg["sfreq"], cfg.get("highpass"), cfg.get("lowpass"),
                                   cfg.get("notch"), cfg.get("notch_width", 6.0), cfg["fir_taps"])
        return EmgPreproc(n_channels, l_freq=None, h_freq=None, notch_freqs=None, kernel=kernel,
                          norm="zscore" if cfg.get("normalize", True) else "none", scope="channel",
                          eps=1e-5, clip=cfg.get("clip"), **common)
    kw = {k: cfg[k] for k in FILTER_KEYS + SCALE_KEYS if k in cfg}
    if kw.get("notch_freqs") is not None:
        kw["notch_freqs"] = tuple(kw["notch_freqs"])
    return EmgPreproc(n_channels, **kw, **common)
