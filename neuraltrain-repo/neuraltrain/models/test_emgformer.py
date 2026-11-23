"""Tests for EmgFormer. Run: python -m pytest neuraltrain/models/test_emgformer.py -q"""
import numpy as np
import pytest
import torch
from scipy import signal

from neuraltrain.models import emgformer as m

B, C, T, J = 2, 16, 10000, 20
SIZES = {
    "S": dict(n_layers=3, d_model=256, n_heads=4, d_ffn=512),
    "M": dict(n_layers=6, d_model=256, n_heads=8, d_ffn=1024),
    "L": dict(n_layers=8, d_model=384, n_heads=12, d_ffn=1536),
}


def _build(n_ch=C, n_out=J, **kw):
    torch.manual_seed(0)
    return m.EmgFormer(**kw).build(n_spatial_locations=n_ch, n_outputs=n_out).eval()


@pytest.mark.parametrize("size", ["S", "M", "L"])
@pytest.mark.parametrize("use_filter", [True, False])
def test_shape_finite_not_constant(size, use_filter):
    net = _build(filter=use_filter, **SIZES[size])
    with torch.no_grad():
        y = net(torch.randn(B, C, T) * 100)
    assert y.shape == (B, T, J)
    assert torch.isfinite(y).all()
    assert y.std(dim=1).min() > 0


def test_token_rate():
    net = _build(**SIZES["S"])
    assert net.featurizer.downsample == 50 and net.feature_rate == 40.0
    with torch.no_grad():
        assert net.featurizer(torch.randn(1, C, T)).shape == (1, 256, 200)


@pytest.mark.parametrize("size,paper_m", [("S", 3.5), ("M", 6.6), ("L", 16.3)])
def test_param_count_near_paper(size, paper_m):
    """Paper counts are for 8 channels and 22 angles; unspecified details leave a gap."""
    net = _build(n_ch=8, n_out=22, **SIZES[size])
    n = sum(p.numel() for p in net.parameters()) / 1e6
    assert abs(n - paper_m) / paper_m < 0.15, n


def test_filter_switch_adds_no_state():
    on, off = _build(filter=True, **SIZES["S"]), _build(filter=False, **SIZES["S"])
    assert on.state_dict().keys() == off.state_dict().keys()
    off.load_state_dict(on.state_dict(), strict=True)
    assert isinstance(off.filter, torch.nn.Identity)
    x = torch.randn(B, C, T)
    with torch.no_grad():
        assert not torch.allclose(on(x), off(x), atol=1e-4)     # the switch does something


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e5])
def test_amplitude_invariance(scale):
    net = _build(**SIZES["S"])
    x = torch.randn(B, C, T)
    with torch.no_grad():
        assert torch.allclose(net(x), net(x * scale), atol=1e-4)


def test_gain_matches_scipy_filtfilt_response():
    fs, order, width = 2000.0, 4, 2.0
    f = m.ZeroPhaseFilter(fs, 20.0, 450.0, 60.0, 450.0, width, order)
    assert f.notch_freqs == (60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0)
    freqs = np.linspace(1.0, 999.0, 4001)
    ref = np.ones_like(freqs)
    for sos in (signal.butter(order, 20.0, "highpass", fs=fs, output="sos"),
                signal.butter(order, 450.0, "lowpass", fs=fs, output="sos")):
        ref *= np.abs(signal.sosfreqz(sos, worN=freqs, fs=fs)[1]) ** 2
    for f0 in f.notch_freqs:
        b, a = signal.iirnotch(f0, f0 / width, fs=fs)
        ref *= np.abs(signal.freqz(b, a, worN=freqs, fs=fs)[1]) ** 2
    got = m.zero_phase_gain(torch.as_tensor(freqs), fs, 20.0, 450.0, f.notch_freqs, width, order).numpy()
    assert np.allclose(got, ref, atol=1e-5)


def test_filter_removes_mains_and_drift_keeps_emg_band():
    t = torch.arange(T) / 2000.0
    f = m.ZeroPhaseFilter(notch_freq=60.0)

    def rms_ratio(freq):
        x = torch.sin(2 * torch.pi * freq * t)[None, None]
        y = f(x)[..., 2000:-2000]
        return (y.pow(2).mean().sqrt() / x[..., 2000:-2000].pow(2).mean().sqrt()).item()

    assert rms_ratio(60.0) < 0.02 and rms_ratio(180.0) < 0.02       # mains harmonics
    assert rms_ratio(2.0) < 0.01 and rms_ratio(800.0) < 0.01        # drift, above the band
    assert 0.97 < rms_ratio(100.0) < 1.01 and 0.97 < rms_ratio(270.0) < 1.01
    assert f(torch.full((1, 1, T), 5.0)).abs().max() < 1e-3         # DC offset


def test_fifty_hz_mains_and_no_notch():
    assert m.ZeroPhaseFilter(notch_freq=50.0).notch_freqs == tuple(50.0 * k for k in range(1, 10))
    assert m.ZeroPhaseFilter(notch_freq=None).notch_freqs == ()


def test_short_window_and_gradients():
    net = _build(**SIZES["S"]).train()
    x = torch.randn(2, C, 4000, requires_grad=True)
    y = net(x)
    assert y.shape == (2, 4000, J)
    y.abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.parameters() if p.requires_grad)


def _envelope_sensitivity(kind):
    """How much the token energy follows a 5x amplitude step in the middle of the window."""
    torch.manual_seed(0)
    feat = m.TdsFeaturizer(C, 256, 8, (11, 5, 9, 3), (5, 2, 5, 1), (0, 0, 5, 3), 16, kind).eval()
    x = torch.randn(4, C, T)
    x[..., T // 2:] *= 5.0
    with torch.no_grad():
        z = feat(m.WindowNorm()(x))                              # (B, F, 200)
    quiet, loud = z[..., 20:80], z[..., 120:180]
    return ((loud.mean((0, 2)) - quiet.mean((0, 2))).abs() / z.std((0, 2))).mean().item()


def test_featurizer_keeps_the_envelope():
    """Per-time-step normalisation erases amplitude changes along the window; the default must not."""
    assert _envelope_sensitivity("time") > 3 * _envelope_sensitivity("step")
    assert _envelope_sensitivity("batch") > 3 * _envelope_sensitivity("step")


@pytest.mark.parametrize("kind", ["time", "batch", "step"])
def test_tds_norm_kinds(kind):
    net = _build(tds_norm=kind, **SIZES["S"])
    with torch.no_grad():
        assert net(torch.randn(B, C, T)).shape == (B, T, J)


def test_config_registered():
    from neuraltrain.models import BaseBrainModelConfig
    assert "EmgFormer" in {c.__name__ for c in BaseBrainModelConfig.__subclasses__()}
