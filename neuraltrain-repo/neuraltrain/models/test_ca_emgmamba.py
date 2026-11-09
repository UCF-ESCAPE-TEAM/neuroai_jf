"""Tests for CaEmgMamba. Run: python -m pytest neuraltrain/models/test_ca_emgmamba.py -q"""
import pytest
import torch

from neuraltrain.models import ca_emgmamba as m

B, C, T, J = 2, 16, 10000, 20
SMALL = dict(stem_channels=(16, 24, 32), d_model=32, n_layers=2, d_state=4, scan_impl="torch")


def _build(**kw):
    torch.manual_seed(0)
    return m.CaEmgMamba(**{**SMALL, **kw}).build(n_spatial_locations=C, n_outputs=J).eval()


@pytest.mark.parametrize("head", ["pose", "vel"])
@pytest.mark.parametrize("ca_kind", ["se", "eca", "none"])
def test_shape_finite_not_constant(head, ca_kind):
    net = _build(head=head, ca_kind=ca_kind)
    with torch.no_grad():
        y = net(torch.randn(B, C, T) * 100)
    assert y.shape == (B, T, J)
    assert torch.isfinite(y).all()
    assert y.std(dim=1).min() > 0                      # varies over time for every joint


def test_frame_rate_and_default_token_count():
    net = _build()
    assert net.stem.downsample == 40 and net.feature_rate == 50.0
    with torch.no_grad():
        assert net.stem(torch.randn(1, C, T)).shape[-1] == 250


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e5])
def test_amplitude_invariance(scale):
    net = _build(head="vel")
    x = torch.randn(B, C, T)
    with torch.no_grad():
        assert torch.allclose(net(x), net(x * scale), atol=1e-4)


def test_dead_electrode_stays_finite():
    x = torch.randn(B, C, T)
    x[:, 3] = 0.0
    x[:, 5] = 7.0                                      # constant offset, also zero variance
    z = m.WindowNorm()(x)
    assert (z[:, 3] == 0).all() and torch.isfinite(z).all()
    with torch.no_grad():
        assert torch.isfinite(_build(head="vel")(x)).all()


@pytest.mark.parametrize("ca_kind", ["se", "eca"])
def test_gate_is_identity_at_init(ca_kind):
    gated, plain = _build(ca_kind=ca_kind), _build(ca_kind="none")
    plain.load_state_dict(gated.state_dict(), strict=False)
    x = torch.randn(B, C, T)
    with torch.no_grad():
        assert torch.allclose(gated(x), plain(x), atol=1e-6)
    ca = gated.stem.stages[0].ca
    assert torch.allclose(ca.gains(torch.randn(3, 16, 50)), torch.ones(3, 16))


def test_ca_stages_selects_stages():
    net = _build(ca_stages=(1,))
    kinds = [type(s.ca).__name__ for s in net.stem.stages]
    assert kinds == ["ChannelAttention", "Identity", "Identity"]
    with pytest.raises(ValueError):
        _build(ca_stages=(4,))


def test_eca_adaptive_kernel():
    assert [m.eca_kernel_size(c) for c in (16, 64, 128, 192)] == [3, 3, 5, 5]


def test_scan_matches_closed_form():
    """y_t = sum_{j<=t} <C_t, exp(A * sum_{k=j+1..t} delta_k) * delta_j * B_j> u_j + D u_t."""
    torch.manual_seed(1)
    N, L, d, S = 2, 7, 3, 4
    u, delta = torch.randn(N, L, d), torch.rand(N, L, d) * 0.5
    A, Bm, Cm, D = -torch.rand(d, S) * 3, torch.randn(N, L, S), torch.randn(N, L, S), torch.randn(d)
    y = m.selective_scan_torch(u, delta, A, Bm, Cm, D)
    ref = torch.zeros_like(y)
    for t in range(L):
        for j in range(t + 1):
            acc = delta[:, j + 1:t + 1].sum(1)                               # (N, d)
            w = torch.exp(acc[..., None] * A) * delta[:, j, :, None] * Bm[:, j, None, :]
            ref[:, t] += (w * Cm[:, t, None, :]).sum(-1) * u[:, j]
        ref[:, t] += D * u[:, t]
    assert torch.allclose(y, ref, atol=1e-5)


def test_forward_block_is_causal_and_bidirectional_is_not():
    blk = m.MambaBlock(16, d_state=4, scan_impl="torch").eval()
    x = torch.randn(1, 30, 16)
    x2 = x.clone()
    x2[:, 20:] += torch.randn(1, 10, 16)               # not a constant: LayerNorm would cancel it
    with torch.no_grad():
        assert torch.allclose(blk(x)[:, :20], blk(x2)[:, :20], atol=1e-6)
        layer = m.MambaLayer(16, 4, 4, 2, 0, True, 0.0, "torch").eval()
        assert not torch.allclose(layer(x)[:, :20], layer(x2)[:, :20], atol=1e-6)


def test_velocity_head_limits():
    torch.manual_seed(0)
    z = torch.randn(2, 8, 40)
    pose_like = m.VelocityHead(8, 5, position_steps=4, fuse_gain=1.0)
    torch.nn.init.normal_(pose_like.vel.weight)
    assert torch.allclose(pose_like(z), pose_like.pos(z), atol=1e-6)        # g = 1 -> position readout
    integ = m.VelocityHead(8, 5, position_steps=4, vel_scale=0.02, fuse_gain=0.0)
    torch.nn.init.normal_(integ.vel.weight)
    y, pos, vel = integ(z), integ.pos(z), integ.vel(z)
    assert torch.allclose(y[..., :4], pos[..., :4])
    assert torch.allclose(y[..., 4:] - y[..., 3:-1], 0.02 * vel[..., 4:], atol=1e-5)
    hold = m.VelocityHead(8, 5, position_steps=4, fuse_gain=0.0)            # zero-init velocity
    assert torch.allclose(hold(z)[..., 4:], hold.pos(z)[..., 3:4].expand(-1, -1, 36), atol=1e-6)


@pytest.mark.parametrize("head", ["pose", "vel"])
def test_state_dict_round_trip_and_grads(head):
    a, b = _build(head=head), _build(head=head)
    for p in b.parameters():
        torch.nn.init.normal_(p)
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(B, C, 2000)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))
    a.train()
    a(x).abs().mean().backward()
    missing = [n for n, p in a.named_parameters() if p.grad is None]
    assert not missing, missing


def test_scan_checkpoint_same_output_and_grads():
    x = torch.randn(4, 30, 16)
    torch.manual_seed(0)
    plain = m.MambaBlock(16, d_state=4, scan_impl="torch", scan_checkpoint=False).train()
    torch.manual_seed(0)
    ckpt = m.MambaBlock(16, d_state=4, scan_impl="torch", scan_checkpoint=True).train()
    y1, y2 = plain(x), ckpt(x)
    assert torch.allclose(y1, y2)
    y1.pow(2).mean().backward()
    y2.pow(2).mean().backward()
    for (name, p1), p2 in zip(plain.named_parameters(), ckpt.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-7), name


def test_variable_window_length():
    net = _build()
    with torch.no_grad():
        for n in (2000, 4000, 12000):
            assert net(torch.randn(1, C, n)).shape == (1, n, J)


def test_fast_scan_parity():
    """Only runs where mamba_ssm + CUDA are available (the training cluster)."""
    if m._fast_scan is None or not torch.cuda.is_available():
        pytest.skip("mamba_ssm CUDA kernel not available")
    torch.manual_seed(0)
    blk = m.MambaBlock(64, scan_impl="auto").cuda().eval()
    x = torch.randn(4, 250, 64, device="cuda")
    with torch.no_grad():
        fast = blk(x)
        blk.scan_impl = "torch"
        slow = blk(x)
    assert torch.allclose(fast, slow, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Time-stretch augmentation (R4)
# ---------------------------------------------------------------------------
def test_time_stretch_factor_one_is_identity():
    ts = m.TimeStretch(prob=1.0, lo=1.0, hi=1.0)
    x, y = torch.randn(3, 16, 2000), torch.randn(3, 20, 2000)
    y[0, :, 100:120] = torch.nan
    xs, ys = ts(x, y)
    assert torch.allclose(xs, x, atol=1e-5)
    assert torch.equal(torch.isnan(ys), torch.isnan(y)) and torch.allclose(ys[~ys.isnan()], y[~y.isnan()], atol=1e-5)
    assert ts.last_stats["ts_frac_masked"] == 0.0


def test_time_stretch_slows_and_speeds_consistently():
    """A ramp target r(t) = t stretched by f about the centre becomes c + (t-c) f."""
    T = 2001
    ramp = torch.arange(T, dtype=torch.float32)
    for f in (0.8, 1.25):
        ts = m.TimeStretch(prob=1.0, lo=f, hi=f)
        x = ramp.expand(1, 2, T).clone()
        y = ramp.expand(1, 3, T).clone()
        xs, ys = ts(x, y)
        c = (T - 1) / 2
        expect = c + (ramp - c) * f
        inside = (expect >= 0) & (expect <= T - 1)
        assert torch.allclose(ys[0, 0][inside], expect[inside], atol=1e-2)
        assert ys[0, 0][~inside].isnan().all()                   # out-of-window frames masked
        assert torch.allclose(xs[0, 0][inside], expect[inside], atol=1e-2)
        assert torch.isfinite(xs).all()                          # EMG edge-held, never NaN
        if f > 1:
            assert ts.last_stats["ts_frac_masked"] > 0.0
        else:
            assert ts.last_stats["ts_frac_masked"] == 0.0


def test_time_stretch_prob_zero_disables_hook():
    off = _build(time_stretch_prob=0.0)
    assert off.time_stretch is None and not hasattr(off, "train_batch_transform")
    on = _build(time_stretch_prob=1.0, time_stretch_range=(0.9, 1.1)).train()
    x, y = torch.randn(2, C, 2000), torch.randn(2, J, 2000)
    xs, ys = on.train_batch_transform(x, y, None)
    assert xs.shape == x.shape and ys.shape == y.shape
    assert set(on.last_mix_stats) == {"ts_factor_mean", "ts_frac_masked"}
    on.eval()
    xe, ye = on.train_batch_transform(x, y, None)
    assert torch.equal(xe, x) and torch.equal(ye, y)             # inert outside training


def test_time_stretch_fields_do_not_change_state_dict():
    a, b = _build(time_stretch_prob=0.0), _build(time_stretch_prob=1.0)
    assert list(a.state_dict()) == list(b.state_dict())           # no parameters, old checkpoints load
