"""Tests for CaEmgMambaCMixup. Run: python -m pytest neuraltrain/models/test_ca_emgmamba_cmixup.py -q"""
import torch

from neuraltrain.models import ca_emgmamba as base
from neuraltrain.models import ca_emgmamba_cmixup as m

B, C, T, J = 4, 16, 2000, 20
SMALL = dict(stem_channels=(16, 24, 32), d_model=32, n_layers=2, d_state=4, scan_impl="torch", head="vel")


def _same(a, b):
    return torch.equal(a, b) or torch.equal(a.nan_to_num(-9.0), b.nan_to_num(-9.0))


def _build(**kw):
    torch.manual_seed(0)
    return m.CaEmgMambaCMixup(**{**SMALL, **kw}).build(n_spatial_locations=C, n_outputs=J)


def _batch():
    torch.manual_seed(1)
    x = torch.randn(B, C, T)
    y = torch.randn(B, J, 1).expand(B, J, T).clone() * 0.3    # one pose per window
    y[0, :, :50] = torch.nan
    sid = torch.tensor([0, 0, 1, 1])
    return x, y, sid


def test_targets_are_convex_mix_and_inputs_follow_in_forward():
    net = _build().train()
    x, y, sid = _batch()
    x2, y2 = net.train_batch_transform(x, y, sid)
    assert torch.equal(x2, x)                                   # input mixed later, after the norm
    assert net.last_mix_stats["mix_applied"] == 1.0
    assert net.last_mix_stats["mix_cross_subject_frac"] == 1.0  # same_subject_weight 0 -> all cross-user
    perm, lam = net._mix_state
    ref = lam.view(-1, 1, 1) * y + (1 - lam.view(-1, 1, 1)) * y[perm]
    assert torch.equal(torch.isnan(y2), torch.isnan(ref)) and torch.allclose(y2[~y2.isnan()], ref[~ref.isnan()])
    assert (perm != torch.arange(B)).all()
    # the pending state is consumed by exactly one forward
    z = net.norm(x)
    mixed = net._after_norm(z)
    assert torch.allclose(mixed, lam.view(-1, 1, 1) * z + (1 - lam.view(-1, 1, 1)) * z[perm])
    assert net._mix_state is None and torch.equal(net._after_norm(z), z)


def test_eval_is_identical_to_base_model_and_state_dict_matches():
    mix = _build().eval()
    torch.manual_seed(0)
    plain = base.CaEmgMamba(**SMALL).build(C, J).eval()
    assert list(mix.state_dict()) == list(plain.state_dict())    # no extra parameters
    plain.load_state_dict(mix.state_dict(), strict=True)
    x, y, sid = _batch()
    x2, y2 = mix.train_batch_transform(x, y, sid)
    assert torch.equal(x2, x) and _same(y2, y) and mix.last_mix_stats["mix_applied"] == 0.0
    with torch.no_grad():
        assert torch.allclose(mix(x), plain(x))


def test_mix_where_input_and_prob_zero():
    net = _build(mix_where="input").train()
    x, y, sid = _batch()
    x2, _ = net.train_batch_transform(x, y, sid)
    assert not torch.equal(x2, x) and net._mix_state is None
    off = _build(mix_prob=0.0).train()
    x3, y3 = off.train_batch_transform(x, y, sid)
    assert torch.equal(x3, x) and _same(y3, y)


def test_composes_with_time_stretch():
    net = _build(time_stretch_prob=1.0, time_stretch_range=(1.2, 1.2)).train()
    x, y, sid = _batch()
    _, y2 = net.train_batch_transform(x, y, sid)
    s = net.last_mix_stats
    assert {"ts_factor_mean", "ts_frac_masked", "mix_applied", "mix_lambda_mean"} <= set(s)
    assert s["ts_frac_masked"] > 0 and y2.isnan().any()


def test_training_step_backward():
    net = _build().train()
    x, y, sid = _batch()
    _, y2 = net.train_batch_transform(x, y, sid)
    pred = net(x).transpose(1, 2)                               # (B, J, T)
    valid = ~y2.isnan()
    (pred[valid] - y2[valid]).abs().mean().backward()
    assert all(p.grad is not None for p in net.parameters())
