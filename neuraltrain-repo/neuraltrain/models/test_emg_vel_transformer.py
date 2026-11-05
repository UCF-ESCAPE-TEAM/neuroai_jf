import pydantic
import pytest
import torch

from .base import BaseBrainModelConfig
from .emg_vel_transformer import EmgVelTransformerModel, RawChannelGate, eca_kernel_size


class _Holder(pydantic.BaseModel):
    brain_model_config: BaseBrainModelConfig


def _small(name="EmgVelTransformer", **over):
    cfg = {
        "name": name, "sfreq": 2000.0, "notch_freqs": None, "l_freq": 20.0, "h_freq": 450.0,
        "bp_taps": 65, "f1": 4, "ms_kernels": (7, 15), "depth_mult": 1, "f2": 8,
        "d_model": 16, "n_heads": 2, "n_layers": 1, "ff_mult": 2, "conv_kernel": 5,
        "channel_attention": True, "ca_hidden": 4, "gate_n_feat": 3, "gate_k": 15, "gate_stride": 4,
        **over,
    }
    return _Holder(brain_model_config=cfg).brain_model_config.build(16, 20)


@pytest.mark.parametrize("excite", ["se", "fc", "eca", "eca_multi"])
@pytest.mark.parametrize("causal", [False, True])
def test_shapes_and_causality(excite, causal):
    torch.manual_seed(0)
    model = _small(gate_excite=excite, causal=causal).eval()
    assert isinstance(model, EmgVelTransformerModel) and isinstance(model.gate, RawChannelGate)
    with torch.no_grad():                       # make the gate and the velocity head non-trivial
        for p in model.gate.parameters():
            p.normal_(std=0.5)
        model.decoder.vel.weight.normal_()
    x = torch.randn(2, 16, 4000) * 100
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4000, 20) and torch.isfinite(y).all() and (y.std(dim=1) > 0).all()
    if causal:
        xb = x.clone()
        xb[..., 2400:] += 1e3
        with torch.no_grad():
            d = (model(x) - model(xb)).abs().amax(dim=-1)
        assert d[:, :2300].max() < 1e-3
        assert d[:, 2600:].max() > 1e-3


def test_identity_at_init_matches_gate_off():
    """Zero-initialised excitation -> gain 1 -> same output as StVelTransformer without a gate."""
    torch.manual_seed(0)
    a = _small(channel_attention=True).eval()
    b = _small(channel_attention=False).eval()
    sd = {k: v for k, v in a.state_dict().items() if not k.startswith(("gate.", "frontend."))}
    b.load_state_dict(sd, strict=False)
    x = torch.randn(2, 16, 4000) * 100
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)
        g = a.gate._gains(a.gate._descriptor(a.frontend(x)))
    assert torch.allclose(g, torch.ones_like(g))


def test_gate_is_not_identity_after_perturbing_weights():
    torch.manual_seed(0)
    gate = RawChannelGate(16, n_feat=3, k=15, stride=4, excite="eca").eval()
    with torch.no_grad():
        gate.excite.weight.normal_()
    x = torch.randn(2, 16, 1000)
    g = gate._gains(gate._descriptor(x))
    assert g.shape == (2, 16) and (g > 0).all() and (g < 2).all()
    assert not torch.allclose(g, torch.ones_like(g))


def test_state_dict_loads_across_causal_settings():
    a = _small(causal=False)
    b = _small(causal=True)
    sd = {k: v for k, v in a.state_dict().items() if not k.startswith("frontend.")}
    missing, unexpected = b.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("frontend.") for k in missing)


def test_bad_args():
    with pytest.raises(ValueError):
        RawChannelGate(16, excite="softmax")
    with pytest.raises(ValueError):
        RawChannelGate(16, excite="eca", eca_k=4)


def test_eca_is_faithful_to_eq9():
    """ECA: one scalar per electrode, k weights, no bias, circular over the ring."""
    gate = RawChannelGate(16, n_feat=3, k=15, stride=4, excite="eca", eca_k=0)
    assert gate.eca_k == 3 and eca_kernel_size(16) == 3 and eca_kernel_size(64) == 3 and eca_kernel_size(256) == 5
    assert sum(p.numel() for p in gate.excite.parameters()) == 3
    assert gate.excite.bias is None
    with torch.no_grad():
        gate.excite.weight.copy_(torch.tensor([[[0.0, 0.0, 1.0]]]))     # gain_i depends on y_{i+1} only
    d = torch.zeros(1, 16, 3)
    d[0, 0] = 5.0                                                        # electrode 0 is loud
    g = gate._gains(d)
    assert g[0, 15] > 1.0 and (g[0, 1:15] == 1.0).all() and g[0, 0] == 1.0   # ring: 15 sees 0


def test_param_counts_follow_the_paper_table():
    C, f = 16, 3
    n = lambda ex, **kw: sum(p.numel() for p in RawChannelGate(C, n_feat=f, excite=ex, hidden=4, **kw).excite.parameters())
    assert n("fc") == C * f * C + C
    assert n("eca", eca_k=3) == 3
    assert n("eca_multi", eca_k=3) == 3 * f + 1
    assert n("se") == C * f * 4 + 4 + 4 * C + C
