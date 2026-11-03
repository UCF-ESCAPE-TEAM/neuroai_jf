import pydantic
import pytest
import torch

from .base import BaseBrainModelConfig
from .st_pose_transformer import (
    MsSpatialBlock,
    SequenceEncoder,
    StPoseTransformerModel,
    attention_mask,
)
from .st_vel_transformer import StVelTransformerModel, VelocityDecoder


class _Holder(pydantic.BaseModel):
    brain_model_config: BaseBrainModelConfig


def _small(name, **over):
    cfg = {
        "name": name, "sfreq": 2000.0, "notch_freqs": None, "l_freq": 20.0, "h_freq": 450.0,
        "bp_taps": 65, "f1": 4, "ms_kernels": (7, 15), "depth_mult": 1, "f2": 8,
        "d_model": 16, "n_heads": 2, "n_layers": 1, "ff_mult": 2, "conv_kernel": 5,
        "channel_attention": True, "ca_hidden": 4, **over,
    }
    return _Holder(brain_model_config=cfg).brain_model_config.build(16, 20)


@pytest.mark.parametrize("name,cls", [("StPoseTransformer", StPoseTransformerModel),
                                      ("StVelTransformer", StVelTransformerModel)])
@pytest.mark.parametrize("causal", [False, True])
def test_shapes_and_causality(name, cls, causal):
    torch.manual_seed(0)
    model = _small(name, causal=causal).eval()
    assert isinstance(model, cls)
    assert isinstance(model.spatial, MsSpatialBlock)
    x = torch.randn(2, 16, 4000) * 100
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4000, 20) and torch.isfinite(y).all()
    assert (y.std(dim=1) > 0).all()
    if causal:
        if hasattr(model, "decoder"):           # undo the hold-the-position init so the
            with torch.no_grad():               # trajectory actually responds to the input
                model.decoder.vel.weight.normal_()
        xb = x.clone()
        xb[..., 2400:] += 1e3
        with torch.no_grad():
            d = (model(x) - model(xb)).abs().amax(dim=-1)
        assert d[:, :2300].max() < 1e-3        # nothing before the perturbation (FIR fp noise aside)
        assert d[:, 2400 + 200:].max() > 1e-3   # something after it


def test_state_dict_loads_across_causal_settings():
    a = _small("StVelTransformer", causal=False)
    b = _small("StVelTransformer", causal=True)
    sd = {k: v for k, v in a.state_dict().items() if not k.startswith("frontend.")}
    missing, unexpected = b.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("frontend.") for k in missing)


def test_encoder_mask_is_causal_and_windowed():
    m = attention_mask(5, causal=True, window=2, device="cpu")
    assert m.tolist() == [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0], [0, 0, 1, 1, 1]]
    assert attention_mask(5, causal=False, window=None, device="cpu") is None
    enc = SequenceEncoder(8, 16, 2, 2, 2, 0.0, 0.0, "conformer", 5, "rope", None, causal=True).eval()
    z = torch.randn(1, 8, 50)
    zb = z.clone()
    zb[..., 30:] += 10
    with torch.no_grad():
        d = (enc(z) - enc(zb)).abs().amax(dim=1)[0]
    assert d[:30].max() == 0 and d[30:].max() > 0


def test_velocity_decoder_limits():
    torch.manual_seed(0)
    z = torch.randn(2, 16, 40)
    dec = VelocityDecoder(16, 20, position_steps=5, vel_scale=0.02, fuse_gain=0.0)
    with torch.no_grad():
        dec.vel.weight.normal_()                        # undo the zero init so velocity is non-trivial
        pos, vel = dec.readouts(z)
        out = dec.integrate(pos, vel)
    assert torch.allclose(out[..., :5], pos[..., :5])
    expected = pos[..., 4:5] + torch.cumsum(vel[..., 5:] * 0.02, dim=-1)
    assert torch.allclose(out[..., 5:], expected)
    dec.fuse_gain = 1.0
    with torch.no_grad():
        assert torch.allclose(dec.integrate(pos, vel), pos)
    dec.fuse_gain = 0.3
    with torch.no_grad():
        fused = dec.integrate(pos, vel)
    # first-order IIR: p_t = 0.7 (p_{t-1} + s v_t) + 0.3 a_t
    p = pos[..., 4]
    for t in range(5, 40):
        p = 0.7 * (p + 0.02 * vel[..., t]) + 0.3 * pos[..., t]
    assert torch.allclose(fused[..., -1], p, atol=1e-5)


def test_velocity_decoder_gru_variant_runs():
    dec = VelocityDecoder(16, 20, gru_hidden=8)
    out = dec(torch.randn(2, 16, 30))
    assert out.shape == (2, 20, 30)
    assert dec.vel.weight.abs().sum() == 0            # hold-the-position init


def test_ms_kernels_must_divide_f1():
    with pytest.raises(ValueError):
        _small("StPoseTransformer", f1=5, ms_kernels=(7, 15))
