import pydantic
import pytest
import torch

from .base import BaseBrainModelConfig
from .st_tcn_pose import StTcnPoseModel
from .st_tcn_pose_cmixup import (
    StTcnPoseCMixup,
    StTcnPoseCMixupModel,
    cmixup_partners,
    window_label_summary,
)


class _Holder(pydantic.BaseModel):
    brain_model_config: BaseBrainModelConfig


def _small_cfg(**mix):
    return {
        "name": "StTcnPoseCMixup", "sfreq": 2000.0, "notch_freqs": None, "l_freq": 20.0,
        "h_freq": 450.0, "bp_taps": 65, "f1": 4, "depth_mult": 1, "f2": 8,
        "tcn_modules": 1, "tcn_blocks": 2, "tcn_filters": 8, **mix,
    }


def test_config_builds_subclass_with_mix_fields():
    cfg = _Holder(brain_model_config=_small_cfg(mix_sigma=0.3)).brain_model_config
    model = cfg.build(n_spatial_locations=16, n_outputs=20)
    assert isinstance(model, StTcnPoseCMixupModel)
    assert isinstance(model, StTcnPoseModel)
    assert model.mix_sigma == 0.3
    # the mix fields must not leak into the architecture / state dict
    assert not any("mix" in k for k in model.state_dict())


def test_window_label_summary_ignores_nan_frames():
    y = torch.zeros(3, 20, 50)
    y[0] = 1.0
    y[0, :, :25] = torch.nan           # half the frames unlabelled -> still 1.0
    y[1] = torch.nan                   # whole window unlabelled -> batch mean
    s = window_label_summary(y)
    assert s.shape == (3, 20)
    assert torch.allclose(s[0], torch.ones(20))
    assert torch.allclose(s[2], torch.zeros(20))
    assert torch.isfinite(s).all()


def test_cmixup_kernel_prefers_close_labels_and_other_subjects():
    torch.manual_seed(0)
    # four windows: 0 and 1 share a pose, 2 and 3 share a different pose (far away)
    labels = torch.tensor([[0.0], [0.05], [3.0], [3.05]])
    sid = torch.tensor([0, 1, 0, 1])
    perms = torch.stack([cmixup_partners(labels, sigma=0.2, subject_ids=sid) for _ in range(300)])
    # window 0 should almost always pair with window 1 (close pose, other subject)
    assert (perms[:, 0] == 1).float().mean() > 0.95
    assert (perms[:, 2] == 3).float().mean() > 0.95
    # same-subject weight 0: window 0 never pairs with window 2 (same subject)
    assert (perms[:, 0] == 2).sum() == 0
    # never with itself
    assert (perms == torch.arange(4)).sum() == 0


def test_cmixup_falls_back_when_no_other_subject():
    labels = torch.randn(5, 3)
    sid = torch.zeros(5, dtype=torch.long)       # single subject in the batch
    perm = cmixup_partners(labels, sigma=1.0, subject_ids=sid, same_subject_weight=0.0)
    assert perm.shape == (5,) and (perm != torch.arange(5)).all()


def _mixed_model(**mix):
    fields = _small_cfg(**mix)
    fields.pop("name")
    return StTcnPoseCMixup(**fields).build(n_spatial_locations=16, n_outputs=20)


def test_train_hook_mixes_targets_and_frontend_features_with_same_lambda():
    torch.manual_seed(1)
    model = _mixed_model(mix_alpha=2.0, mix_sigma=100.0, mix_where="frontend")
    model.train()
    x = torch.randn(6, 16, 400) * torch.tensor([1.0, 50.0, 1.0, 50.0, 1.0, 50.0]).view(-1, 1, 1)
    y = torch.randn(6, 20, 400)
    sid = torch.tensor([0, 1, 0, 1, 0, 1])
    x_out, y_out = model.train_batch_transform(x, y, sid)
    assert x_out is x                                      # input untouched here
    perm, lam = model._mix_state
    lam_y = lam.view(-1, 1, 1)
    assert torch.allclose(y_out, lam_y * y + (1 - lam_y) * y[perm])
    assert model.last_mix_stats["mix_cross_subject_frac"] == 1.0
    # forward must apply the SAME perm / lambda after the front end
    feats = model.frontend(x)
    expected = lam_y * feats + (1 - lam_y) * feats[perm]
    seen = {}
    h = model.spatial.register_forward_pre_hook(lambda m, inp: seen.__setitem__("x", inp[0]))
    out = model(x)
    h.remove()
    assert torch.allclose(seen["x"], expected, atol=1e-6)
    assert out.shape == (6, 400, 20)
    assert model._mix_state is None                        # consumed exactly once


def test_input_mode_mixes_raw_emg():
    torch.manual_seed(2)
    model = _mixed_model(mix_where="input", mix_sigma=100.0)
    model.train()
    x, y = torch.randn(4, 16, 300), torch.randn(4, 20, 300)
    x_out, y_out = model.train_batch_transform(x, y, torch.tensor([0, 1, 0, 1]))
    assert not torch.equal(x_out, x) and model._mix_state is None
    assert torch.isfinite(x_out).all() and torch.isfinite(y_out).all()


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_eval_and_prob_zero_never_mix(prob):
    model = _mixed_model(mix_prob=prob)
    x, y = torch.randn(4, 16, 300), torch.randn(4, 20, 300)
    model.eval()
    x_out, y_out = model.train_batch_transform(x, y, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(y_out, y) and model._mix_state is None
    model.train()
    x_out, y_out = model.train_batch_transform(x, y, torch.tensor([0, 1, 0, 1]))
    if prob == 0.0:
        assert torch.equal(y_out, y) and model._mix_state is None
    with torch.no_grad():
        model.eval()
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)                               # inference is deterministic


def test_nan_frames_propagate_so_the_module_masks_them():
    torch.manual_seed(3)
    model = _mixed_model(mix_sigma=100.0)
    model.train()
    y = torch.randn(4, 20, 100)
    y[0, :, :10] = torch.nan
    _, y_out = model.train_batch_transform(torch.randn(4, 16, 100), y, torch.tensor([0, 1, 0, 1]))
    assert y_out[0, :, :10].isnan().all()


def test_checkpoint_loads_into_plain_st_tcn_pose():
    cfg = _Holder(brain_model_config=_small_cfg()).brain_model_config
    mixed = cfg.build(16, 20)
    plain_cfg = dict(_small_cfg())
    plain_cfg["name"] = "StTcnPose"
    plain = _Holder(brain_model_config=plain_cfg).brain_model_config.build(16, 20)
    plain.load_state_dict(mixed.state_dict(), strict=True)
