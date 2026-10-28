import pydantic
import pytest
import torch

from .base import BaseLoss
from .hard_example import GhmRegressionLoss, OhemRegressionLoss, asl1


class _Holder(pydantic.BaseModel):
    loss: BaseLoss


@pytest.mark.parametrize(
    "cfg",
    [
        {"name": "GhmRegressionLoss", "mu": 0.05, "bins": 10},
        {"name": "OhemRegressionLoss", "keep_fraction": 0.5, "select": "element"},
    ],
)
def test_config_round_trip(cfg):
    built = _Holder(loss=cfg).loss.build()
    assert type(built).__name__ == cfg["name"]
    assert not built.state_dict()  # nothing to ship in a checkpoint


def _data(seed=0, n=2000, j=20):
    g = torch.Generator().manual_seed(seed)
    y_true = torch.zeros(n, j)
    err = torch.randn(n, j, generator=g) * 0.15
    err[: n // 20] *= 6  # a hard 5 % tail
    return (y_true + err).requires_grad_(True), y_true


def test_ghm_mean_weight_one_and_eval_plain():
    y_pred, y_true = _data()
    loss = GhmRegressionLoss(mu=0.1, bins=20, momentum=0.0)
    loss.train()
    out = loss(y_pred, y_true)
    assert set(out) == {"total", "ghm_w_top", "ghm_frac_top"}
    out["total"].backward()
    assert torch.isfinite(y_pred.grad).all()
    # momentum 0, normalize_by nonempty_bins: weights average exactly 1
    with torch.no_grad():
        d = y_pred - y_true
        g = d.abs() / torch.sqrt(d * d + loss.mu**2)
        idx = (g * loss.bins).long().clamp(0, loss.bins - 1)
        dens = torch.bincount(idx.reshape(-1), minlength=loss.bins).float() / d.numel()
        w = torch.where(dens > 0, 1 / (dens.clamp(min=1e-6) * (dens > 0).sum()), 0.0)
        assert torch.allclose(w[idx].mean(), torch.tensor(1.0), atol=1e-4)
    loss.eval()
    plain = loss(y_pred, y_true)["total"]
    assert torch.allclose(plain, asl1(y_pred - y_true, loss.mu).mean())


def test_ghm_ema_updates_only_in_train():
    y_pred, y_true = _data()
    loss = GhmRegressionLoss(momentum=0.75)
    assert not bool(loss.ema_initialized)
    loss.eval()
    loss(y_pred, y_true)
    assert not bool(loss.ema_initialized)
    loss.train()
    loss(y_pred, y_true)
    first = loss.ema_density.clone()
    assert torch.allclose(first.sum(), torch.tensor(1.0))
    loss(y_pred * 0.3, y_true)
    assert not torch.allclose(first, loss.ema_density)


@pytest.mark.parametrize("select", ["element", "frame"])
def test_ohem_focuses_on_hard_tail(select):
    y_pred, y_true = _data()
    n = y_pred.shape[0]
    loss = OhemRegressionLoss(keep_fraction=0.2, easy_weight=0.0, select=select)
    loss.train()
    out = loss(y_pred, y_true)
    out["total"].backward()
    grad = y_pred.grad.abs()
    assert grad[: n // 20].mean() > 3 * grad[n // 20 :].mean()
    if select == "frame":
        kept = (grad > 0).any(dim=-1)
        assert kept.sum() == round(0.2 * n)
        assert kept[: n // 20].all()
    loss.eval()
    assert torch.allclose(loss(y_pred, y_true)["total"], (y_pred - y_true).abs().mean())


def test_empty_input():
    for loss in (GhmRegressionLoss(), OhemRegressionLoss()):
        loss.train()
        z = torch.zeros(0, 20, requires_grad=True)
        out = loss(z, z.detach())["total"]
        assert out.requires_grad and out.item() == 0.0
