"""Numerical companion to the hard-example-mining part of
tutorial/training_tutorial.md.  Run: python tutorial/hem_demo.py

Part 1  ASL1 and its gradient norm g = |d| / sqrt(d^2 + mu^2): why mu must sit
        near the model's typical error (the "g saturates at 1" failure).
Part 2  GHM-R on a synthetic error distribution: the per-bin weights, mean
        weight 1, and what happens to the 5 % hard tail and to outliers.
Part 3  OHEM on the same errors: which frames survive, and how much of the
        gradient a few label-noise frames would take.
Part 4  C-Mixup + GHM in one training step (the CM5 run of the ablation table).
"""

import torch

from neuraltrain.losses.hard_example import (
    GhmRegressionLoss,
    OhemRegressionLoss,
    asl1,
    asl1_gradient_norm,
)

torch.manual_seed(0)

# ----------------------------------------------------------------------------
print("\nPart 1 -- ASL1 gradient norm vs error, for three choices of mu (rad)")
d = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
print("  |error| rad :", " ".join(f"{v:6.2f}" for v in d))
for mu in (0.02, 0.1, 0.3):
    g = asl1_gradient_norm(d, mu)
    print(f"  g (mu={mu:<4}):", " ".join(f"{v:6.2f}" for v in g))
print("  -> with mu = 0.02 every error above ~0.1 rad has g ~ 1: they all land in the\n"
      "     last bin and GHM cannot tell a 0.1 rad miss from a 1 rad one.\n"
      "     mu = 0.1 (R1's MAE is 0.29 rad; the median error is lower) spreads them out.")

# ----------------------------------------------------------------------------
print("\nPart 2 -- GHM-R weights on 10,000 (frame, joint) errors")
n, j = 500, 20
err = torch.randn(n, j) * 0.15                      # typical errors, ~0.12 rad median
err[: n // 20] *= 6                                 # a hard 5 % tail (~0.7 rad)
err[0, :] = 3.0                                     # one frame of label noise (3 rad = 172 deg)
y_true = torch.zeros(n, j)
y_pred = (y_true + err).requires_grad_(True)

probe = torch.tensor([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 3.0])
print("  |error| rad  :", " ".join(f"{v:5.2f}" for v in probe))
for mu in (0.1, 0.3):
    ghm = GhmRegressionLoss(mu=mu, bins=30, momentum=0.0).train()
    yp = (y_true + err).requires_grad_(True)
    out = ghm(yp, y_true)
    out["total"].backward()
    # Recover the per-element weight GHM applied: grad = w * dASL1/dd / numel
    w = yp.grad.detach().abs() * err.numel() / asl1_gradient_norm(err, mu).clamp(min=1e-9)
    # weight as a function of |error|: nearest element to each probe value
    idx = (err.abs().reshape(-1)[None, :] - probe[:, None]).abs().argmin(dim=1)
    print(f"  weight mu={mu:<4}:", " ".join(f"{v:5.2f}" for v in w.reshape(-1)[idx]),
          f"  mean {w.mean().item():.2f}, frac_top {out['ghm_frac_top'].item():.2f}")
print("  -> mean weight is 1 either way, but WHICH errors count depends on mu.\n"
      "     mu = 0.1: the crowded 0.2-0.3 rad region (the bulk of a 0.15-std error\n"
      "     cloud) is halved, the rare tiny errors are boosted, and everything above\n"
      "     ~0.4 rad shares ONE saturated bin with the whole 5 % tail, so the tail\n"
      "     and the 3 rad outlier all get the same 0.72.\n"
      "     mu = 0.3: the tail spreads over several sparse bins and is boosted 4-9x,\n"
      "     the bulk drops to ~0.7. That is the 'focus on hard examples' regime --\n"
      "     and the regime in which a label-noise frame is boosted along with them.\n"
      "     frac_top is the gauge: high -> raise mu. GHM's weight is bounded by\n"
      "     1/density (never by the error size as in L2), which is its outlier guard.")

# ----------------------------------------------------------------------------
print("\nPart 3 -- OHEM (keep_fraction 0.3, select=frame) on the same errors")
for easy_w in (0.0, 0.1):
    ohem = OhemRegressionLoss(keep_fraction=0.3, easy_weight=easy_w, select="frame", base="l1").train()
    yp = (y_true + err).requires_grad_(True)
    o = ohem(yp, y_true)
    o["total"].backward()
    g = yp.grad.abs()
    share_noise = g[0].sum() / g.sum()
    frames_kept = (g.sum(dim=1) > g.sum(dim=1).max() * 0.5).float().mean()
    print(f"  easy_weight={easy_w}: loss {o['total'].item():.3f}, hard-frame mean |err| "
          f"{o['ohem_hard_mean'].item():.3f}, frames at full weight {frames_kept:.0%}, "
          f"gradient share of the ONE label-noise frame {share_noise:.1%}")
print("  -> OHEM gives every selected frame the same weight regardless of how wrong it\n"
      "     is (L1), so a single 3 rad outlier takes ~1/150 of the gradient, same as any\n"
      "     other selected frame -- milder than GHM's up-weighting but the easy 70 % are\n"
      "     dropped entirely unless easy_weight > 0.")

# ----------------------------------------------------------------------------
print("\nPart 4 -- C-Mixup and GHM-R in one step (StTcnPoseCMixup + GhmRegressionLoss)")
from neuraltrain.models.st_tcn_pose_cmixup import StTcnPoseCMixup

model = StTcnPoseCMixup(notch_freqs=None, bp_taps=65, f1=4, depth_mult=1, f2=8,
                        tcn_modules=1, tcn_blocks=2, tcn_filters=8, mix_sigma=0.5).build(16, 20).train()
loss_fn = GhmRegressionLoss(mu=0.1).train()
emg = torch.randn(8, 16, 2000) * 50
target = torch.randn(8, 20, 2000) * 0.3
sid = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
emg, target_mixed = model.train_batch_transform(emg, target, sid)   # C-Mixup: targets
y_pred = model(emg)                                                 # C-Mixup: features
out = loss_fn(y_pred, target_mixed.transpose(1, 2))                 # GHM-R on mixed pairs
out["total"].backward()
print(f"  loss {out['total'].item():.3f}, ghm_frac_top {out['ghm_frac_top'].item():.2f}, "
      f"mix stats {({k: round(v, 2) for k, v in model.last_mix_stats.items()})}")
print("  -> they compose: C-Mixup changes WHAT the examples are (points between users),\n"
      "     GHM changes HOW MUCH each example counts (by gradient density). Nothing in\n"
      "     either needs to know about the other; the YAML just carries both blocks.")
