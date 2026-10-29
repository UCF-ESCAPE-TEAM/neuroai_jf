"""Numerical companion to training_tutorial.md (Part A). Run: python tutorial/cmixup_demo.py

Part 1  why plain mixup is risky for REGRESSION: the mixed target
        lam*f(x_i) + (1-lam)*f(x_j) is far from f(lam*x_i + (1-lam)*x_j) when
        the two labels are far apart, and close when they are close.
Part 2  the C-Mixup kernel in the real implementation
        (neuraltrain.models.st_tcn_pose_cmixup.cmixup_partners): partner
        distance and cross-subject fraction as a function of sigma.
Part 3  what the StTcnPoseCMixup hook does to a batch, end to end on CPU.
"""

import torch

from neuraltrain.models.st_tcn_pose_cmixup import cmixup_partners, window_label_summary

torch.manual_seed(0)

# ----------------------------------------------------------------------------
print("\nPart 1 -- off-manifold targets: vanilla mixup vs C-Mixup on y = tanh(3x)")
f = lambda x: torch.tanh(3 * x)                      # smooth, non-linear, monotone "true" decoder
# (monotone so that similar labels <=> similar inputs, as for hand poses; a
#  periodic f would break that link and C-Mixup would gain little)
x = torch.rand(2000, 1) * 2 - 1                      # inputs in [-1, 1]
y = f(x)
lam = torch.distributions.Beta(2.0, 2.0).sample((2000, 1))


def manifold_error(perm):
    x_mix = lam * x + (1 - lam) * x[perm]            # what the network SEES
    y_mix = lam * y + (1 - lam) * y[perm]            # what the network is TOLD
    return (y_mix - f(x_mix)).abs().mean().item()    # distance from the truth at x_mix


vanilla = torch.randperm(2000)
print(f"  vanilla mixup  (any partner)            |y_mix - f(x_mix)| = {manifold_error(vanilla):.3f}")
for sigma in (1.0, 0.3, 0.1, 0.03):
    perm = cmixup_partners(y, sigma=sigma)
    print(f"  C-Mixup sigma={sigma:<5}                   |y_mix - f(x_mix)| = {manifold_error(perm):.3f}"
          f"   mean |y_i - y_j| = {(y - y[perm]).abs().mean():.3f}")
print("  -> the smaller the label distance between partners, the more the mixed\n"
      "     target agrees with the true function at the mixed input.")

# ----------------------------------------------------------------------------
print("\nPart 2 -- the kernel on pose-like labels (20 joints, 4 'users' in a batch of 128)")
B, J = 128, 20
sid = torch.arange(B) % 4                            # 4 subjects in the batch
pose = torch.randn(B, 3) @ torch.randn(3, J) * 0.4  # 20-d mean poses on a 3-d manifold, rad
pairwise = torch.cdist(pose, pose)
print(f"  typical pairwise distance in this batch: median {pairwise.median():.2f} rad")
for sigma in (100.0, 1.0, 0.5, 0.25):
    for w in (1.0, 0.0):
        perm = cmixup_partners(pose, sigma=sigma, subject_ids=sid, same_subject_weight=w)
        d = (pose - pose[perm]).norm(dim=1).mean().item()
        cross = (sid != sid[perm]).float().mean().item()
        print(f"  sigma={sigma:<6} same_subject_weight={w}:  mean partner dist {d:.2f} rad, "
              f"cross-subject {cross:.0%}")
print("  -> sigma sets how similar the partner's pose is (train/mix_partner_dist in the log);\n"
      "     same_subject_weight=0 forces the partner to be another user (train/mix_cross_subject_frac).")

# ----------------------------------------------------------------------------
print("\nPart 3 -- the training hook on a real StTcnPoseCMixup (CPU, tiny config)")
import pydantic
from neuraltrain.models.base import BaseBrainModelConfig


class _H(pydantic.BaseModel):
    brain_model_config: BaseBrainModelConfig


cfg = _H(brain_model_config={
    "name": "StTcnPoseCMixup", "notch_freqs": None, "bp_taps": 65, "f1": 4, "depth_mult": 1,
    "f2": 8, "tcn_modules": 1, "tcn_blocks": 2, "tcn_filters": 8, "mix_sigma": 0.5,
}).brain_model_config
model = cfg.build(16, 20).train()
emg = torch.randn(8, 16, 2000) * torch.tensor([1, 40, 1, 40, 1, 40, 1, 40.0]).view(-1, 1, 1)
target = torch.randn(8, 20, 2000) * 0.3
subjects = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])

emg_out, target_out = model.train_batch_transform(emg, target, subjects)
perm, lam = model._mix_state
print(f"  lambda per window : {[f'{v:.2f}' for v in lam.tolist()]}")
print(f"  partner per window: {perm.tolist()}   subjects: {subjects.tolist()}")
print(f"  stats logged      : { {k: round(v, 3) for k, v in model.last_mix_stats.items()} }")
print(f"  raw EMG returned unchanged (mix_where=frontend): {emg_out is emg}")
feats = model.frontend(emg)                          # FIR + per-channel max-abs -> [-1, 1]
print(f"  front-end output range per window: {[f'{v:.2f}' for v in feats.abs().amax(dim=(1, 2)).tolist()]}")
print("  (raw amplitudes differ 40x between users; after the front end both are O(1),\n"
      "   so lambda on the features means the same thing as lambda on the targets)")
out = model(emg)                                     # consumes the pending pairing
print(f"  forward output {tuple(out.shape)}, pending pairing cleared: {model._mix_state is None}")
model.eval()
with torch.no_grad():
    a, b = model(emg), model(emg)
print(f"  eval mode is deterministic and un-mixed: {torch.equal(a, b)}")
