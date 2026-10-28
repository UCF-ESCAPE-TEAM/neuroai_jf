# C-Mixup for EMG-to-pose (for `st_tcn_pose_cmixup.py`)

Companion script: `tutorial/cmixup_demo.py` — every numeric claim below is checked there.
Model: `neuraltrain/models/st_tcn_pose_cmixup.py` · YAML: `neuralbench/models/st_tcn_pose_cmixup.yaml`
· hook: `neuralbench/pl_module.py` (`train_batch_transform`).

## 1. The problem we are trying to fix

emg2pose is scored on **users the model never saw**. All our models (StTcnPose R1 16.60°,
StLstmPose 17.06°, VEMG2Pose 16.78°) land within half a degree of each other, while the train
loss keeps going down — the gap is between *training users* and *test users*, not a lack of
capacity. That is a **domain-generalization** gap: the training set covers the EMG manifold
only at a few dozen points (users), and the network is free to do anything in between.

The NeurIPS 2020 mixreg paper makes the same diagnosis for RL agents trained on a few hundred
game levels, and shows that interpolating *between* training environments fills in the
manifold far better than perturbing each observation on its own (cutout, random crop).
The RL wrapper is incidental; the regularizer they use is **mixup**. For our supervised
regression we can apply it directly — and C-Mixup is the version of mixup built for regression.

## 2. Mixup in one line

Mixup (Zhang et al., ICLR 2018) replaces each training pair by a convex combination of two:

```
lam ~ Beta(alpha, alpha)
x~  = lam * x_i + (1 - lam) * x_j
y~  = lam * y_i + (1 - lam) * y_j
```

and trains on `(x~, y~)` as if they were real data. Two readings of what this does:

1. **Data side (Vicinal Risk Minimization).** The empirical distribution (a spike on every
   training sample) is replaced by a distribution supported on all the segments between pairs of
   samples. The network now sees inputs *between* users, which is exactly where a test user lives.
2. **Function side.** The loss on mixed points is minimised when `f(x~) ≈ lam f(x_i) + (1-lam) f(x_j)`,
   i.e. when `f` is **close to linear along every segment between two training points**. That is a
   smoothness / small-Lipschitz prior: the decoder cannot bend wildly between two users'
   EMG patterns. Mixreg's Figure 2 visualises exactly this — the value function over the convex
   hull of three observations becomes smooth once both inputs *and* targets are mixed.

Reading 2 is why mixing the **targets** matters. Mixreg's ablation (their Figure 8) shows that
mixing inputs alone, keeping the label of the dominant sample, often does nothing; `mix_where`
and `train_batch_transform` in our code therefore always mix both.

## 3. Why vanilla mixup is risky for regression

For classification, a mixed label `0.6·fist + 0.4·open` is a perfectly sensible soft target.
For regression it is a *claim about the value of the function at the mixed input*:

```
network is told:   y~        = lam f(x_i) + (1-lam) f(x_j)
truth at x~ :      f(x~)     = f(lam x_i + (1-lam) x_j)
```

These agree only if `f` is linear between `x_i` and `x_j`. The EMG→angle map is not — mixing
a fist window with an open-hand window gives EMG that corresponds to no hand, labelled with a
half-open hand. `cmixup_demo.py` Part 1 measures this on `y = tanh(3x)` (smooth, non-linear):

| partner choice | mean partner label distance | `|y~ − f(x~)|` |
|---|---|---|
| any (vanilla mixup) | 0.49 | 0.142 |
| C-Mixup, σ = 0.3 | 0.16 | 0.022 |
| C-Mixup, σ = 0.1 | 0.06 | 0.006 |

The off-manifold error falls by 20× when partners have similar labels (the demo also notes
why this needs *similar label ⇒ similar input*, which holds for hand poses but not for a
periodic toy function). For regression you
want the *regularizing* effect of mixup (smooth `f`) without *lying* about `f` far from the data.

## 4. C-Mixup: choose the partner by label distance

C-Mixup (Yao, Wang, Zhang, Liang, Finn, NeurIPS 2022) keeps mixup's recipe and changes only
*who gets mixed with whom*. Instead of a random partner, sample `j` for each `i` with

```
P(j | i)  ∝  exp( − ‖y_i − y_j‖² / (2 σ²) )          (Gaussian kernel on the labels)
```

- `σ` is the **bandwidth**: `σ → ∞` is vanilla mixup; `σ → 0` always picks the nearest label,
  i.e. mixes a window with its near-duplicate (little regularization, no off-manifold risk).
- The labels used for the distance are a summary of the window: here the **mean 20-joint pose**
  over the 5-s window (`window_label_summary`, NaN frames ignored). The full frame-by-frame
  targets are still mixed sample by sample.
- `λ ~ Beta(α, α)` is unchanged. `α = 2` gives λ around 0.5 (strong mixing); `α = 0.2` gives λ
  near 0 or 1 (mostly the original window plus a little of the partner).

Their theory (Theorem 1 of the paper, informally): the kernel makes mixup a **label-smoothness
regularizer** — it penalises how fast the prediction changes across inputs whose labels are
close — instead of a global linearity constraint. For out-of-distribution test users that is the
property you want: an unseen EMG pattern should land near the pose of its nearest neighbours.

## 5. What the implementation adds: mixing *across users*, *after* the front end

Two choices in `st_tcn_pose_cmixup.py` go beyond the paper, both motivated by emg2pose:

**Partner from another user (`mix_same_subject_weight: 0`).** mixreg mixes observations from
*different levels* because the gap is between levels. Our gap is between users, so the kernel
is multiplied by `mix_same_subject_weight` for same-user pairs; `0` means "another user, with
the closest pose", with a fallback to any pose-similar partner if the batch has only one user.
`train/mix_cross_subject_frac` in the log shows how often that succeeded (expect ~1.0 with
batch 128 and dozens of users).

**Mix the front-end output, not the raw EMG (`mix_where: frontend`).** Raw EMG amplitude
differs by an order of magnitude between users (electrode impedance, muscle mass, placement).
If you mix raw windows, `0.5·x_loud + 0.5·x_quiet` is essentially the loud user with a
`λ = 0.5` target — the input λ and the label λ disagree. After `EmgPreproc` (FIR + per-channel
max-abs) every window is in `[−1, 1]`, so λ means the same thing on both sides. Mechanically:

```
BrainModule._run_step (train)                      StTcnPoseCMixupModel
  ├─ model.train_batch_transform(neuro, y, sid) ─►  draw perm, λ  (C-Mixup kernel, cross-user)
  │                                                 y~ = λ y + (1-λ) y[perm]   ← returned
  │                                                 keep (perm, λ) for the next forward
  ├─ y_pred = model(neuro)                      ─►  x = frontend(neuro)
  │                                                 x = λ x + (1-λ) x[perm]   ← same perm, λ
  │                                                 spatial → TCN → head  (unchanged)
  └─ loss(y_pred, y~)
```

In `eval()` nothing is mixed, the state dict is identical to `StTcnPose`, and the existing
`st_tcn_pose_submission.py` loads a C-Mixup checkpoint unchanged (`test_checkpoint_loads_into_plain_st_tcn_pose`).

`mix_where: input` is kept as the vanilla-mixup / mixreg control for an ablation.

## 6. How it should improve the score (and how it can fail)

Expected effects, in order of confidence:

1. **Smaller train/val gap.** The mixed data is harder to fit, so train MAE goes up a little;
   val MAE on held-out users should go down. If train MAE goes *down* with C-Mixup on,
   something is wrong (mixing not applied — check `train/mix_applied`).
2. **Better behaviour on unusual users.** The decoder is now constrained between users, so an
   unseen user falls on a learned interpolation rather than an extrapolation.
3. **Slower convergence.** Expect to need more epochs; `patience 20` in the task config may
   stop too early — watch the val curve before trusting a comparison.

Failure modes and the knob that fixes each:

| symptom | cause | fix |
|---|---|---|
| train loss ≈ val loss, both higher than R1 | too much regularization | `mix_alpha` 2 → 0.4, or `mix_prob` 1 → 0.5 |
| `train/mix_partner_dist` ≈ typical pairwise distance | σ too large: this is vanilla mixup | lower `mix_sigma` (0.5 → 0.25) |
| `train/mix_partner_dist` ≈ 0, no gain | σ too small: mixing near-duplicates | raise `mix_sigma` |
| `train/mix_cross_subject_frac` ≪ 1 | batches dominated by one user | check the sampler; or set the weight to 0.1 instead of 0 |
| val MAE improves, Codabench does not | front-end scale mismatch, not mixup | the usual `[input]` diagnostic in the submission log |

## 7. Running it

```bash
# fork root, env challenge_2026, from a compute node (never the login node)
neuralbench emg pose -m st_tcn_pose_cmixup --debug     # smoke test
neuralbench emg pose -m st_tcn_pose_cmixup             # full run (R1 architecture + C-Mixup)
```

Metrics to read in `.out` besides `val/mae`: `train/mix_lambda_mean` (≈0.5 for α = 2),
`train/mix_partner_dist` (rad; want it clearly below the batch's typical pairwise
mean-pose distance — in the demo's synthetic batch σ = 0.5 cuts it from 3.8 to 1.1 rad; on
real poses it should be lower still, and the first real run tells you where to put σ), `train/mix_cross_subject_frac` (≈1).

Planned comparisons (one change per run, same seed):

| run | YAML change | question |
|---|---|---|
| R1 | — (`st_tcn_pose.yaml`) | baseline, 16.60° |
| CM1 | `st_tcn_pose_cmixup.yaml` as is | does C-Mixup help? |
| CM2 | `mix_where: input` | is mixing after the front end the right place? |
| CM3 | `mix_sigma: 100` | is the label kernel doing anything (vanilla mixup control)? |
| CM4 | `mix_same_subject_weight: 1` | does cross-user pairing matter? |
| CM5 | CM1 + `loss:` block from `st_tcn_pose_ghm.yaml` | HEM and C-Mixup together |

Weight extraction and submission are unchanged: `extract_weights_rev1.py` with
`CONFIG_YAML = st_tcn_pose_cmixup.yaml`, then the StTcnPose `submission.py` (the embedded config
keeps only keys the submission knows, so the `mix_*` keys are dropped automatically).

## 8. Reading `train_batch_transform` with this in hand

```python
labels = window_label_summary(target)                 # §4: (B, 20) mean pose per window
perm   = cmixup_partners(labels, sigma, subject_ids, same_subject_weight)   # §4-5: kernel
lam    = Beta(alpha, alpha).sample((B,))              # §2
target_mixed = lam * target + (1 - lam) * target[perm]   # §2: targets mixed frame by frame
self._mix_state = (perm, lam)                         # §5: forward() mixes frontend(x) the same way
```

That is the whole method: keep the network, change what a "training example" is — a point
on the segment between two users with similar poses — so that the decoder learns the space
between the users it saw, which is where the users it will be scored on live.

## References

- Zhang, Cissé, Dauphin, Lopez-Paz. *mixup: Beyond empirical risk minimization.* ICLR 2018.
- Yao, Wang, Zhang, Liang, Finn. *C-Mixup: Improving generalization in regression.* NeurIPS 2022.
- Wang, Kang, Shao, Feng. *Improving generalization in reinforcement learning with mixture
  regularization.* NeurIPS 2020 (mixreg; §3.2 and Fig. 8 for "mix the supervision too").
