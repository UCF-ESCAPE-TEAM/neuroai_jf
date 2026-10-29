# Training-side generalization for EMG-to-pose: C-Mixup and hard example mining

Two ways to improve a decoder **without touching its architecture**: change what a training
example *is* (Part A, C-Mixup) and change how much each example *counts* (Part B, hard
example mining with GHM-R / OHEM). Part C is how they combine.

Companion scripts — every numeric claim below is checked there:
`tutorial/cmixup_demo.py` (Part A) and `tutorial/hem_demo.py` (Parts B–C).

Code: `neuraltrain/models/st_tcn_pose_cmixup.py` · `neuraltrain/losses/hard_example.py` ·
hook in `neuralbench/pl_module.py` (`train_batch_transform`) · YAMLs
`neuralbench/models/st_tcn_pose_cmixup.yaml`, `st_tcn_pose_ghm.yaml`, `st_tcn_pose_ohem.yaml`.

---

# Part A — C-Mixup

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

---

# Part B — Hard example mining (GHM-R and OHEM)

## 9. The problem *this* one fixes

Plain L1 on a 5-s window treats every (frame, joint) element the same. But most elements are
easy: the hand is resting, or moving slowly, and the model already gets them within a few
degrees. They are the majority, so they dominate the gradient, and the model keeps polishing
what it already does well. The frames that decide the MAE — fast transitions, unusual
grips, poorly-recorded users — are a minority and their gradient is drowned out.

Hard example mining re-balances the loss toward that minority. Two classic ways:

| | idea | origin |
|---|---|---|
| **OHEM** | keep only the hardest fraction of examples per batch, drop the rest | Shrivastava, Gupta, Girshick, CVPR 2016 (detection RoIs) |
| **GHM** | weight every example by the *inverse density* of its gradient norm, so crowded regions of "how wrong am I" are down-weighted and sparse ones up-weighted | Li, Liu, Wang, AAAI 2019 (GHM-C for classification, GHM-R for box regression); reused for EMG torque/angle by Lin et al., Measurement 2026 (MSST) |

Both are implemented as drop-in replacements for `L1Loss` in `neuraltrain/losses/hard_example.py`
and selected from the YAML `loss:` block. Neither changes the model or the checkpoint.

## 10. GHM-R step by step

**Base loss.** GHM-R needs a per-element loss whose gradient norm is bounded, so it uses the
*Authentic Smooth L1*:

```
ASL1(d) = sqrt(d² + μ²) − μ            d = y_pred − y_true (rad)
g(d)    = |dASL1/dd| = |d| / sqrt(d² + μ²)   ∈ [0, 1)
```

`ASL1` is L2 for `|d| ≪ μ`, L1 for `|d| ≫ μ`. Its gradient norm `g` is a monotone, saturating
measure of "how wrong is this element": `g = 0.71` at `|d| = μ`, `g > 0.95` beyond `3 μ`.
That is why **μ must be chosen for the error scale of the task** (`hem_demo.py` Part 1):

| \|error\| rad | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|
| g, μ = 0.02 | 0.93 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| g, μ = 0.1 | 0.45 | 0.71 | 0.89 | 0.95 | 0.98 | 1.00 |
| g, μ = 0.3 | 0.16 | 0.32 | 0.55 | 0.71 | 0.86 | 0.96 |

The detection papers use μ = 0.02 because their targets live in [0, 1]; for us (R1's MAE is
0.29 rad, the median error lower) that would put *every* error above 0.1 rad in the same bin
and GHM would have nothing to harmonize.

**Gradient density.** Cut `[0, 1]` into `M = bins` equal regions, count how many of the
batch's `N` elements fall in each (`GD(g)`), optionally as an EMA over batches
(`momentum`, GHM Eq. 12–13) so the density does not jump from batch to batch.

**Weight.** Each element gets `β = N / GD(g)` (Eq. 7/10), i.e. *inverse density*, scaled
so the mean weight is exactly 1 (`normalize_by: nonempty_bins`, the authors' released code).
The loss is `mean(β · ASL1(d))`. Mean weight 1 means the loss scale and the learning rate
carry over from the L1 runs unchanged.

**What it does to a batch** (`hem_demo.py` Part 2 — 10 000 elements, a Gaussian error cloud
of std 0.15 rad, a 5 % tail at ~0.7 rad and one 3-rad label-noise frame):

| \|error\| rad | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | 0.70 | 1.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| weight, μ = 0.1 | 1.74 | 1.44 | 0.86 | 0.44 | 0.46 | 0.72 | 0.72 | 0.72 | 0.72 |
| weight, μ = 0.3 | 0.63 | 0.69 | 0.72 | 0.88 | 1.63 | 6.67 | 8.55 | 3.70 | 2.85 |

Read the two rows:

- **μ = 0.1** halves the crowded 0.2–0.3 rad bulk and boosts the rare tiny errors — but
  everything above ~0.4 rad shares *one* saturated bin with the whole 5 % tail, so the hard
  tail is not boosted at all (0.72) and a 3-rad outlier gets exactly the weight of a 0.5-rad
  miss. `ghm_frac_top` = 0.05 in the log tells you the saturated bin holds 5 % of elements.
- **μ = 0.3** spreads the tail over several sparse bins and boosts it 4–9×, while the bulk
  drops to ~0.7. This is the "focus on hard examples" regime the paper intends.

So the single most important knob is μ, and `train/loss_ghm_frac_top` is its gauge: if the
fraction in the top bin is large, μ is too small for your current error level. Because the
model's errors shrink during training, a μ that is right at epoch 1 can be too large at
epoch 50 (nothing saturates, the weights flatten toward 1) — that is harmless, it just turns
GHM off gently.

**Outliers.** GHM's weight is bounded by `1/density`, never by the error size (unlike L2,
where a 3-rad error contributes 100× the gradient of a 0.3-rad one). That is its protection
against label noise — partial: in the μ = 0.3 row the outlier is still boosted ~3×, along
with the legitimate hard tail it shares bins with. GHM cannot tell a hard-but-correct frame
from a wrongly-labelled one; only data cleaning can.

YAML (`st_tcn_pose_ghm.yaml`):
```yaml
loss:
  =replace=: true
  name: GhmRegressionLoss
  mu: 0.1            # rad; raise if train/loss_ghm_frac_top is high
  bins: 30           # GHM found 10-30 enough; insensitive above ~20
  momentum: 0.75     # EMA of the per-bin density; 0 = per-batch only
  normalize_by: nonempty_bins
  weight_in_eval: false   # val/loss is plain ASL1 -> comparable to the L1 runs
```

## 11. OHEM step by step

OHEM is simpler: compute the per-element loss, rank, back-propagate only the top
`keep_fraction`. Our regression version adds two things the original did not need:

- `select: frame` ranks whole frames by their joint-mean loss and keeps all 20 joints of a
  selected frame. Elements of one frame are strongly correlated (the regression analogue of
  OHEM's overlapping RoIs, which they de-duplicate with NMS), so mining per frame is the
  closer match; `select: element` mines each (frame, joint) on its own.
- `easy_weight` keeps the non-selected majority alive at a reduced weight instead of
  discarding it (original OHEM sets it to 0). With 70 % of the frames dropped every step the
  model can forget what it knew about the easy regime; 0.1 is a soft compromise.

The loss is normalised by the total weight, so its scale does not depend on `keep_fraction`.

`hem_demo.py` Part 3, same errors as above, `keep_fraction 0.3`: the kept frames have mean
error 0.26 rad vs the batch's ~0.12; every selected frame gets the same weight regardless of
how wrong it is (L1), so the single 3-rad label-noise frame takes 0.7 % of the gradient —
no more than any other selected frame. OHEM is thus *milder* toward outliers than
GHM-at-a-good-μ, but *harsher* toward easy frames (dropped entirely unless `easy_weight > 0`).

YAML (`st_tcn_pose_ohem.yaml`):
```yaml
loss:
  =replace=: true
  name: OhemRegressionLoss
  keep_fraction: 0.3   # OHEM kept ~3 % of RoIs; for pose 0.25-0.5
  easy_weight: 0.1     # 0 = original OHEM
  select: frame        # frame | element
  base: l1             # l1 | asl1 (then mu applies)
  weight_in_eval: false
```

## 12. GHM-R vs OHEM: which to try first

| | GHM-R | OHEM |
|---|---|---|
| what is re-weighted | every element, continuously, by gradient density | top-k frames at weight 1, rest at `easy_weight` |
| hyper-parameter that matters | μ (must match the error scale; watch `ghm_frac_top`) | `keep_fraction` |
| behaviour on label noise | bounded by 1/density; boosted along with the hard tail | same weight as any selected frame |
| behaviour on the easy majority | down-weighted, never dropped | dropped (or ×`easy_weight`) |
| loss scale vs L1 | same (mean weight 1) | same (normalised by total weight) |
| cost | one histogram per step | one k-th-value per step |

Try GHM-R first: it degrades gracefully (weights → 1 when nothing is special), it is what the
EMG literature (MSST) reports gains with, and its gauge is logged. Use OHEM as the second
data point; if both help, the hard examples are real; if only OHEM helps, suspect that GHM's
μ was wrong rather than the idea.

What the "hard" frames usually are in emg2pose: fast finger transitions, the first/last
0.5 s of a window (the FIR's edge), rare grips, and users with poor electrode contact. Before
trusting a gain from either loss, dump the top-loss windows of one validation epoch and look
at which users/sessions they come from — if they cluster in one or two recordings, you are
mining label noise (glove drift, tracking failures), and the fix is to exclude or
down-weight those recordings, not to mine harder.

---

# Part C — Using them together

## 13. Why they compose

They act on different things and do not know about each other:

```
batch ──► C-Mixup: x~ = λ x_i + (1-λ) x_j,  y~ = λ y_i + (1-λ) y_j   (WHAT an example is)
      ──► model forward
      ──► GHM-R / OHEM on (y_pred, y~)                                (HOW MUCH it counts)
```

`hem_demo.py` Part 4 runs exactly this on a `StTcnPoseCMixup` with `GhmRegressionLoss`; the
YAML for it is `st_tcn_pose_cmixup.yaml` with the `loss:` block from `st_tcn_pose_ghm.yaml`
pasted in (run CM5 in the ablation table of §7).

Two interactions to keep in mind:

1. **Mixed targets are slightly easier.** `y~` is an average, so its extremes are tamed and
   the error distribution narrows; the right μ for GHM may be a little *smaller* with
   C-Mixup on than without. Use `ghm_frac_top` to re-check, not the value from the L1 run.
2. **Both reduce the easy majority's gradient.** C-Mixup replaces easy windows by in-between
   windows; GHM then down-weights whatever is still crowded. Expect the train loss to be
   noticeably higher than R1's and converge slower; compare on `val/mae` only, and extend
   `patience` if the val curve is still falling when early stopping triggers.

## 14. Reading the logs

| key | from | healthy value |
|---|---|---|
| `train/mix_applied` | C-Mixup | 1.0 (= `mix_prob`) |
| `train/mix_lambda_mean` | C-Mixup | ≈ 0.5 for α = 2 |
| `train/mix_partner_dist` | C-Mixup | well below the batch's typical mean-pose distance |
| `train/mix_cross_subject_frac` | C-Mixup | ≈ 1.0 |
| `train/loss_ghm_frac_top` | GHM-R | small (< 0.2); high → raise μ |
| `train/loss_ghm_w_top` | GHM-R | the weight the saturated bin gets; ≫ 1 means the tail is being mined |
| `train/loss_ohem_hard_mean` | OHEM | mean error of the mined frames; should stay above `train/loss` |
| `val/loss` | all | plain L1 / ASL1 (`weight_in_eval: false`), comparable across runs |

Submission is unchanged for every combination: the losses have no parameters and are not in
the checkpoint, and `StTcnPoseCMixup` ships as `StTcnPose`.

## References

- Shrivastava, Gupta, Girshick. *Training region-based object detectors with online hard example
  mining.* CVPR 2016.
- Li, Liu, Wang. *Gradient harmonized single-stage detector.* AAAI 2019 (GHM-C / GHM-R).
- Lin et al. MSST, *Measurement* 2026 (GHM-R / "GHM-L" for EMG torque and angle regression; Eq. 9–15).

- Zhang, Cissé, Dauphin, Lopez-Paz. *mixup: Beyond empirical risk minimization.* ICLR 2018.
- Yao, Wang, Zhang, Liang, Finn. *C-Mixup: Improving generalization in regression.* NeurIPS 2022.
- Wang, Kang, Shao, Feng. *Improving generalization in reinforcement learning with mixture
  regularization.* NeurIPS 2020 (mixreg; §3.2 and Fig. 8 for "mix the supervision too").
