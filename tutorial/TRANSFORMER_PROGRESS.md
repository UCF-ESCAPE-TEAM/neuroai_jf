# Transformer models for EMG-to-pose: progress and handoff

*Status as of 2026-11-03 21:50 ET, commit `e2693a5`+ (loss variants and _newton cluster YAMLs added) on `UCF-ESCAPE-TEAM/neuroai_jf`, branch `emg-pose-cnn`. Written so a teammate can pick this thread up in Claude Code. Read `CLAUDE.md` first, then this file, then `tutorial/transformer_pose_models.md` for the design rationale. The how-to for training / weights / submission is the `emg-pose-neuralbench-pipeline` skill (`.claude/skills/emg-pose-neuralbench-pipeline/SKILL.md`). Update this file whenever a run scores or a decision changes.*

## 1. Where this stands

| Step | State |
|---|---|
| Models `StPoseTransformer`, `StVelTransformer` written and registered | **done** (`neuraltrain-repo/neuraltrain/models/st_pose_transformer.py`, `st_vel_transformer.py`, `__init__.py`) |
| Training YAMLs | **done** — six (+ six `_newton` twins with the Newton 2-GPU `infra:` block, see `tutorial/neuralbench_slurm_infra.md`): `st_pose_transformer{,_ghm,_l1}.yaml`, `st_vel_transformer{,_ghm,_l1}.yaml` in `neuralbench-repo/neuralbench/models/` (OHEM / GHM-R run-2 setting / plain L1; identical otherwise) |
| `emg_vel_transformer` (raw-signal channel gate) | **code + YAML (+`_newton`) + 8 tests, Nov 5; not trained.** `StVelTransformer` with `ChannelGate` replaced by `RawChannelGate`: learned squeeze (bank of `gate_n_feat` temporal filters shared across electrodes, squared, time-averaged, log) instead of log-RMS/log-MAV; excitation `gate_excite: se` (SE Eq. 2, bottleneck) / `fc` (SE-Var3, full layer) / `eca` (ECA Eq. 9, faithful: one scalar per electrode, k-weight 1-D conv over the ring, no bias, `gate_eca_k: 0` = adaptive k) / `eca_multi` (ECA conv on all n_feat energies). `emg_vel_transformer_eca{,_newton}.yaml` is the base file with the one line changed to `eca`. Identity at init; `channel_attention: false` makes it == StVelTransformer. 1.84 M params. Compare against `st_vel_transformer` with everything else equal (ablation B′). |
| Unit tests | **done, passing** (`neuraltrain/models/test_st_transformer_pose.py`, 9 tests, ~5 s on CPU) |
| CPU smoke test on `(8, 16, 10000)` → `(8, 10000, 20)` | **done** (both models; finite, non-constant, strict state-dict round trip, causal variant verified) |
| `--debug` run on real data | **not done** |
| Full training runs (baseline, P×3 losses, V0, V1×3 losses) | **not started** |
| Codabench `submission.py` | **done** — `NIPS_CHALLENGE_2026/submission/st_{pose,vel}_transformer{,_ghm,_l1}/submission.py` (commit `ef4da53`), verified strict-load + output match against the fork models @ `cc2ecea`+ |
| Weight extraction / zip | **not done** |

Nothing has been trained. Every number in this file is a design choice, not a result.

## 2. What the models are (one paragraph each)

**`st_pose_transformer`** — StTcnPose R1 front end (FIR 20–450 Hz + notches, per-window z-score, bad-channel rejection) → SE-style gate over the 16 electrodes (`channel_attention`, identity at init) → the R1 `SpatialBlock` with its single 65-tap temporal bank replaced by a 4-kernel multi-scale bank (`ms_kernels: [15, 31, 63, 127]`) → pool to 50 Hz → 4 pre-norm Conformer blocks (d 128, 4 heads, FFN ×4, 15-frame conv module, rotary positions) → 1×1 readout of 20 absolute joint angles per frame → ×40 linear upsample. 1.62 M params.

**`st_vel_transformer`** — identical trunk → `VelocityDecoder`: a position readout and a velocity readout per frame; pose is the position for the first 12 frames (240 ms), then `pose_t = (1−g)(pose_{t−1} + 0.02·vel_t) + g·pos_t` with `fuse_gain g = 0.1` (complementary filter: velocity shapes the fast motion, position stops drift). `g = 0` is pure VEMG2Pose-style integration, `g = 1` is the pose model. `vel_gru_hidden > 0` adds a GRU state in front of the velocity readout.

Both: `causal: false` by default (bidirectional, what the competition allows). Loss: three YAMLs per model, differing **only** in the `loss:` block — base file = OHEM (keep 30 %, easy weight 0.1), `_ghm` = GHM-R run-2 setting (mu 0.35, 10 bins, momentum 0), `_l1` = plain L1 (the control). Optimiser AdamW 3e-4, OneCycleLR 10 % warmup + cosine, weight decay 0.05, clip 1.0 — **this differs from every earlier run (constant Adam 1e-3)**, see §4.

Every addition is a YAML switch back to the R1 trunk: `ms_kernels: null`, `channel_attention: false`, `block_style: transformer`, `attn_window: <frames>`, `n_layers`.

## 3. Next steps, in order

Each step is something you can hand to Claude Code as-is. Record results in §5 and in `tutorial/PROGRESS.md` §2.

1. **Debug run on the cluster** (10 min). From an interactive compute node (never the login node), `module purge`, activate `challenge_2026`, then
   ```bash
   neuralbench emg pose -m st_pose_transformer_newton --debug
   neuralbench emg pose -m st_vel_transformer_newton --debug
   ```
   On Newton always use the `_newton` stems: they carry the SLURM `infra:` block (2 GPUs, DDP, 24 h, evc22-50, mail on END/FAIL). The non-`_newton` files are for a local machine or a hand-written sbatch with `CLUSTER: null`. `--debug` forces 1 GPU.
   Pass criterion: the run reaches the validate table and one epoch without an error. Likely failure points: a config key the installed NeuralBench build rejects (then check `pip show neuraltrain` points into the clone, re-`pip install -e`), or OneCycleLR complaining about `total_steps` (it is injected from `estimated_stepping_batches` in `neuralbench/pl_module.py`; if the debug trainer gives 0 steps, run without `--debug` and `-f`).
2. **Baseline with the new optimiser.** The transformer YAMLs change the optimiser, so R1 + OHEM must be re-run with the same block or the comparison is confounded. Make `st_tcn_pose_ohem_adamw.yaml`: copy `st_tcn_pose_ohem.yaml`, confirm the R1 widths (`f1 32, f2 128, tcn_filters 128, norm zscore`), and paste in the `lightning_optimizer_config` + `trainer_config` blocks from `st_pose_transformer.yaml`. Train it.
3. **Full runs.** Two groups, submit each group to SLURM in one go (2 GPUs/node, ~0.3 h/epoch for R1; expect similar or slower). Seed 33 only (the grid is trimmed).
   - On Newton, append `_newton` to every stem below (e.g. `-m st_pose_transformer_newton`). The effective batch is 2 × 128 with DDP; the OneCycle `max_lr 3e-4` was set with that in mind, so do not halve it.
   - *Encoder/decoder question* (OHEM everywhere): P = `st_pose_transformer`; V0 = `st_vel_transformer` with `fuse_gain: 0.0` (make `st_vel_transformer_g0.yaml` by copying the base file); V1 = `st_vel_transformer`.
   - *Loss question* (same model, three losses): `st_pose_transformer_l1`, `st_pose_transformer_ghm`, and `st_vel_transformer_l1`, `st_vel_transformer_ghm`. Together with the base files that is a 2 × 3 grid; the L1 runs are the controls. The earlier OHEM > L1 > GHM ranking was measured on StTcnPose with constant Adam 1e-3, so it is not known to carry over to a transformer trunk under AdamW + warmup, which is why all three are re-run here rather than assumed.
   - If GPU time is short, run order: P-OHEM, P-L1, V1-OHEM, V1-L1, then V0, then the two GHM runs (GHM lost on the TCN; it is here to confirm, not to win).
4. **Read the runs out locally before Codabench.** For each: `best.ckpt` epoch, `val/mae`, `test/mae` on the default `user_stage` split, and `--dataset heldout_user` / `--dataset heldout_stage` (same config hash + dataset suffix, no retraining). Compare against R1 local: val 0.231 rad, test 0.287 rad. One sealed upload per day, so pick the order from local test MAE.
5. **Decide.** If P ≈ baseline (within ~0.2°), the TCN's inductive bias is adequate: stop the encoder work and put uploads into training/data (`min_finite_target_fraction 0.5`, C-Mixup) per `PROGRESS.md` §7. If P or V1 wins, run ablations A–E from `transformer_pose_models.md` §4 on the winner, one at a time.
6. **Submission.** `submission.py` already exists for all six configs in `NIPS_CHALLENGE_2026/submission/<stem>/` (Solver names `StPoseTransformer{,Ghm,L1}`, `StVelTransformer{,Ghm,L1}`). Per run: `extract_weights_rev1.py` with `CKPT_PATH` = the run's `best.ckpt`, `CONFIG_YAML` = the YAML actually trained (the `_newton` one on Newton), `OUT_PATH` = `submission/<stem>/weights.pt`; then `run_submissions.sh submission /path/to/2026-competition --only <stem>`. If the model code in the fork changes, regenerate the six scripts (the classes must stay verbatim) and re-run the strict-load check.

## 4. Decisions already made, and why

- **Replace the TCN, don't stack on it.** MSST stacks CNN → BiLSTM → Transformer; here the encoder takes the TCN's place at the same 50 Hz token rate, so the comparison is a clean encoder swap.
- **Conformer blocks, pre-norm, RoPE.** Local conv branch does most of the work on biosignals; pre-norm trains without careful warmup tuning; rotary positions don't tie the model to the 250-token training length. Rationale in `transformer_pose_models.md` §3.5.
- **Kernel lengths 15/31/63/127, not MSST's 1/3/5/7.** Theirs were for 500 Hz; at 2 kHz these span 7.5–63 ms.
- **No VMD, no GHM.** VMD compensated for a 3-channel, 7-subject dataset; GHM already lost to L1 here (17.53 vs 17.23). OHEM won the loss ablation, so both YAMLs use it.
- **Two models rather than one flag.** P vs V0 vs V1 measures smoothness vs anchoring with everything else fixed; StLstmPose v1 (17.06°) couldn't, because it changed encoder and decoder together.
- **AdamW + warmup instead of constant Adam 1e-3.** Transformers at the TCN's schedule tend to train slowly rather than diverge, which would make the encoder look worse than it is. Cost: the baseline must be re-run (§3 step 2).
- **`reject_channels: true`, `norm: zscore`** in both YAMLs: zscore is what R1 used; rejection matches the other current YAMLs.

## 5. Results log

| Date | Run | YAML (+ overrides) | Commit | best epoch | val/mae (rad) | test user_stage (rad / °) | heldout_user | heldout_stage | Codabench |
|---|---|---|---|---|---|---|---|---|---|
| | baseline R1+OHEM+AdamW | `st_tcn_pose_ohem_adamw.yaml` (to create) | | | | | | | |
| | P-OHEM | `st_pose_transformer.yaml` | | | | | | | |
| | P-GHM | `st_pose_transformer_ghm.yaml` | | | | | | | |
| | P-L1 | `st_pose_transformer_l1.yaml` | | | | | | | |
| | V0 | `st_vel_transformer.yaml`, `fuse_gain 0.0` | | | | | | | |
| | V1-OHEM | `st_vel_transformer.yaml` | | | | | | | |
| | V1-GHM | `st_vel_transformer_ghm.yaml` | | | | | | | |
| | V1-L1 | `st_vel_transformer_l1.yaml` | | | | | | | |

(1 rad = 57.2958°. R1 reference: val 0.231, test 0.287 ≈ 16.4°, Codabench 16.60°.)

## 6. Open questions

- Is 1.6 M parameters too much for a train-users → test-users gap? If train/val diverge early, first lever is `enc_dropout 0.2` + `n_layers 2`, not a new trunk.
- Does the OneCycle schedule finish before early stopping (`patience 20`) fires? If runs stop at epoch ~30 of 100, the LR never decays and the comparison with the TCN is unfair; set `n_epochs` to match the typical stop or switch the scheduler to `CosineAnnealingLR` with `T_max` = expected epochs × steps.
- `fuse_gain 0.1` was chosen by reasoning (200 ms drift time constant), not tuned. If V1 wins, try 0.05 and 0.3.
- The channel gate and `band_offsets [-1, 0, 1]` augmentation target the same cross-user problem; if the gate shows no gain (ablation B), consider widening the rotation augmentation instead.

## 7. File map

```
neuraltrain-repo/neuraltrain/models/st_pose_transformer.py   model + all shared blocks
neuraltrain-repo/neuraltrain/models/st_vel_transformer.py    velocity model (imports the above)
neuraltrain-repo/neuraltrain/models/test_st_transformer_pose.py
neuraltrain-repo/neuraltrain/models/__init__.py              two registration lines at the end
neuralbench-repo/neuralbench/models/st_pose_transformer.yaml
neuralbench-repo/neuralbench/models/st_vel_transformer.yaml
neuralbench-repo/neuralbench/models/st_{pose,vel}_transformer_{ghm,l1}.yaml   loss variants (only the loss: block differs)
neuraltrain-repo/neuraltrain/models/emg_vel_transformer.py        RawChannelGate + EmgVelTransformer (subclasses the vel model)
neuralbench-repo/neuralbench/models/emg_vel_transformer{,_eca}{,_newton}.yaml   se vs eca excitation, one line apart
neuralbench-repo/neuralbench/models/st_*_transformer*_newton.yaml           same six + Newton `infra:` block (2 GPUs, DDP)
tutorial/neuralbench_slurm_infra.md                           what every infra: key does, how to launch
tutorial/transformer_pose_models.md                          design + rationale + ablation plan
tutorial/TRANSFORMER_PROGRESS.md                             this file
tutorial/PROGRESS.md                                         team-wide scoreboard (rows added for both models)
```

Quick local check after pulling: `cd neuraltrain-repo && python -m pytest neuraltrain/models/test_st_transformer_pose.py -q` (needs torch, pydantic, exca, scipy, tqdm).
