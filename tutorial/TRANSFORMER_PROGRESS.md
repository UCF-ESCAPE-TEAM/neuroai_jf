# Transformer models for EMG-to-pose: progress and handoff

*Status as of 2026-11-03 17:00 ET, commit `f1076b0` on `UCF-ESCAPE-TEAM/neuroai_jf`, branch `emg-pose-cnn`. Written so a teammate can pick this thread up in Claude Code. Read `CLAUDE.md` first, then this file, then `tutorial/transformer_pose_models.md` for the design rationale. The how-to for training / weights / submission is the `emg-pose-neuralbench-pipeline` skill (`.claude/skills/emg-pose-neuralbench-pipeline/SKILL.md`). Update this file whenever a run scores or a decision changes.*

## 1. Where this stands

| Step | State |
|---|---|
| Models `StPoseTransformer`, `StVelTransformer` written and registered | **done** (`neuraltrain-repo/neuraltrain/models/st_pose_transformer.py`, `st_vel_transformer.py`, `__init__.py`) |
| Training YAMLs | **done** (`neuralbench-repo/neuralbench/models/st_pose_transformer.yaml`, `st_vel_transformer.yaml`) |
| Unit tests | **done, passing** (`neuraltrain/models/test_st_transformer_pose.py`, 9 tests, ~5 s on CPU) |
| CPU smoke test on `(8, 16, 10000)` → `(8, 10000, 20)` | **done** (both models; finite, non-constant, strict state-dict round trip, causal variant verified) |
| `--debug` run on real data | **not done** |
| Full training runs (baseline, P, V0, V1) | **not started** |
| Codabench `submission.py` for either model | **not written** |
| Weight extraction / zip | **not done** |

Nothing has been trained. Every number in this file is a design choice, not a result.

## 2. What the models are (one paragraph each)

**`st_pose_transformer`** — StTcnPose R1 front end (FIR 20–450 Hz + notches, per-window z-score, bad-channel rejection) → SE-style gate over the 16 electrodes (`channel_attention`, identity at init) → the R1 `SpatialBlock` with its single 65-tap temporal bank replaced by a 4-kernel multi-scale bank (`ms_kernels: [15, 31, 63, 127]`) → pool to 50 Hz → 4 pre-norm Conformer blocks (d 128, 4 heads, FFN ×4, 15-frame conv module, rotary positions) → 1×1 readout of 20 absolute joint angles per frame → ×40 linear upsample. 1.62 M params.

**`st_vel_transformer`** — identical trunk → `VelocityDecoder`: a position readout and a velocity readout per frame; pose is the position for the first 12 frames (240 ms), then `pose_t = (1−g)(pose_{t−1} + 0.02·vel_t) + g·pos_t` with `fuse_gain g = 0.1` (complementary filter: velocity shapes the fast motion, position stops drift). `g = 0` is pure VEMG2Pose-style integration, `g = 1` is the pose model. `vel_gru_hidden > 0` adds a GRU state in front of the velocity readout.

Both: `causal: false` by default (bidirectional, what the competition allows). Loss OHEM (keep 30 %, easy weight 0.1). Optimiser AdamW 3e-4, OneCycleLR 10 % warmup + cosine, weight decay 0.05, clip 1.0 — **this differs from every earlier run (constant Adam 1e-3)**, see §4.

Every addition is a YAML switch back to the R1 trunk: `ms_kernels: null`, `channel_attention: false`, `block_style: transformer`, `attn_window: <frames>`, `n_layers`.

## 3. Next steps, in order

Each step is something you can hand to Claude Code as-is. Record results in §5 and in `tutorial/PROGRESS.md` §2.

1. **Debug run on the cluster** (10 min). From an interactive compute node (never the login node), `module purge`, activate `challenge_2026`, then
   ```bash
   neuralbench emg pose -m st_pose_transformer --debug
   neuralbench emg pose -m st_vel_transformer --debug
   ```
   Pass criterion: the run reaches the validate table and one epoch without an error. Likely failure points: a config key the installed NeuralBench build rejects (then check `pip show neuraltrain` points into the clone, re-`pip install -e`), or OneCycleLR complaining about `total_steps` (it is injected from `estimated_stepping_batches` in `neuralbench/pl_module.py`; if the debug trainer gives 0 steps, run without `--debug` and `-f`).
2. **Baseline with the new optimiser.** The transformer YAMLs change the optimiser, so R1 + OHEM must be re-run with the same block or the comparison is confounded. Make `st_tcn_pose_ohem_adamw.yaml`: copy `st_tcn_pose_ohem.yaml`, confirm the R1 widths (`f1 32, f2 128, tcn_filters 128, norm zscore`), and paste in the `lightning_optimizer_config` + `trainer_config` blocks from `st_pose_transformer.yaml`. Train it.
3. **P, V0, V1 full runs.** `st_pose_transformer` as shipped (P); `st_vel_transformer` with `fuse_gain: 0.0` (V0, make `st_vel_transformer_g0.yaml`); `st_vel_transformer` as shipped (V1). Submit all three to SLURM at once (2 GPUs/node, ~0.3 h/epoch for R1; expect the transformer to be similar or slower). Seeds: the grid is trimmed to seed 33; keep it.
4. **Read the runs out locally before Codabench.** For each: `best.ckpt` epoch, `val/mae`, `test/mae` on the default `user_stage` split, and `--dataset heldout_user` / `--dataset heldout_stage` (same config hash + dataset suffix, no retraining). Compare against R1 local: val 0.231 rad, test 0.287 rad. One sealed upload per day, so pick the order from local test MAE.
5. **Decide.** If P ≈ baseline (within ~0.2°), the TCN's inductive bias is adequate: stop the encoder work and put uploads into training/data (`min_finite_target_fraction 0.5`, C-Mixup) per `PROGRESS.md` §7. If P or V1 wins, run ablations A–E from `transformer_pose_models.md` §4 on the winner, one at a time.
6. **Submission files** (only for whatever goes to Codabench). In `NIPS_CHALLENGE_2026/submission/st_pose_transformer/` (or `st_vel_transformer/`): `submission.py` per the skill's template, with verbatim copies of `_same_pad`, `_RingConv2d`, `SpatialBlock`, `_MultiScaleTemporal`, `MsSpatialBlock`, `ChannelGate`, `_rotary`, `_rotary_tables`, `_sinusoid`, `attention_mask`, `_SelfAttention`, `_FeedForward`, `_ConvModule`, `EncoderBlock`, `SequenceEncoder`, `upsample_frames`, plus `PosePost` (pose) or `VelocityDecoder` (velocity), and the `*Model` class. `CONFIG` = the YAML's `brain_model_config` minus `name`. `Solver.name = "StPoseTransformer"` / `"StVelTransformer"`. Then `extract_weights_rev1.py` pointed at the YAML actually trained, and `run_submissions.sh` with `--only`. Do the four checks in the skill §3 (strict load, non-constant output, raw-ckpt load, timing).

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
| | baseline R1+OHEM+AdamW | | | | | | | | |
| | P | `st_pose_transformer.yaml` | | | | | | | |
| | V0 | `st_vel_transformer.yaml`, `fuse_gain 0.0` | | | | | | | |
| | V1 | `st_vel_transformer.yaml` | | | | | | | |

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
tutorial/transformer_pose_models.md                          design + rationale + ablation plan
tutorial/TRANSFORMER_PROGRESS.md                             this file
tutorial/PROGRESS.md                                         team-wide scoreboard (rows added for both models)
```

Quick local check after pulling: `cd neuraltrain-repo && python -m pytest neuraltrain/models/test_st_transformer_pose.py -q` (needs torch, pydantic, exca, scipy, tqdm).
