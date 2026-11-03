# NeurIPS 2026 EEG/EMG Challenge, Track 4 (EMG-to-pose) — team progress

*As of 2026-11-03 (supersedes the 2026-11-02 version). Written for the team and for Claude (Claude Code or claude.ai) picking the work up. Pair it with the `emg-pose-neuralbench-pipeline` skill (`.claude/skills/…/SKILL.md`), which holds the how-to; this file holds the what and the why. Sealed phase is running: 1 submission/day until Nov 21.*

## 1. Setting

- **Task.** emg2pose (Salter et al. 2024): 16-channel wrist EMG at 2 kHz → 20 hand joint angles, scored as MAE in degrees over independent 5-s windows. `predict(X)`: `(B,16,10000)` → `(B,20,10000)` radians. Bidirectional processing inside a window is allowed; 60 min on one A100; harness modifications disqualify. Codabench test partition = `user_stage` (unseen users performing unseen stages), windowed by the organizers with the IK-failure-free rule.
- **Stack.** Training in the fork `UCF-ESCAPE-TEAM/neuroai_jf`, branch `emg-pose-cnn` (NeuralTrain models + NeuralBench configs/CLI; latest commit `157826c`, Nov 1). Submission packaging in `UCF-ESCAPE-TEAM/NIPS_CHALLENGE_2026` (`submission/<model>/submission.py` on `main`, one folder per model, `weights.pt` git-ignored; latest `c54c50a`, Oct 30). Local testing with the official `2026-competition` benchopt harness. Claude has push access to both repos.
- **Compute.** Lab Linux PC, and the university SLURM cluster (`/lustre/fs1/home/ji045311/challenge_2026/{Data,Cache,Result,neuroai_jf}`; launch from an interactive compute node, never the login node).
- **Team.** Jirui (models, training, submissions) and Haonan (dataset analysis; added the `heldout_user`/`heldout_stage` test variants and measured the IK-failure losses, Nov 1). Both use Claude on the same repos.
- **Phases.** Warm-up (5 subs/day) ended Oct 25; sealed Oct 28 – Nov 21 (1/day); top-3 reproducibility audit.
- **Baseline to beat.** VEMG2Pose (braindecode): strided TDS conv encoder with LayerNorm over channels, 50 Hz rollout, 2-layer LSTM (hidden 512) conditioned on the previous pose, hybrid position→velocity output, ~6 M params. **16.78°** on Codabench.

## 2. Scoreboard (Codabench, degrees)

| Model / run | What it is | Score |
|---|---|---|
| VEMG2Pose | official baseline | 16.78 |
| StTcnPose raw | first version, no front end, original widths | 16.96 / 16.98 |
| StTcnPose R0 | front end present, normalisation **off** | 19.41 |
| **StTcnPose R1** | front end on (zscore/channel), f1 32, f2 128, tcn_filters 128 | **16.60 — best** |
| StTcnPose L1-loss control (original widths) | plain L1, un-widened trunk | 17.23 |
| StTcnPose + GHM-R run 1 | same trunk, GHM mu 0.1 / 30 bins | 17.53 (worse than its L1 control) |
| StLstmPose v1 | R1 trunk + VEMG2Pose-style LSTM, pose feedback, hidden 256 | 17.06 |
| StTcnPose + GHM-R run 2 (mu 0.35, 10 bins, no EMA) | YAML on branch | pending |
| StTcnPose + OHEM (keep 30 %, easy weight 0.1) | YAML on branch | pending |
| StTcnPoseCMixup | narrow trunk + C-Mixup batches | pending (submission.py in team repo) |
| StLstmPose L1 variant | `condition_on_pose: false`, `bidirectional: true`, `parameterization: position` (what `st_lstm_pose.yaml` now encodes) | pending |
| StTcnPose R1 + `min_finite_target_fraction 0.5` | R1 trunk, 1.9× training windows (see §4.12) | **not yet run — top of the queue** |
| StPoseTransformer | R1 front end + channel gate + multi-scale bank → 4-layer Conformer encoder (50 Hz) → position readout; OHEM, AdamW/OneCycle | not yet run (added Nov 3, `tutorial/transformer_pose_models.md`) |
| StVelTransformer | same trunk → position + velocity readouts, complementary fusion `fuse_gain 0.1` | not yet run (Nov 3) |

Local numbers for R1: val 0.231 rad, test 0.287 rad (≈16.4°), so the local `user_stage` test split tracks Codabench to within ~0.2° (at the default `min_finite_target_fraction 1.0`).

*If any "pending" row has a Codabench number by the time you read this, fill it in and re-rank §7.*

## 3. Models we built

All in `neuraltrain-repo/neuraltrain/models/` with YAMLs in `neuralbench-repo/neuralbench/models/`; submissions in `NIPS_CHALLENGE_2026/submission/<model>/`.

| Model | Idea | Status |
|---|---|---|
| `cnn1d_pose` | minimal strided 1-D CNN, pipeline smoke test | template only |
| `ulhm_pose` | upper-limb hand model with auxiliary pose loss (13-line `pl_module.py` patch for `auxiliary_losses`) | trained once; not pursued |
| `neuraldrive_pose` | neural-drive / decomposition-inspired CNN | trained; superseded |
| **`st_tcn_pose`** | `EmgPreproc` front end (FIR 20–450 Hz + 60/120/180 notches, per-window per-channel scaling, optional ring-interpolated bad-channel rejection) → EEGNet-style SpatialBlock (temporal FIR bank, ring convs on the 16-electrode band, depthwise collapse, pool ÷5, separable conv, pool ÷8 → 50 Hz) → EEG-TCNet TCN (2 modules × 3 dilated residual blocks, RF 85 frames = 1.7 s) → head (`linear` / `smooth` / `velocity`) → ×40 upsample. Model-level `causal` flag. | best model (R1) |
| `st_tcn_pose_cmixup` | same architecture (`StTcnPoseCMixup`), training batches replaced by C-Mixup convex mixes with a pose-similarity kernel (`mix_alpha 2`, `mix_sigma 0.5 rad`, cross-user pairs forced, mixing after the front end). Weights load into the plain StTcnPose submission. | trained/submitted, score pending |
| `st_tcn_pose_ghm` / `st_tcn_pose_ohem` | StTcnPose with `loss:` replaced by `GhmRegressionLoss` or `OhemRegressionLoss` (`neuraltrain/losses/hard_example.py`) | GHM run 1 done (worse); run 2 + OHEM pending |
| **`st_lstm_pose`** | same trunk → 1×1 to 64 → LSTM decoder (2 layers, hidden 256, hybrid/velocity/position heads, `output_scalar 0.01`); `condition_on_pose` toggles VEMG2Pose-style pose feedback (sequential, ~20× slower) vs one vectorised `nn.LSTM` call that can be bidirectional | v1 17.06; L1 variant pending |
| `st_pose_transformer` | R1 front end → optional SE gate over the 16 electrodes (`channel_attention`) → SpatialBlock with an optional multi-scale temporal bank (`ms_kernels` 15/31/63/127 taps, MSST idea at 2 kHz) → `SequenceEncoder` (4 × pre-norm Conformer blocks, d 128, 4 heads, RoPE, 300 ms conv module; `block_style: transformer` / `attn_window` / `causal` switches) → 1×1 position readout. 1.6 M params. Every addition is a YAML switch back to the R1 trunk. | code + YAML + tests, Nov 3; not trained |
| `st_vel_transformer` | same trunk → `VelocityDecoder`: position + velocity readouts, VEMG2Pose hybrid integration with a complementary filter (`fuse_gain`: 0 = pure integration, 1 = position readout) and an optional GRU state on the velocity head (`vel_gru_hidden`). Isolates the decoder question (smoothness vs anchoring) that StLstmPose v1 confounded with its causal encoder. | code + YAML + tests, Nov 3; not trained |

**Caution on the YAMLs (still true on `157826c`):** `st_tcn_pose.yaml`, `_ghm`, `_ohem` and `_cmixup` carry the *original* widths (`f1 16, f2 64, tcn_filters 64`) and `norm: maxabs`, not R1's (`32/128/128`, `zscore`). Only `st_lstm_pose.yaml` has the R1 trunk. Any run meant to compare against R1 needs those four values set explicitly, and `extract_weights_rev1.py` must point at the YAML that was actually trained.

**Dataset variants (new, Nov 1):** `tasks/emg/pose/datasets/heldout_user.yaml` and `heldout_stage.yaml`, selected with `neuralbench emg pose -m <model> --dataset heldout_user|heldout_stage`. Each overrides only the test-split `QueryEvents` filter (`generalization == 'user'` / `'stage'`), so train/val and the cache are shared with the default `user_stage` run. Use them to see which generalisation axis a change helps.

## 4. What we learned (the expensive lessons)

1. **Per-window normalisation inside the model is mandatory.** R0: 16.66° local, 19.41° Codabench. The server delivers EMG at std ~1e2 while NeuralBench batches were ~1e5–1e6; BatchNorm running variances of ~1e12 then produced a near-constant pose. The baseline survives because its first LayerNorm normalises over channels per sample. Fix: `EmgPreproc` scaling per window; non-persistent buffers rebuilt from config, never loaded from the checkpoint. The front end's `norm` default is now `maxabs`; R1 used `zscore`.
2. **NeuralBench checkpoints carry no hyper-parameters** (`save_weights_only=True`), and `best.ckpt` is deleted at exit unless `delete_checkpoints_on_exit: false`. `extract_weights_rev1.py` embeds the YAML's `brain_model_config` into `weights.pt` so `submission.py` rebuilds the exact architecture.
3. **Ctrl+C is graceful** (runs test, then deletes the checkpoint if configured); `kill -TERM` / `scancel` keeps `best.ckpt`.
4. **Causality is optional for the challenge** and costs accuracy: the non-causal TCN sees ±0.85 s. The v1 LSTM decoder is causal by construction, the likeliest explanation of its 0.46° deficit; hence the bidirectional L1 variant.
5. **Width before depth.** Doubling the three widths (R1) gave −0.36°; RF unchanged, so the gain was capacity, not context.
6. **Architecture alone has plateaued.** R1, StLstmPose and VEMG2Pose all land within half a degree while train loss keeps falling: the gap is train-users vs. test-users (and train-stages vs. test-stages), a domain-generalisation gap, not capacity. Hence the current effort on *training and data* (C-Mixup, hard-example mining, IK-window recovery, augmentation), not new trunks. Full argument in `tutorial/training_tutorial.md` + `cmixup_demo.py` / `hem_demo.py`.
7. **GHM-R down-weighted the hard tail.** Run 1 (mu 0.1, 30 bins) put 19–27 % of elements (|error| > ~22°) in the saturated top bin at weight 0.12–0.18, i.e. trained the hard frames at ~1/6 weight — the opposite of the intent — and scored 17.53° vs 17.23° for L1. Run 2 sets `mu 0.35`, `bins 10`, `momentum 0` (the paper's regression setting); watch `train/loss_ghm_frac_top` (< 0.15). OHEM is the mirror image: hardest 30 % of frames at full weight, rest at 0.1.
8. **C-Mixup diagnostics to read:** `train/mix_partner_dist` should sit clearly below the typical pairwise mean-pose distance in a batch (else lower `mix_sigma`); `train/mix_lambda_mean` ≈ 0.5 for α = 2; if train ≈ val and both high, raise `sigma` or lower `alpha`. Expect slower convergence; extend `patience` if val is still falling when early stopping fires.
9. **Linear ×40 upsampling at the output costs nothing measurable** (labels are 60 Hz mocap); ZOH in causal mode adds a staircase error.
10. **Kernel sizes**: at 2 kHz use 65–129 taps (32–64 ms) for the temporal FIR bank, not the EEGNet "sfreq/2" rule.
11. **Submission timing**: pose-feedback LSTM = 250 sequential steps/window (fine on GPU, 1.4 s per 8 windows on CPU); `condition_on_pose: false` is ~20× faster.
12. **The training set is starved of exactly the hard stages (Nov 1).** emg2pose's IK solver failed on 12.7 % of frames (`BAD_IK`, NaN angles in NeuralBench). `pl_module._run_step` already masks NaN frames out of the loss and metrics (upstream, not our patch), but the task config's `data.min_finite_target_fraction: 1.0` drops every window that overlaps a failure. Measured on the real data: **44 % of labelled training frames discarded**; windows 96k → 180k at threshold 0.5 (98 % of frames recovered). The drop is extremely uneven — Throwing ~95 % of windows removed, Punch ~77 % — i.e. the fast, large-amplitude stages the model generalises worst on are systematically under-sampled. Fix is one YAML line, `data: {min_finite_target_fraction: 0.5}` (new cache dir, rerun with `-f`). It cannot inflate the Codabench score (server windows at 1.0), it does not affect per-frame losses (L1/GHM/OHEM/C-Mixup are unaffected by label gaps), but it makes local val/test slightly harder and not comparable to 1.0 runs unless the threshold is applied to train only (small `data.py` patch: `keep |= split != 'train'`). The BAD_IK labels themselves (zeros/NaN) are never usable; the unlabelled EMG could additionally feed self-supervised pretraining later.
13. **Left-hand polarity needs no handling**: the sign flip was applied in the released HDF5; the BIDS conversion only rescaled units; skewness checks on 12 left/right pairs confirm one convention.
14. **All three loss variants were evaluated on the same frames** (threshold 1.0 clean windows), so their local numbers are mutually comparable; a 0.5-threshold run is not.

## 5. Training loop: what is fixed, what is configurable

All YAML-only; every change alters the config hash, so rerun with `-f`.

**Early stopping / checkpoint selection.** `TrainerConfig` (`neuralbench/utils.py:544-546`) has `patience`, `monitor`, `mode`; `main.py:353-373` passes the same `monitor`/`mode` to both `EarlyStopping` and `ModelCheckpoint(save_top_k=1)`, so one setting decides when training stops and which epoch becomes `best.ckpt`. Pose task default: `monitor: val/mae, mode: min, patience: 20, n_epochs: 100`. Override in the model YAML with a `trainer_config:` block. Monitorable: `val/loss`, `val/mae`, `val/mae_subject_mean`, `val/mae_subject_std`, `val/rmse`, `val/r2_score` (`mode: max`). A misspelled key fails only at the first validation end — smoke-test with `--debug`.

**What the validation split is.** `PredefinedSplit` with `valid_split_by: null` uses the emg2pose paper's own `split` column verbatim. Only **test** is filtered to one generalisation scenario (`user_stage` by default; `user` / `stage` via `--dataset`). **Val pools held-out users and held-out stages** into one `val/mae` weighted by window count. Options to narrow it: (a) an extra `QueryEvents` in `data.study`, e.g. `query: "split != 'val' or generalization == 'user'"`; (b) keep the pool but add `data.stream_by: [generalization]` plus a `GroupedMetric` without `reduction`, which logs `val/<name>/user` and `val/<name>/stage` separately and can be monitored; (c) `valid_split_by: user` to re-split train by user. Check the "Dataset summary" table at the top of `.out` before trusting a filter.

## 6. Infrastructure and files

| Thing | Where |
|---|---|
| Model code | `neuroai_jf/neuraltrain-repo/neuraltrain/models/{emg_frontend,st_tcn_pose,st_tcn_pose_cmixup,st_lstm_pose,neuraldrive_pose,ulhm_pose,cnn1d_pose}.py`; losses in `neuraltrain/losses/hard_example.py`; C-Mixup hook `train_batch_transform` in `neuralbench/pl_module.py` |
| Training YAMLs | `neuroai_jf/neuralbench-repo/neuralbench/models/*.yaml` (task config: batch 128, StandardScaler ×1e6, seed 33 only, L1 loss, `band_offsets [-1,0,1]` rotation augmentation, `min_finite_target_fraction 1.0`) |
| Test-set variants | `neuralbench-repo/neuralbench/tasks/emg/pose/datasets/{heldout_user,heldout_stage}.yaml` (`--dataset <stem>`) |
| Per-epoch metrics | `csv_config: {}` writes `lightning_logs/version_0/metrics.csv` under the run dir; `.out` has the validate table and epoch lines, `.err` the EarlyStopping lines and `Best epoch` |
| Weight extraction | `NIPS_CHALLENGE_2026/prepare_submission/extract_weights_rev1.py` (hard-coded paths at the top; worth converting to argparse for automation) |
| Submissions | `NIPS_CHALLENGE_2026/submission/<model>/submission.py` (st_tcn_pose, st_tcn_pose_ghm, st_tcn_pose_cmixup, st_lstm_pose, neuraldrive_pose, ulhm_pose); project copies under `claude/` |
| Local test + zip | `NIPS_CHALLENGE_2026/shell/run_submissions.sh SUBMISSION_ROOT COMPETITION_DIR` |
| Tutorials | fork `tutorial/neuroai-fork-install-and-update.md`, `tutorial/training_tutorial.md` (+ `cmixup_demo.py`, `hem_demo.py`), `tutorial/cmixup_tutorial.md`; claude.ai project `claude/lstm_tutorial.md`, `claude/slurm_tutorial.md`, `claude/neuralbench_slurm_infra.md`; Claude Doc "StTcnPose vs StLstmPose — architecture tutorial" |
| Results | `SAVE_DIR/neuralbench.main.Experiment.run,1/<hash>/best.ckpt` (cluster: `.../challenge_2026/Result`) |

## 7. Plan from here (ranked)

1. **Read out the pending runs** (GHM run 2, OHEM, CMixup, LSTM L1) — Codabench first, then `metrics.csv`: patience-stop vs `n_epochs`, train/val gap, and the loss-specific diagnostics in §4.7–4.8. One sealed submission per day, so decide the order from local test MAE.
2. **Run R1 with `min_finite_target_fraction: 0.5`** (§4.12). One line, nearly doubles the training windows, specifically restores the hard stages. Do this *before* any further loss/architecture work; optionally with the train-only `data.py` patch so `val/mae` stays comparable. Compare on Codabench against 16.60°.
3. **Bring the training-side YAMLs up to the R1 trunk** (widths 32/128/128, `norm: zscore`) so loss/mixup effects are measured against the 16.60° model, not the 17.23° one.
4. **Split the test number by generalisation axis**: rerun the scored checkpoints with `--dataset heldout_user` and `--dataset heldout_stage` (no retraining needed if the cache is intact — the same config hash plus a dataset suffix). If "stage" dominates the error, item 2 is the right lever; if "user", C-Mixup/augmentation is.
5. **R3** — StTcnPose `head_kind: velocity` (then `smooth`): explicit frame coupling without recurrence.
6. **Transformer trunk + decoder ablation** (`tutorial/transformer_pose_models.md` §4): P vs V0 vs V1 on `--dataset heldout_user` first, against an R1+OHEM baseline trained with the same AdamW/OneCycle block; only the winner goes to Codabench. If P ≈ baseline, stop here and spend uploads on training/data.
7. **Combine** the 0.5 threshold + C-Mixup + the better of GHM-2/OHEM (append the `loss:` block to the cmixup YAML), then stronger augmentation (`band_offsets ±2`, gain jitter, channel dropout).
8. **Ensemble** R1 + best training-side variant (+ seeds); error types differ (local jitter vs slow drift).
9. **Audit package**: fork commit hash, YAML, seed and `extract_weights` summary with every uploaded zip.

## 8. Open questions

- Does GHM-R with the paper's regression settings (run 2) stop down-weighting the tail, or is hard-example mining simply the wrong tool when the hard frames *are* the unseen users/stages?
- Is C-Mixup's pose-similarity kernel too tight (`sigma 0.5 rad`) for cross-user pairs to exist in a batch of 128? `train/mix_cross_subject_frac` answers this.
- Should early stopping track held-out-user val MAE alone, given the Codabench test is `user_stage`? Needs §5(b) numbers from a couple of runs.
- Do the heavily-dropped stages (Throwing, Punch) overlap the held-out stages of the test partition (paper Table 3 / Fig. 3)? If yes, §4.12 could matter more than any loss work.
- After the 0.5-threshold run: is the local val still a usable early-stopping signal, or is the train-only patch needed?

## 9. For Claude Code: things worth automating

Each is self-contained enough to hand off as one task; all paths above.

- **`run-readout`**: given a run dir (or a model/loss name to grep for under `SAVE_DIR`), print best epoch, stop reason, final train/val/test MAE in degrees, and the loss-specific diagnostics from `metrics.csv`; flag runs that hit `n_epochs` with val still falling.
- **`package-submission`**: given `best.ckpt` + training YAML + target `submission/<model>/`, run the extraction (argparse-ified `extract_weights_rev1.py`), the three Python ship-checks from the skill (§3), `run_submissions.sh --only <model>`, and write `submission/<model>/PROVENANCE.md` (commit hash, YAML copy, seed, extraction summary) for the audit.
- **`r1-ify`**: rewrite `st_tcn_pose{,_ghm,_ohem,_cmixup}.yaml` to the R1 trunk (f1 32, f2 128, tcn_filters 128, `norm: zscore`) with a one-line commit; or generate `_r1` copies if the narrow originals must be kept for comparison.
- **`min-finite-train-only`**: patch `neuralbench/data.py` so `min_finite_target_fraction` filters train only (`keep |= events.split != 'train'`), add a test, and a `st_tcn_pose_r1_mf05.yaml`.
- **`axis-split`**: for each scored checkpoint, launch `--dataset heldout_user` / `heldout_stage` evaluations (sbatch from a compute node) and tabulate `test/mae` per axis next to the Codabench number.
- **`daily-submit`**: a checklist/script that picks the best un-submitted zip by local test MAE, records provenance, and reminds that only one upload per day counts.
