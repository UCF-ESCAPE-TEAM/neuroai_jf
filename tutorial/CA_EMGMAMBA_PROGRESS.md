# CaEmgMamba: progress and handoff

*Status as of 2026-11-06, updated from Claude Code on the lab workstation. Read `CLAUDE.md` in the fork first, then this file. The how-to for registering, training, extracting weights and submitting is the `emg-pose-neuralbench-pipeline` skill (`.claude/skills/emg-pose-neuralbench-pipeline/SKILL.md`). Update this file whenever a step completes or a decision changes.*

## 1. Where this stands

| Step | State |
|---|---|
| Model `CaEmgMamba` | **done, runs** (`neuraltrain/models/ca_emgmamba.py`, registered in `models/__init__.py`) |
| Training YAMLs + `_newton` twins | **done**; the two non-Newton YAMLs ran end to end with `--debug`; the twins were only checked to equal them plus the `infra:` block, never launched on Newton |
| Unit tests | **22 pass, 1 skipped** (`test_fast_scan_parity`, needs `mamba_ssm`) |
| `mamba-ssm` kernel, parity test | **NOT verified.** Never built or run. The call was only compared by reading the mamba-ssm 2.3.2 source (`Mamba.forward`); it matches, but `scan_impl: auto` is unproven until step 4 passes |
| Full runs | **not started**: no real validation number exists yet |
| Codabench `submission.py` (one per head) | **not started** (needs weights from a full run) |

**What was tested, on real data:** `neuralbench emg pose -m ca_emgmamba_{pose,vel} --debug` (workstation, pure-PyTorch scan, one GPU). Both finish train, validate, test; `best.ckpt` has the expected parameter count. Debug val/mae fell in both (pose 0.433 -> 0.367 rad; vel 0.486 -> 0.356 rad). These runs use the tiny debug subset, so they show the pipeline works, not how good the model is.

**Fixed while testing:**
- `norm_eps` 1e-8 -> 1e-12. At 1e-8, input in volts (std ~1e-6) was attenuated by about 1 %, breaking the amplitude-invariance claim (real bug, caught by `test_amplitude_invariance`).
- `test_forward_block_is_causal_and_bidirectional_is_not` perturbed future frames by a constant, which `LayerNorm` cancels exactly, so the layer saw no change. The test now uses a random perturbation (test bug, model unchanged).
- Added `scan_checkpoint` (default true): batch 128 ran out of memory on a 23.6 GiB GPU slice with the pure-PyTorch scan.
- `ca_emgmamba_pose.yaml` now has `n_epochs: 40`, `patience: 15`, as `st_pose_transformer.yaml` does. The vel YAML keeps the task default, as `st_vel_transformer.yaml` does. Without this the "comparable runs" claim was false.
- Added `test_dead_electrode_stays_finite` and `test_scan_checkpoint_same_output_and_grads`.

## 2. What the model is

```
EMG (B, 16, 10000) @ 2 kHz
 -> WindowNorm        z-score per electrode over the window, no parameters
 -> ConvStem          3 x [Conv1d(stride) -> GroupNorm(8) -> GELU -> Dropout -> channel attention]
                      k 31/9/5, stride 5/4/2, channels 64/128/192 : 2000 -> 400 -> 100 -> 50 Hz
 -> MambaEncoder      4 bidirectional Mamba layers, d_model 192, d_state 16, expand 2, d_conv 4
 -> head              pose: 1x1 readout of 20 absolute angles per frame
                      vel : position + velocity readouts, integrated with complementary fusion
 -> x40 linear upsample -> (B, 10000, 20), time-major
```

- **One config class, `CaEmgMamba`, with `head: pose | vel`.** The two YAMLs select the head; the trunk is identical. "Two heads" was read as two selectable variants. Jerry has not confirmed this reading (see section 6).
- **Channel attention** (`ChannelAttention`): squeeze = mean over time of the post-GELU features; excite = SE bottleneck MLP (`ca_kind: se`, reduction 8) or ECA 1-D conv across channels (`ca_kind: eca`, adaptive kernel). Gain is `2 * sigmoid(.)` with a zero-initialised last layer, so every gate is the identity at init. `ca_stages` (1-based list) picks which stem stages get a gate.
- **Mamba block** (`MambaBlock`): original Mamba (Gu & Dao 2023). Parameter names and shapes mirror `mamba_ssm.Mamba` (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `A_log`, `D`, `out_proj`). A layer is pre-LayerNorm, forward block plus a second block on the reversed frames, averaged, residual.
- **Scan** (`scan_impl`): `torch` is a pure-PyTorch loop, one step per frame; `scan_checkpoint: true` recomputes each scan in backward instead of storing it (same gradients, tested). `auto` (the YAML default) calls `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` when the package imports and the tensors are on CUDA, and otherwise falls back to the loop. Same weights either way.
- **Velocity head** (`VelocityHead`): same maths as `VelocityDecoder` in `st_vel_transformer.py` without the GRU option. `position_steps 12`, `vel_scale 0.02`, `fuse_gain 0.1`.
- **Loss / optimiser:** OHEM (keep 30 %, easy weight 0.1), AdamW 3e-4, OneCycleLR 10 % warmup + cosine, weight decay 0.05, clip 1.0. Copied from `st_pose_transformer.yaml` so runs are comparable.
- **Self-contained:** the model file imports only `.base.BaseBrainModelConfig`. Nothing from `emg_frontend`, `st_tcn_pose` or the `st_*_transformer` files.

## 3. Next steps, in order

Steps 1-3 (files in the fork, tests, parameter count and timing) and step 5 (`_newton` twins) are done; see sections 1 and 5.

4. **Install `mamba-ssm` in the `challenge_2026` conda env on Newton** (`pip install mamba-ssm`; it compiles CUDA code, so do it on a compute node with `cuda/cuda-12.6.0` loaded). Then run the test file on a GPU node and confirm `test_fast_scan_parity` passes rather than skips. **Do this before any training on a machine that has the package**, because `auto` silently switches to the kernel there. If the package will not build, use the pure-PyTorch scan (`auto` already falls back to it).
6. **Debug run, then full runs.** `neuralbench emg pose -m ca_emgmamba_pose_newton --debug`, same for `_vel`. Then both full runs, seed 33. Pure-PyTorch cost is measured in section 5: about 3.1 s per step at batch 128. At ~750 steps per epoch (96k windows) that is roughly 39 min per epoch before data loading, on one GPU: about 26 h for 40 epochs (pose) and up to 65 h for 100 epochs (vel), so a 48 h limit may not hold for vel without the kernel. The kernel should be much faster; not measured.
7. **Compare** against `st_vel_transformer` / `st_pose_transformer` and the R1 baseline under the same optimiser (see `TRANSFORMER_PROGRESS.md` section 3 step 2: the AdamW baseline still has to be run). R1 reference: local val 0.231 rad, test 0.287 rad, Codabench 16.60 deg.
8. **Submission.** Write `NIPS_CHALLENGE_2026/submission/ca_emgmamba_{pose,vel}/submission.py` following the skill, section 3. Copy the model classes verbatim, drop the `mamba_ssm` import and force the pure-PyTorch scan (Codabench may not have the package; inference under `torch.inference_mode()` has no memory problem). Solver names `CaEmgMambaPose`, `CaEmgMambaVel`. Run the strict-load and timing checks from the skill.

## 4. Decisions made in the 2026-11-06 design discussion, and why

- **No EmgPreproc front end.** Jerry's choice: a per-channel normalisation layer replaces the FIR band-pass, notches and bad-channel rejection. Known risk: StTcnPose R0 (filters *and* normalisation off) scored 19.41 deg vs 16.98 deg. This model keeps the normalisation, so R0 does not predict its result; a with/without-filter comparison is still worth one run.
- **No spatial block, nothing from EEGNet.** Jerry's reasoning: EEG electrodes map to brain regions and HD-EMG grids capture MUAP propagation, but the armband is a ring of interference-EMG electrodes, so a depthwise spatial convolution is not justified. The 16 electrodes are plain input channels of the first `Conv1d`. An earlier proposal (shared per-electrode stem + ring-aware mixing) was dropped at his request.
- **Normalisation is a manual z-score, not `nn.LayerNorm`.** Same operation as LayerNorm over time without affine weights; epsilon is added to the standard deviation so the output does not depend on the amplitude unit. `nn.LayerNorm([T])` with affine weights would learn 10,000 per-sample weights and fix the window length.
- **SE by default, ECA as an option.** After the CNN the channels are learned features with no neighbour order, so ECA's local convolution across them is arbitrary.
- **Gate after every stage by default.** A gate only at the end sits right before Mamba, which has its own input-dependent gating, so it is the most redundant position. Stage 1 is the most useful single position. `ca_stages` makes the four placements one-line ablations.
- **Mamba instead of a transformer for the temporal encoder.** Jerry's choice for this model. At 250 tokens attention is cheap, so Mamba has no cost advantage; its case is a higher token rate (ablation C: 200 Hz) or, for a future paper, streaming / variable-length inference.
- **Original Mamba, not Mamba-2 or Mamba-3.** Later generations are validated on language models only, and the diagonal recurrence is the simplest to reimplement for a submission without `mamba-ssm`.
- **Mamba Modulation (Lu et al., NeurIPS 2025) is not used.** It rescales the transition matrix A so a trained Mamba handles inputs longer than training. The challenge uses fixed 5-s windows for training and inference, so there is no length mismatch. It is a candidate for the future variable-length paper.
- **Bidirectional = two blocks with separate weights, averaged.** Mamba is causal; the challenge allows bidirectional processing inside a window.
- **Same loss and optimiser as the transformer YAMLs**, so the comparison isolates the architecture.

### Added 2026-11-09: time-stretch augmentation (R4)

`CaEmgMamba` has two new config fields, `time_stretch_prob` (default 0.0 = off, so every
earlier YAML is unchanged) and `time_stretch_range` (default `(0.8, 1.2)`). When on, the
model exposes `train_batch_transform` (the hook `pl_module._run_step` already calls for
C-Mixup) and `TimeStretch` resamples EMG and targets together about the window centre by a
per-window factor; targets outside the window become NaN and are masked. No parameters, so
R0 checkpoints still load strictly. YAML: `ca_emgmamba_vel_newton_R4.yaml`. Tests:
`test_time_stretch_*` in `test_ca_emgmamba.py`. Rationale and the R0 three-split readout
(heldout_user 12.58 deg, heldout_stage 16.94 deg, user_stage 16.89 deg, Codabench 19.01
deg) are in the tuning plan Google Doc "CaEmgMamba tuning plan (NeurIPS 2026 Track 4)".

## 5. Results log

| Date | Run | YAML (+ overrides) | Commit | params | best epoch | val/mae (rad) | test user_stage (rad / deg) | heldout_user | heldout_stage | Codabench |
|---|---|---|---|---|---|---|---|---|---|---|
| | M-pose | `ca_emgmamba_pose.yaml` | | | | | | | | |
| | M-vel | `ca_emgmamba_vel.yaml` | | | | | | | | |

Measured on the default config, pure-PyTorch scan, random input, one 23.6 GiB slice of an RTX PRO 6000:

| Head | params | train step, batch 128, `scan_checkpoint: true` | peak GPU memory | inference, 64 windows |
|---|---|---|---|---|
| pose | 2,262,212 | 3.08 s | 7.04 GiB | 0.094 s |
| vel | 2,266,072 | 3.09 s | 7.04 GiB | 0.094 s |

Without checkpointing: 11.67 GiB at batch 64 (pose); out of memory at batch 128 on this slice.

## 6. Open questions

- **"Two heads".** Built as two selectable variants on one trunk. If Jerry meant one model emitting both outputs at once, note that `VelocityHead` already contains both readouts; exposing the position readout as a second output (or adding an auxiliary loss on it) is a small change.
- **Is the fast-scan call correct?** Argument order and shapes match the mamba-ssm 2.3.2 source by reading, but the path has never executed. `test_fast_scan_parity` is the check. Whether 2.3.2 builds against torch 2.14 / CUDA 13 is also unknown.
- **Training speed without `mamba-ssm`.** Memory is solved by `scan_checkpoint`; speed is the remaining cost (section 3, step 6).
- **LayerNorm vs RMSNorm in the Mamba layers.** LayerNorm was used for compatibility with older PyTorch in the submission environment. Not expected to matter.
- **No `causal` switch.** `bidirectional: false` makes the encoder causal, but the stem convolutions are centred and the upsample interpolates. Add a switch only if a causal model is needed.
- **`fuse_gain`, `position_steps` and `vel_scale` assume 50 Hz tokens.** Rescale them if the token rate changes (ablation C).

## 7. File map

```
neuraltrain-repo/neuraltrain/models/ca_emgmamba.py        model
neuraltrain-repo/neuraltrain/models/test_ca_emgmamba.py   tests
neuralbench-repo/neuralbench/models/ca_emgmamba_{pose,vel}.yaml          local configs
neuralbench-repo/neuralbench/models/ca_emgmamba_{pose,vel}_newton.yaml   + SLURM infra block
tutorial/CA_EMGMAMBA_PROGRESS.md                          this file
```

Related docs: `tutorial/TRANSFORMER_PROGRESS.md` (the six transformer configs, none trained yet), `tutorial/PROGRESS.md` (team scoreboard), `tutorial/neuralbench_slurm_infra.md`, `.claude/skills/emg-pose-neuralbench-pipeline/SKILL.md`.
