# StPoseTransformer and StVelTransformer: architecture and design rationale

*Added 2026-11-03. Files: `neuraltrain-repo/neuraltrain/models/st_pose_transformer.py`, `st_vel_transformer.py`, `test_st_transformer_pose.py`; configs `neuralbench-repo/neuralbench/models/st_pose_transformer.yaml`, `st_vel_transformer.yaml`. Both register through `models/__init__.py` and train with `neuralbench emg pose -m st_pose_transformer` / `-m st_vel_transformer`.*

## 1. What these two models are

Two models that share one trunk and differ only in how the last layer turns per-frame features into a joint-angle trajectory:

| | `StPoseTransformer` | `StVelTransformer` |
|---|---|---|
| readout | one absolute angle vector per 50 Hz frame | a position **and** a velocity per frame |
| trajectory | the readout itself (optionally a 5-tap learned smoother) | integrated from the velocity; pulled towards the position readout by a complementary filter |
| continuity prior | none | built in (integration) |
| drift | none | possible when `fuse_gain = 0`; corrected when `fuse_gain > 0` or with the GRU state |
| extra knobs | `head_kind`, `head_smooth_k` | `position_steps`, `vel_scale`, `fuse_gain`, `vel_gru_hidden`, `vel_gru_layers` |

The trunk replaces StTcnPose's TCN with a Conformer-style sequence encoder and adds two optional, cheap modules in front of it (an electrode gate and a multi-scale temporal filter bank). Every new module can be switched off in the YAML, so each one is a one-line ablation against the R1 trunk.

## 2. Data flow

```
EMG (B, 16, 10000)  2 kHz, 5-s window, any amplitude scale
 │
 ├─ EmgPreproc (no parameters)             FIR 20-450 Hz + 60/120/180 Hz notches, bad-channel
 │                                         rejection + ring interpolation, per-window z-score
 │                                         per channel  → (B, 16, 10000)
 ├─ ChannelGate   [channel_attention]      2*sigmoid(MLP(log RMS, log MAV per channel)),
 │                                         one gate per window (per sample when causal)
 │                                         → (B, 16, 10000), identity at init
 ├─ MsSpatialBlock                         multi-scale temporal bank (f1 = 32 filters split over
 │     ├ temporal bank  [ms_kernels]       15/31/63/127 taps = 7.5-63 ms) → BN
 │     ├ 2 × ring conv (3 × 9)             circular padding over the 16-electrode band
 │     ├ depthwise collapse over C, ELU    (16, 1) kernel per temporal map, ×2 depth
 │     ├ AvgPool ÷5  → 400 Hz
 │     ├ separable conv (16 taps) → f2 = 128
 │     └ AvgPool ÷8  → 50 Hz               → (B, 128, 250)   one token per 20 ms
 ├─ SequenceEncoder                        1×1 proj → 4 × ConformerBlock(d 128, 4 heads, FFN ×4,
 │                                         depthwise conv 15 frames = 300 ms, RoPE) → LN
 │                                         → (B, 128, 250)
 ├─ readout
 │     StPoseTransformer: Conv1d(128 → 20)                    → (B, 20, 250)
 │     StVelTransformer : Conv1d(128 → 20) position, Conv1d/GRU+Linear → velocity,
 │                        integrate with complementary fusion → (B, 20, 250)
 ├─ upsample ×40 (linear; zero-order hold when causal)        → (B, 20, 10000)
 └─ transpose                                                 → (B, 10000, 20) radians, time-major
```

Parameter count with the shipped YAMLs: **1.62 M** (spatial block 0.07 M, encoder 1.55 M, gate 1.6 k). StTcnPose R1 is ~0.4 M; VEMG2Pose ~6 M. CPU forward for 8 windows ≈ 2.1 s, so the whole Codabench test set is minutes on an A100, far under the 60-min budget.

## 3. Why each piece is there

### 3.1 Front end (unchanged from R1)
R0 vs R1 (19.41° with the per-window scaling off, 16.60° with it on) is the strongest single result on this project, so the front end is carried over verbatim: FIR band-pass + notches, per-window per-channel z-score, and the bad-channel rejection with ring interpolation. The z-score is what makes the model unit-invariant between the NeuralBench cache (µV, ~1e5) and the Codabench server (~1e2); a transformer with LayerNorm everywhere would survive a scale change better than BatchNorm did, but there is no reason to find out on a sealed-phase upload.

### 3.2 ChannelGate (SE-style channel attention)
Band placement and rotation differ between users, and the test users are unseen. A squeeze-and-excitation gate over the 16 electrodes lets the model re-weight channels *per window* from two scale-free statistics (log RMS, log mean absolute value of the already-normalised signal). It is 1.6 k parameters and starts as the identity (last layer zeroed, `2*sigmoid(0) = 1`), so turning it on cannot hurt at initialisation; whether it helps is ablation B. It is deliberately a gate on the *input* channels, not attention over channels inside the encoder: the EEGNet-style depthwise collapse already mixes electrodes spatially, and the thing that varies across users is which electrodes carry signal, which is exactly what a per-window gate can express.

Note that the task config already applies emg2pose's band-rotation augmentation (`band_offsets [-1, 0, 1]`), so the gate is not a substitute for that; they target the same cross-user problem from the data and the model side respectively.

### 3.3 Multi-scale temporal bank (from MSST, adapted)
The one module in Lin et al. (2026, *Measurement* 257:118953) whose removal hurt most in their ablation is the parallel multi-kernel convolution. Here it replaces the single 65-tap temporal bank at the front of the spatial block: the 32 filters are split evenly over four kernel lengths and the outputs are concatenated. The kernel lengths are **not** the paper's 1/3/5/7: those were chosen for 500 Hz data, and a length-1 kernel is a scaling, not a filter. At 2 kHz, 15/31/63/127 taps span 7.5–63 ms, from single motor-unit bursts to activation envelopes. Setting `ms_kernels: null` restores R1's single bank (ablation A). Everything downstream of the bank (ring convs, collapse, pooling) is the R1 `SpatialBlock` unchanged, which is why `MsSpatialBlock` subclasses it and only swaps `temporal`.

### 3.4 Conv-subsampling before attention
The encoder never sees 2 kHz samples. The spatial block pools ÷40 to 50 Hz, so a 5-s window is 250 tokens, each already a 128-d summary of 20 ms across all 16 electrodes. This is the conv-subsampling pattern of speech Conformers and it does two things: it makes attention cheap (250² instead of 10 000²), and it means the encoder models muscle-activation dynamics at the tempo of hand motion rather than the EMG waveform itself. The TCN in R1 sat at the same 50 Hz rate, so the comparison is clean: same tokens, different sequence model.

### 3.5 SequenceEncoder: Conformer blocks, pre-norm, RoPE
- **Why Conformer rather than a vanilla transformer.** Each block is ½ FFN → self-attention → depthwise conv (15 frames, 300 ms) → ½ FFN → LayerNorm (Gulati et al. 2020). For biosignals the local conv branch usually does most of the work and attention adds global mixing; it is the natural successor to a TCN, which is all local conv. `block_style: transformer` drops the conv module and the macaron FFN to give a plain pre-norm transformer (ablation C). If C scores the same as the Conformer, the attention is doing the work; if it scores like R1, the conv module was.
- **Pre-norm.** LayerNorm goes *before* attention and FFN (MSST's equations are post-norm). Pre-norm trains stably at this scale without a carefully tuned warmup, and the shipped optimiser is deliberately mild.
- **Rotary positions.** Positions enter through RoPE on the query/key vectors rather than an additive absolute encoding, so the model is not tied to the 250-token training length and relative offsets mean the same thing anywhere in the window. `pos_enc: sinusoid` is the additive alternative; `none` relies on the conv modules alone for order.
- **Width/depth.** 4 layers × 128-d × 4 heads × FFN 512 ≈ 1.5 M parameters. Lesson 5 in `PROGRESS.md` ("width before depth": doubling R1's widths gave −0.36°) argues for capacity, lesson 6 ("architecture has plateaued", the gap is train-users vs test-users) argues against a lot of it. 1.5 M is the compromise: a few times R1, a quarter of VEMG2Pose. `n_layers: 2 / 6` is ablation D.
- **Context.** Attention is unrestricted within the window (±2.5 s), against the TCN's ±0.85 s receptive field. `attn_window: 25` limits it to ±0.5 s (ablation E); if that scores the same, the long context is not what helps. With `causal: true` the mask also removes the future.

### 3.6 Causality
Unchanged in meaning from StTcnPose: `causal: true` left-pads every convolution (bank, ring convs, separable conv, Conformer conv module), switches the front end to expanding-window scaling, masks attention to the past, makes the channel gate per-sample from running statistics, and uses a zero-order hold for upsampling. The unit test perturbs the input after 2.4 s and checks nothing before it changes. The competition does not require causality and the non-causal setting scores better; the flag exists so the same checkpoint can be evaluated online later.

### 3.7 The two readouts

**Position readout (`StPoseTransformer`).** A 1×1 conv reads 20 angles out of every frame independently. No drift, anchored every frame, but no continuity prior: the output can be jittery, and on low-SNR slow movements the model tends to regress towards the mean posture. `head_kind: smooth` adds a learned 5-tap depthwise filter per joint (moving average at init) as a minimal continuity prior.

**Velocity decoder (`StVelTransformer`).** VEMG2Pose's hybrid parameterisation, written out without the LSTM:

```
pose_t = position_t                                                   t < position_steps (12 frames = 240 ms)
pose_t = (1 − g) · (pose_{t−1} + vel_scale · velocity_t) + g · position_t     otherwise
```

- `g = 0` is pure integration: smooth and good on transients because the head only has to get the *change* right, but a memoryless velocity head cannot know it has drifted, so over 5 s the integrated angle walks away from the truth. This is the failure the baseline's LSTM state exists to prevent.
- `g = 1` collapses to the position readout (the unit test checks this exactly), i.e. `StPoseTransformer`.
- `0 < g < 1` is a complementary filter: a first-order IIR in which the velocity head governs everything faster than ~1/g frames and the position head anchors everything slower. With `g = 0.1`, drift is forgotten with a ~200 ms time constant while 20-ms-scale motion still comes from the velocity. Both heads receive gradient through the fused output, so no auxiliary loss and no pl_module patch is needed.
- `vel_gru_hidden > 0` puts a GRU over the encoder frames before the velocity readout. This is the "recurrent state tracks the pose" argument from VEMG2Pose, kept cheap: one vectorised `nn.GRU` call (cuDNN), no per-frame Python loop, no pose feedback. It can be combined with `g`.
- The velocity readout is zero-initialised (as in `PosePost`), so an untrained model holds the position instead of random-walking. The unit tests undo that init before checking causality, otherwise a held pose cannot respond to a perturbation.

The reason for two models instead of one flag: the comparison *is* the experiment. Everything up to the readout is identical, so P (pose) vs V0 (`g = 0`) vs V1 (`g = 0.1`) measures how much the metric rewards smoothness versus anchoring, with no confound. StLstmPose v1 (17.06°) could not answer this, because it changed the encoder (causal LSTM) and the decoder at the same time.

### 3.8 Loss and optimiser
- **OHEM** (`keep_fraction 0.3`, `easy_weight 0.1`, per-frame ranking by joint-mean L1) is the loss that won the hard-example-mining ablation and is used for both models. For the velocity model the mining ranks frames by the error of the *integrated* trajectory, which is what the metric scores.
- **AdamW + OneCycleLR** instead of the task's constant Adam 1e-3: 10 % linear warmup to 3e-4, cosine decay, weight decay 0.05, gradient clip 1.0, β₂ 0.98. Transformers without warmup at the TCN's learning rate tend to train slowly rather than diverge, which would make the encoder look worse than it is. `total_steps` is injected by the trainer from `estimated_stepping_batches`.

### 3.9 What was deliberately left out of MSST
- **VMD.** Five-mode decomposition per channel multiplies the input by five to compensate for a 3-channel, 7-subject dataset. emg2pose has 16 channels and 25 k recordings; the multi-scale bank gives the encoder the same kind of frequency-split input as learned filters, at 2 kHz resolution, for free.
- **GHM.** Already tried (17.53° vs 17.23° for its L1 control); OHEM replaced it.
- **BiLSTM between the conv and the transformer.** MSST stacks CNN → BiLSTM → Transformer. Here the encoder *replaces* the sequence model rather than being stacked on it; StLstmPose already covers the LSTM route, and stacking all three only adds parameters to a model whose bottleneck is cross-user generalisation, not capacity.
- **Post-norm Transformer with absolute encodings.** See 3.5.

## 4. Experiment plan

One change per run, evaluated with `--dataset heldout_user` (and `heldout_stage`) locally before a Codabench upload; runs so far differ by 0.2–0.4° and the noise floor is not yet known.

| Run | Config | Question |
|---|---|---|
| Baseline | `st_tcn_pose_ohem.yaml` with R1 widths **and the AdamW/OneCycle block from these YAMLs** | the honest comparison point |
| P | `st_pose_transformer.yaml` as shipped | encoder swap + gate + bank, position head |
| V0 | `st_vel_transformer.yaml`, `fuse_gain: 0.0` | pure integration |
| V1 | `st_vel_transformer.yaml` as shipped (`fuse_gain 0.1`) | complementary fusion |
| V2 | V1 + `vel_gru_hidden: 128` | recurrent state on the velocity head |
| A–E | P with `ms_kernels: null` / `channel_attention: false` / `block_style: transformer` / `n_layers: 2, 6` / `attn_window: 25` | which trunk additions earn their keep |

Expected: V1 ≈ V2 > P > V0, with the P-vs-V0 gap telling how much the metric rewards smoothness versus anchoring. If P ≈ Baseline, the TCN's inductive bias is adequate for this data and the conclusion is to spend the remaining uploads on training/data (C-Mixup, `min_finite_target_fraction 0.5`), not on the encoder. That is a legitimate result, not a failed experiment.

## 5. Submission notes
- `submission.py` needs verbatim copies of `ChannelGate`, `_MultiScaleTemporal`, `MsSpatialBlock`, `SpatialBlock`, `_RingConv2d`, `_same_pad`, the encoder classes, `PosePost` (pose model) or `VelocityDecoder` (velocity model), and `upsample_frames`, with the same class and attribute names so the state dict matches. Drop `frontend.*` keys on load as before; the FIR buffers are non-persistent.
- `CONFIG` in the submission mirrors the YAML's `brain_model_config`; `extract_weights_rev1.py` embeds it. Keep the two YAMLs' trunk sections identical when editing one.
- The velocity decoder with `fuse_gain > 0` runs a 238-step Python loop over frames on tensors of shape (B, 20); it is negligible next to the encoder (measured: no difference in CPU time between `g = 0` and `g = 0.1`).
- Record commit hash, YAML and seed with every zip (top-3 audit).
