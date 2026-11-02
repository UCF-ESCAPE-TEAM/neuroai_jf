---
name: emg-pose-neuralbench-pipeline
description: End-to-end workflow for NeurIPS 2026 EEG/EMG Challenge Track 4 (EMG-to-pose) in the UCF ESCAPE team setup — add and register a custom model in the neuroai_jf fork (NeuralTrain + NeuralBench), train it (locally or on the SLURM cluster), read its metrics, extract weights.pt from best.ckpt, write a self-contained Codabench submission.py, and test/zip it with run_submissions.sh. Use whenever the user mentions neuralbench, neuraltrain, emg2pose, Track 4, submission.py, weights.pt, best.ckpt, Codabench, metrics.csv, heldout_user/heldout_stage, min_finite_target_fraction, or the 2026-competition benchopt harness.
---

# EMG-to-pose pipeline (NeuralBench → Codabench)

*Updated 2026-11-02. Read `tutorial/PROGRESS.md` first for the scoreboard, what has been tried, and the ranked plan; this file is the how-to.*

Five procedures, in the order a new experiment goes through them. Repos:

| Repo | Role |
|---|---|
| `github.com/UCF-ESCAPE-TEAM/neuroai_jf`, branch **`emg-pose-cnn`** | fork of `facebookresearch/neuroai`; `neuraltrain-repo/` holds model code, `neuralbench-repo/` holds training configs and the `neuralbench` CLI |
| `github.com/UCF-ESCAPE-TEAM/NIPS_CHALLENGE_2026` (`main`) | team scripts: `prepare_submission/extract_weights_rev1.py`, `shell/run_submissions.sh`, `submission/<model>/submission.py` (one folder per model, `weights.pt` git-ignored) |
| `2026-competition` (official harness, cloned locally) | `tracks/emg_pose/`, `benchopt run`, `benchmark_utils.base_solver.CompetSolver` |

Install once per machine: `pip uninstall -y neuralbench neuraltrain`, then from the fork root `pip install -e neuraltrain-repo && pip install -e 'neuralbench-repo/.[dev]' && pip install 'eegdash>=0.8.2'`; verify `python -c "import neuraltrain, neuralbench; print(neuraltrain.__file__, neuralbench.__file__)"` paths point into the clone, not `site-packages`. `~/.neuralbench/config.json` (USER, ENTITY_NAME, PROJECT_NAME, DATA_DIR, CACHE_DIR, SAVE_DIR, CLUSTER) is per machine and not in git. Full install notes: `tutorial/neuroai-fork-install-and-update.md`.

## The contract you are building for

- `predict(X)`: `X` is `(B, 16, 10000)` float32 EMG at 2000 Hz (5-s windows, microvolt-ish scale on the server, but **do not rely on the scale**), return `(B, 20, 10000)` joint angles in **radians**. Windows are independent, batch order meaningless, bidirectional processing inside a window is allowed, TTA allowed.
- Budget: 60 min on one A100 for the whole test set. Scored by MAE in degrees on the `user_stage` partition (unseen users doing unseen stages), with the organizers' own windowing (IK-failure-free windows only).
- NeuralBench models return **time-major** `(B, T, n_outputs)`; the submission wrapper transposes to `(B, J, T)`.
- Modifying the scoring harness = disqualification. Only `submission.py` + `weights.pt` (+ small extra files) go in the zip.
- Sealed phase Oct 28 – Nov 21 2026, **1 submission/day**; top-3 get a reproducibility audit, so record fork commit hash + YAML + seed with every zip.

## 1. Add and register a model in the fork

Three files; keep everything in **new files** so upstream merges stay clean.

### 1a. `neuraltrain-repo/neuraltrain/models/<my_model>.py`

```python
from torch import nn
import torch.nn.functional as F
from .base import BaseBrainModelConfig          # pydantic config, discriminated on class name


class MyPose(BaseBrainModelConfig):             # <- the YAML's `name:` must equal this class name
    hidden: int = 128                           # every hyper-parameter is a typed field with a default
    dropout: float = 0.1

    # Parameter names must match BrainModelBuildContext fields; the factory injects by name.
    def build(self, n_spatial_locations: int, n_outputs: int | None) -> nn.Module:
        assert n_outputs is not None
        return MyPoseModel(n_spatial_locations, n_outputs, **self.model_dump(exclude={"name"}))


class MyPoseModel(nn.Module):
    MODEL_FIELDS = ("hidden", "dropout")        # keep: the submission rebuilds from these names
    def __init__(self, n_in_channels, n_outputs, hidden=128, dropout=0.1): ...
    def forward(self, x):                       # x: (B, C, T)
        y = ...                                 # (B, n_outputs, T') at a lower rate
        y = F.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)
        return y.transpose(1, 2)                # (B, T, n_outputs)  <- time-major
```

Rules learned the hard way:
- Put a **per-window normalisation** inside the model (`EmgPreproc` from `emg_frontend.py`: `norm: maxabs|minmax|robust|zscore|none`, `scope: channel`). A model trained on NeuralBench's amplitude and shipped without it produced a constant pose on Codabench (19.4° vs 16.7° local). BatchNorm on raw input is *not* enough.
- Non-persistent buffers (FIR kernels etc.) must be rebuilt from config, never loaded from the checkpoint; `submission.py` drops `frontend.*` keys for that reason.
- A model-level `causal` flag is optional; the challenge does not require causality and non-causal models score better.
- Reuse the existing trunk where possible: `StTcnPose` (`st_tcn_pose.py`) is the best model so far; `StTcnPoseCMixup` and `StLstmPose` subclass/reuse its front end + `SpatialBlock`. Loss-only experiments need **no new model class** — see 1c.

### 1b. `neuraltrain-repo/neuraltrain/models/__init__.py`

Append one line (the import registers the pydantic subclass):
```python
from . import my_pose as _my_pose  # noqa: F401
```

### 1c. `neuralbench-repo/neuralbench/models/<my_model>.yaml`

```yaml
brain_model_config:
  =replace=: true
  name: "MyPose"            # class name from 1a
  hidden: 256               # any config field; omitted ones use the Python defaults
delete_checkpoints_on_exit: false   # REQUIRED or best.ckpt is deleted when the run ends
csv_config: {}                       # writes lightning_logs/version_0/metrics.csv under the run dir
```
Optional top-level overrides in the same file (all YAML-only, each changes the config hash so rerun with `-f`):
- `trainer_config: {n_epochs, patience, monitor, mode, precision, gradient_clip_val}` — `monitor`/`mode` drive **both** EarlyStopping and which epoch becomes `best.ckpt` (`main.py:353-373`). Task default: `monitor: val/mae, mode: min, patience: 20, n_epochs: 100`. Monitorable keys: `val/loss`, `val/mae`, `val/mae_subject_mean`, `val/mae_subject_std`, `val/rmse`, `val/r2_score` (needs `mode: max`). A misspelled key fails only at the first validation end, so smoke-test with `--debug`.
- `loss: {name: GhmRegressionLoss | OhemRegressionLoss, ...}` — replaces the task's L1 (`neuraltrain/losses/hard_example.py`); templates in `st_tcn_pose_ghm.yaml`, `st_tcn_pose_ohem.yaml`.
- `lightning_optimizer_config: {optimizer: {lr}}`, `data: {batch_size}`.
- `data: {min_finite_target_fraction: 0.5}` — see §1e.
The YAML stem (`my_model`) is the CLI name.

**Trunk-width caution.** `st_tcn_pose.yaml`, `_ghm`, `_ohem` and `_cmixup` on the branch carry the *original* widths (`f1 16, f2 64, tcn_filters 64`) and `norm: maxabs`. The best run (R1, 16.60°) used `f1 32, f2 128, tcn_filters 128` and `norm: zscore`. Any experiment meant to be compared with R1 must set those four values explicitly, and `CONFIG_YAML` in the extraction step must be the YAML that was actually trained.

### 1d. Train

```bash
neuralbench emg pose -m my_model --debug      # smoke test: tiny config, forces a rerun
neuralbench emg pose -m my_model              # full run; -f forces a rerun of a finished config
neuralbench emg pose -m my_model --dataset heldout_user    # test set = unseen users, seen stages
neuralbench emg pose -m my_model --dataset heldout_stage   # test set = seen users, unseen stages
```
`--dataset <stem>` merges `tasks/emg/pose/datasets/<stem>.yaml` over the task config; the two variants (added Nov 1 2026) only change the `QueryEvents` filter on the **test** split, so train/val and the cache are unchanged — use them to split the Codabench-style `user_stage` number into its two generalisation axes. Default (no flag) is `user_stage`, the Codabench partition.

Results land under `SAVE_DIR/neuralbench.main.Experiment.run,1/<long hash dir>/` with `best.ckpt`, `.out` (pre-training "Dataset summary" and validate table, per-epoch lines), `.err` (EarlyStopping "Metric val/mae improved" lines, `Best epoch`) and `lightning_logs/version_0/metrics.csv` (per-epoch `train/loss`, `val/mae`, loss-specific diagnostics such as `train/loss_ghm_frac_top`, `train/loss_ohem_hard_mean`, `train/mix_partner_dist`, `train/mix_lambda_mean`). Find a run's dir by grepping the hash dir name for the model name and loss name.

Current task config: patience 20, n_epochs 100, lr 1e-3 held, batch 128, seed 33 only (`defaults/grid.yaml` trimmed), `data.neuro.scaler: StandardScaler` (×1e6), `band_offsets [-1,0,1]` rotation augmentation, L1 loss.

Stopping a run: **Ctrl+C is graceful** — Lightning runs the test stage and then (if `delete_checkpoints_on_exit` is true) deletes the checkpoint; `kill -TERM <pid>` / `scancel` keeps `best.ckpt`. Find the pid with `ps aux | grep neuralbench`.

**Cluster** (`/lustre/fs1/home/ji045311/challenge_2026/{Data,Cache,Result,neuroai_jf}`, see `tutorial/slurm_tutorial.md` in the project): never launch from the login node (torch import hits the address-space cap); use an interactive compute node (`evc29`-style) or an sbatch job with `CLUSTER: null`; `module purge` first, then `module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0`, `conda activate challenge_2026`. Partition `normal` (GPU nodes evc22-50), also `preemptable`, `short`, `highgpu`.

### 1e. Data: recover the IK-failure windows (found Nov 1 2026)

emg2pose marks mocap IK failures as `BAD_IK` (angles NaN in NeuralBench). `pl_module._run_step` already masks NaN frames out of the loss and every metric (upstream PR #229, v0.3.0), so partially-labelled windows are safe to train on. The task config nevertheless sets `data.min_finite_target_fraction: 1.0` (mirrors emg2pose's `skip_ik_failures`), which discards every window overlapping a failure: on the real dataset that drops **44 % of labelled training frames** (train windows 96k → 180k at 0.5) and the loss is concentrated in the fast, large-amplitude stages (Throwing ~95 % dropped, Punch ~77 %) — exactly the stages the model generalises worst on. One line in the model YAML restores them:

```yaml
data:
  min_finite_target_fraction: 0.5
```
It is part of the config hash (new cache dir, re-segmentation takes a while; run with `-f`). Caveats: (a) it applies to **every** split, so local val/test numbers get slightly harder and are no longer comparable to 1.0 runs or the paper's tables — judge the run on Codabench against R1, or patch `data.py` to apply the threshold to train only (`keep |= split != 'train'` before `dataset.select(keep)`); (b) the Codabench test set is windowed by the organizers at 1.0, so this cannot leak or inflate the leaderboard score; (c) per-frame losses (L1, GHM, OHEM, C-Mixup) are unaffected by label gaps; any *temporal* loss must mask the differences before flattening. Left/right-hand EMG polarity needs no handling (flip already applied in the released HDF5; BIDS conversion only rescaled units).

## 2. Extract `weights.pt` from `best.ckpt`

Use `NIPS_CHALLENGE_2026/prepare_submission/extract_weights_rev1.py`. Edit the three paths at the top, then `python extract_weights_rev1.py`:

| Variable | Set to |
|---|---|
| `CKPT_PATH` | the run's `best.ckpt` |
| `OUT_PATH` | `<submission_root>/<model_folder>/weights.pt` |
| `CONFIG_YAML` | the training YAML from 1c (**the one actually trained**) |

What it does and why: NeuralBench saves with `save_weights_only=True`, so the checkpoint has **no hyper_parameters**. The script strips the Lightning prefix (`model.`), drops probe/loss/metric/EMA sub-trees, reads `brain_model_config` from the YAML and writes `{"state_dict", "config", "source", "epoch"}`. The embedded `config` lets `submission.py` rebuild exactly the trained architecture. Check the printed summary: `kept` tensor count and element count should match the model's parameter count, and `config: from yaml …`. (When automating, prefer turning the three constants into `argparse` arguments rather than editing the file per run.)

## 3. Write `submission.py`

One self-contained file (no imports from neuraltrain), in `NIPS_CHALLENGE_2026/submission/<model_folder>/submission.py`. Existing ones: `st_tcn_pose`, `st_tcn_pose_ghm`, `st_tcn_pose_cmixup`, `st_lstm_pose`, `neuraldrive_pose`, `ulhm_pose`. A loss-only or data-only variant of an existing architecture **reuses that architecture's submission.py unchanged** (e.g. GHM/OHEM/CMixup/min_finite runs of StTcnPose all load into `submission/st_tcn_pose/`); only a new architecture needs a new file. Structure that passes the harness:

```python
"""Codabench submission — <Model>."""
import warnings, numpy as np, torch, torch.nn.functional as F
from torch import nn
from benchmark_utils.base_solver import CompetSolver   # provided by the harness

N_CHANS_TRAINED, N_JOINTS_TRAINED = 16, 20
CONFIG = dict(sfreq=2000.0, hidden=256, dropout=0.1, ...)   # mirror of the training YAML
LEGACY_KEYS = ()                                             # old config names still accepted

# ---- verbatim copies of the model classes from neuraltrain (same class/attr names
#      so state_dict keys match), minus the pydantic config class ----

def build_model(n_chans, n_joints, config):
    cfg = dict(config)
    return MyPoseModel(n_chans, n_joints, **{k: cfg[k] for k in MyPoseModel.MODEL_FIELDS})

def _read_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state, config = (ckpt["state_dict"], ckpt.get("config")) if "state_dict" in ckpt else (ckpt, None)
    if any(k.startswith("model.") for k in state):            # raw best.ckpt also accepted
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    state = {k: v for k, v in state.items() if not k.startswith("frontend.")}  # non-persistent parts
    if config is not None:
        config = {k: v for k, v in config.items() if k in CONFIG or k in LEGACY_KEYS}
    return state, config

class MyPoseRegressor:
    def __init__(self, n_chans, n_joints, config, device):
        self.device, self.config = device, dict(config)
        self.n_chans, self.n_joints = n_chans, n_joints
        self.net = build_model(n_chans, n_joints, self.config).to(device)
    def load(self, path):
        state, config = _read_checkpoint(path, self.device)
        if config and {**self.config, **config} != self.config:    # embedded config wins
            self.config = {**self.config, **config}
            self.net = build_model(self.n_chans, self.n_joints, self.config).to(self.device)
        self.net.load_state_dict(state, strict=True)
        return self
    def fit(self, train_loader): ...                             # only used if benchopt trains
    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if not getattr(self, "_logged", False):                   # one-off scale diagnostic in the log
            print(f"[input] shape {tuple(X.shape)}, |X| mean {X.abs().mean():.4g}, std {X.std():.4g}")
            self._logged = True
        return self.net(X).transpose(1, 2)                        # (B, J, T) radians

class Solver(CompetSolver):
    name = "MyPose"                      # run_submissions.sh reads this; must be unique per model
    requirements = []
    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.pt"
        if self.train_loader is None:    # inference-only (Codabench)
            if not weights.exists(): raise FileNotFoundError(weights)
            model = MyPoseRegressor(N_CHANS_TRAINED, N_JOINTS_TRAINED, CONFIG, self.device).load(weights)
            if meta.get("n_chans") != N_CHANS_TRAINED: raise RuntimeError("channel count mismatch")
            return model
        return MyPoseRegressor(meta["n_chans"], meta["n_joints"], {**CONFIG, "sfreq": float(meta["sfreq"])}, self.device)
    def fit(self, model, train_loader): model.fit(train_loader)
    def save_model(self, model, path):
        torch.save({"state_dict": model.net.state_dict(), "config": model.config}, path / "weights.pt")
```

Checks before shipping (do them in Python, not by eye):
1. Build from `CONFIG`, `load()` the real `weights.pt` with `strict=True`, and run `predict` on `torch.randn(8, 16, 10000) * 100`: shape `(8, 20, 10000)`, finite, **not constant over time** (std over T > 0).
2. Save a NeuralBench-style dict `{"state_dict": {"model."+k: v}}` and confirm `load()` accepts the raw checkpoint too.
3. Time `predict` on 8–64 windows; extrapolate to ~4000 windows and stay far under 60 min on GPU. Sequential decoders (pose-feedback LSTM: 250 steps/window) are the ones to watch; `condition_on_pose: false` is ~20× faster.
4. Keep `CONFIG` and the YAML in sync whenever widths change; the embedded config covers mismatches only for keys present in `CONFIG`.

## 4. Test locally and zip with `run_submissions.sh`

Layout: `SUBMISSION_ROOT/<model_folder>/{submission.py, weights.pt}` — one folder per model. Then, in the competition environment (benchopt installed):

```bash
bash NIPS_CHALLENGE_2026/shell/run_submissions.sh SUBMISSION_ROOT /path/to/2026-competition \
     [--only model_a,model_b] [--dataset Simulated] [--no-zip | --zip-only] [--keep-solvers] [--dry-run]
```
Per folder it: checks both files exist → reads `Solver.name` → copies `submission.py` to `tracks/emg_pose/solvers/<folder>.py` → runs `COMPET_SUBMISSION_DIR=<folder> benchopt run tracks/emg_pose -d Simulated -s <Solver.name>` with the log in `SUBMISSION_ROOT/logs/<folder>.log` → on success zips the folder **flat** into `SUBMISSION_ROOT/zips/<folder>.zip` → prints a summary table; exit 1 if any model failed. Upload each zip to Codabench (1/day in the sealed phase).

## 5. Read out a finished run (before spending the day's submission)

1. `.err`: `Best epoch` and whether EarlyStopping fired on patience or the run hit `n_epochs` (if the latter and val was still falling, extend `n_epochs`/`patience`).
2. `metrics.csv`: train/val gap over epochs; the loss-specific diagnostics (`train/loss_ghm_frac_top` < 0.15 for GHM; `train/loss_ohem_hard_mean` well above `train/loss` for OHEM; `train/mix_partner_dist` clearly below the batch's mean pairwise pose distance and `train/mix_lambda_mean` ≈ 0.5 for C-Mixup).
3. `.out` tail: `test/mae` in radians × 57.3 → degrees; the local `user_stage` test split tracks Codabench to within ~0.2° (R1: 0.287 rad ≈ 16.4° local vs 16.60° Codabench) **as long as `min_finite_target_fraction` is 1.0**.
4. Optional: rerun with `--dataset heldout_user` and `--dataset heldout_stage` (same cache, only the test stage changes) to see which generalisation axis the change helped.
5. Rank candidates by local test MAE; submit the best one; record commit hash + YAML + seed + extraction summary next to the zip.

## Diagnosing a bad Codabench score

| Symptom | Likely cause | Fix |
|---|---|---|
| Codabench ≫ local MAE (e.g. 19.4° vs 16.7°), `[input]` std ~1e2 in the log while training batches were ~1e5 | amplitude scale mismatch; BN running stats saturate | per-window normalisation inside the model; retrain |
| `load_state_dict` strict error | class/attribute names differ between neuraltrain file and submission copy, or a buffer was persisted | copy classes verbatim; drop non-persistent buffers in `_read_checkpoint` |
| unexpected keyword in `__init__` | `CONFIG`/embedded config has a key the submission model does not take | filter config to `CONFIG` keys; keep `MODEL_FIELDS` in sync |
| weights load but score is far off the local number | `CONFIG_YAML` at extraction was not the YAML that was trained (e.g. narrow-trunk YAML for an R1-width checkpoint — this fails loudly; wrong `norm` does not) | re-extract with the right YAML; check `norm` in the embedded config |
| timeout | sequential Python loop per frame | vectorise, or larger batches on GPU |
| `benchopt` can't find solver | `Solver.name` missing or duplicated | unique `name = "..."` on the class |
