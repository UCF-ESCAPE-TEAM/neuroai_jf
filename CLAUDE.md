# neuroai_jf — UCF ESCAPE team fork for NeurIPS 2026 EEG/EMG Challenge, Track 4 (EMG-to-pose)

Working branch: `emg-pose-cnn`. Upstream: `facebookresearch/neuroai` (keep our changes in new files so merges stay clean).

## Read first
- `tutorial/PROGRESS.md` — scoreboard, what has been tried, lessons, ranked plan, and a list of tasks worth automating (§9). Update it when a run scores or a decision changes.
- `.claude/skills/emg-pose-neuralbench-pipeline/SKILL.md` — the how-to: register a model, train (local / SLURM / `--dataset heldout_user|heldout_stage`), read a run out, extract `weights.pt`, write and test `submission.py`.
- `tutorial/training_tutorial.md`, `tutorial/cmixup_tutorial.md` — why we are working on training/data rather than architecture.
- `tutorial/transformer_pose_models.md` — the Conformer-encoder models (`st_pose_transformer`, `st_vel_transformer`): architecture, rationale, ablation plan.
- `tutorial/TRANSFORMER_PROGRESS.md` — handoff for the transformer thread: what is done, exact next commands, results log. Start here if you are continuing that work.

## Layout
- `neuraltrain-repo/neuraltrain/models/` — our models: `emg_frontend.py`, `st_tcn_pose.py` (best, R1 = 16.60° Codabench), `st_tcn_pose_cmixup.py`, `st_lstm_pose.py`, `st_pose_transformer.py`, `st_vel_transformer.py`, `emg_vel_transformer.py` (vel model with a raw-signal channel gate), `neuraldrive_pose.py`, `ulhm_pose.py`. Register new ones in `models/__init__.py`.
- `neuraltrain-repo/neuraltrain/losses/hard_example.py` — GHM-R / OHEM losses.
- `neuralbench-repo/neuralbench/models/*.yaml` — training configs (YAML stem = CLI name). `tasks/emg/pose/config.yaml` is the task; `tasks/emg/pose/datasets/` holds test-set variants.
- Submission packaging lives in the sibling repo `UCF-ESCAPE-TEAM/NIPS_CHALLENGE_2026` (`submission/<model>/submission.py`, `prepare_submission/extract_weights_rev1.py`, `shell/run_submissions.sh`).

## Hard rules
- Every training YAML needs `delete_checkpoints_on_exit: false` and `csv_config: {}`.
- Models must normalise per window inside `forward` (the `EmgPreproc` front end); never rely on input scale.
- `st_tcn_pose{,_ghm,_ohem,_cmixup}.yaml` still have the narrow trunk (`f1 16, f2 64, tcn_filters 64`, `norm: maxabs`); R1 is `32/128/128`, `zscore`. Say which trunk a run used, every time.
- `extract_weights_rev1.py` must be pointed at the YAML that was actually trained.
- Sealed phase: 1 Codabench upload per day until Nov 21 2026. Record commit hash + YAML + seed with every zip (top-3 audit).
- On the SLURM cluster never launch from the login node; `module purge` before `sbatch`.
- On Newton use the `*_newton.yaml` configs (they carry the `infra:` SLURM block; `tutorial/neuralbench_slurm_infra.md` explains the keys).
- Don't edit the benchopt harness (`2026-competition`) — disqualification.
