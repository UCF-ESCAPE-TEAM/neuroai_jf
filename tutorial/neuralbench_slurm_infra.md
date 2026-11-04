# NeuralBench `infra:` block for the UCF GPU cluster

Paste one of the blocks below at the top level of any
`neuralbench-repo/neuralbench/models/<model>.yaml`. It replaces the defaults in
`neuralbench/defaults/config.yaml` key by key (nested merge), so keys you leave
out keep their default. `infra` is not part of the experiment cache key, so
changing it never invalidates finished results.

## 1 GPU (baselines, reproduces paper settings)

```yaml
infra:
  gpus_per_node: 1
  tasks_per_node: 1
  cpus_per_task: 10          # keep equal to N_CPUS (data.num_workers) in config.json
  mem_gb: 64
  timeout_min: 1440          # 24 h; `normal` has no partition limit
  slurm_use_srun: true
  slurm_additional_parameters:
    exclude: "evc[1-13,15-21,26]"   # stay on evc22-50; drop 26 if it should be allowed
    mail-type: "END,FAIL"
    mail-user: "ji045311@ucf.edu"
  slurm_setup:
    - "unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS"
    - "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK"
```

## 2 GPUs, DDP (own models)

Same block with the first two lines changed:

```yaml
infra:
  gpus_per_node: 2
  tasks_per_node: 2          # one process per GPU; needs slurm_use_srun: true
  cpus_per_task: 10
  mem_gb: 64
  timeout_min: 1440
  slurm_use_srun: true
  slurm_additional_parameters:
    exclude: "evc[1-13,15-21,26]"
    mail-type: "END,FAIL"
    mail-user: "ji045311@ucf.edu"
  slurm_setup:
    - "unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS"
    - "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK"
```

`data.batch_size` is per device, so the effective batch doubles; adjust it or
the learning rate if the run should match a 1-GPU result. Models that use
`unfreeze_at_epoch` need `trainer_config.strategy: ddp_find_unused_parameters_true`
on top.

## What each key does

| Key | Becomes | Notes |
|---|---|---|
| `gpus_per_node` | `#SBATCH --gpus-per-node` and Lightning `devices` | `--debug` and `--prepare` force it to 1 |
| `tasks_per_node` | `#SBATCH --ntasks-per-node` | equal to `gpus_per_node`; needs `slurm_use_srun: true` to launch >1 process |
| `cpus_per_task` | `#SBATCH --cpus-per-task` | dataloader workers come from `N_CPUS` in config.json, keep them equal |
| `mem_gb` | `#SBATCH --mem` | |
| `timeout_min` | `#SBATCH --time` | |
| `slurm_use_srun` | `srun python ...` instead of `python ...` | required for DDP |
| `slurm_partition` | `#SBATCH --partition` | leave out; comes from `SLURM_PARTITION` in config.json (`normal`) |
| `slurm_additional_parameters` | any extra `#SBATCH --<key>=<value>` | long option names without dashes |
| `slurm_setup` | shell lines run in the job before `python` | do **not** put `module purge` here: the anaconda modulefile calls `conda deactivate` on unload, which loops forever in a batch shell |
| `conda_env` | job's python path | leave out; submitit uses the launching env's python automatically |

## How to launch

The login node (`evuser2`) cannot import torch (process and memory limits), so
launch from a compute node:

```bash
srun -p short --pty bash          # or any interactive allocation
module purge && module load anaconda/anaconda-2024.10 && conda activate challenge_2026
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
neuralbench emg pose -m <model>          # add -r to retry a cancelled/failed job
```

`~/.neuralbench/config.json` needs `"CLUSTER": "auto"` (or `"slurm"`) and
`"SLURM_PARTITION": "normal"`. The command blocks until the jobs finish, so run
it inside `tmux`. The generated sbatch script, `.out` and `.err` are under
`SAVE_DIR/neuralbench.main.Experiment.run,1/<experiment>/`.

To run inside a hand-written sbatch file instead, point `NEURALBENCH_CONFIG` at a
copy of config.json with `"CLUSTER": null` (see `vemg2pose_baseline.slurm`); the
`slurm_*` keys are then ignored and the sbatch header supplies the resources, but
`gpus_per_node` still sets Lightning's device count.

## Ready-made cluster YAMLs

`st_pose_transformer{,_ghm,_l1}_newton.yaml` and `st_vel_transformer{,_ghm,_l1}_newton.yaml`
are the transformer configs with the 2-GPU block appended; they share the
experiment cache with their non-`_newton` twins.
