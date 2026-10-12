# Installing and updating a forked `neuroai` (NeuralTrain + NeuralBench)

This tutorial covers how to install NeuralTrain and NeuralBench from your own fork of
`facebookresearch/neuroai`, and how to keep every machine up to date when you add new
models or training code.

Replace `YOUR_USERNAME` with your GitHub username and `emg-pose-cnn` with your branch name
throughout.

> **Status of these instructions.** The NeuralBench commands come from its installation
> page. The `neuraltrain-repo` editable install and the folder paths are inferred from the
> repository layout and have not been run end to end. Step 3.4 verifies them on your machine.

---

## 1. How the pieces fit

- **One repository, several packages.** `neuroai` is a monorepo. NeuralTrain lives in
  `neuraltrain-repo/` and NeuralBench in `neuralbench-repo/`. You fork once and get both.
- **Two ways to install from a fork:**

| | A. Clone + editable install | B. `pip install` from the GitHub URL |
|---|---|---|
| Where the code lives | A folder you cloned | A frozen copy in `site-packages` |
| Picking up your changes | `git pull` (no reinstall) | Push, then reinstall |
| Best for | Any machine where you edit or train regularly | A throwaway machine or container |

Use **A** unless you have a reason not to. Section 7 covers **B**.

---

## 2. One-time setup: fork and clone

### 2.1 Fork

On GitHub, open `https://github.com/facebookresearch/neuroai` and click **Fork**.

> **Privacy.** A fork of a public repository is public, and GitHub does not let you make it
> private. If your model code should stay private during the competition, create an empty
> **private** repository instead and push a clone of `neuroai` to it. Everything below works
> the same, with your private repository's URL.

### 2.2 Clone and create a branch

```bash
git clone https://github.com/YOUR_USERNAME/neuroai.git
cd neuroai
git remote add upstream https://github.com/facebookresearch/neuroai.git
git checkout -b emg-pose-cnn
```

- `origin` is your fork (you push here).
- `upstream` is the original repository (you pull updates from here).

---

## 3. Install from the fork

### 3.1 Prerequisites

- Python 3.12 or newer.
- For training: an NVIDIA GPU with a driver that matches your `torch` build. Training has no
  CPU fallback, and that includes `--debug` runs.

### 3.2 Remove any existing install

If you added or edited files inside an old `site-packages` copy, copy them into the fork
first. Then:

```bash
pip uninstall -y neuralbench neuraltrain
```

Leave `neuralset`, `neuralfetch`, `torch` and the other dependencies installed.

### 3.3 Editable install

From the root of the cloned `neuroai` folder:

```bash
pip install -e neuraltrain-repo
pip install -e 'neuralbench-repo/.[dev]'
pip install 'eegdash>=0.8.2'        # needed for the emg pose dataset
```

Install `neuraltrain` first, so NeuralBench uses your local copy instead of downloading the
released one from PyPI.

If the code later reports that `neuralset` or `neuralfetch` is too old, install those from
the checkout as well:

```bash
pip install -e neuralset-repo -e neuralfetch-repo
```

### 3.4 Verify

```bash
python -c "import neuraltrain, neuralbench; print(neuraltrain.__file__); print(neuralbench.__file__)"
neuralbench --help
```

Both printed paths must point inside your cloned `neuroai` folder. If either contains
`site-packages`, that package is still the PyPI copy: uninstall it and repeat step 3.3.

---

## 4. Configure each machine (once per machine)

NeuralBench reads its paths from `~/.neuralbench/config.json`. This file is **not** part of
the repository, so every machine needs its own.

```json
{
  "USER": "your-username",
  "ENTITY_NAME": "your-username",
  "PROJECT_NAME": "neuralbench",
  "DATA_DIR": "/path/to/data",
  "CACHE_DIR": "/path/to/cache",
  "SAVE_DIR": "/path/to/results",
  "CLUSTER": null
}
```

- The first six keys are required; a missing one fails with a bare `KeyError`.
- `"CLUSTER": null` runs everything locally. Remove it, or set `"auto"`, on a SLURM cluster.
- Without this file, a non-interactive run (`nohup`, batch scripts, some notebooks) falls
  back to `/tmp/neuralbench/...`, which the system clears. Write the file before the first run.
- Disk space for Track 4: about 330 GB in `DATA_DIR` and about 440 GB in `CACHE_DIR`.

---

## 5. Add a new model or training code

Work inside the cloned folder. For a new model, three files are involved:

| File | Purpose |
|---|---|
| `neuraltrain-repo/neuraltrain/models/my_model.py` | The config class and `nn.Module` |
| `neuraltrain-repo/neuraltrain/models/__init__.py` | Add `from . import my_model as _my_model  # noqa: F401` |
| `neuralbench-repo/neuralbench/models/my_model.yaml` | Tells NeuralBench which architecture `-m my_model` means |

Test it before committing:

```bash
neuralbench emg pose -m my_model --debug
```

Put new work in **new files** where you can. It keeps merges with upstream almost
conflict-free (section 8).

---

## 6. Updating with an editable install (option A)

### 6.1 On the machine where you edit

Nothing to reinstall. An editable install runs the files in your cloned folder, so edits and
new files normally take effect the next time you start Python or run `neuralbench`.

Save and share the work:

```bash
git status
git add -A
git commit -m "Describe the change"
git push -u origin emg-pose-cnn      # the first time
git push                             # afterwards
```

### 6.2 On every other machine

First time: follow sections 2.2 to 4, using `git checkout emg-pose-cnn` instead of creating
the branch.

Every time after that:

```bash
cd neuroai
git pull
```

That is the whole update. No `pip` command is needed.

### 6.3 When you do need to reinstall

Re-run the two `pip install -e` commands from step 3.3 only when the packaging itself
changed, for example:

- a `pyproject.toml` was edited (new dependency, new command-line entry point);
- you merged an upstream update that changed dependencies;
- an import fails with a missing package after a `git pull`.

Editing or adding `.py` and `.yaml` files does not require a reinstall.

---

## 7. Installing straight from GitHub (option B)

Use this when you do not want a clone on the machine.

### 7.1 Install

```bash
pip install \
  "neuraltrain @ git+https://github.com/YOUR_USERNAME/neuroai.git@emg-pose-cnn#subdirectory=neuraltrain-repo" \
  "neuralbench @ git+https://github.com/YOUR_USERNAME/neuroai.git@emg-pose-cnn#subdirectory=neuralbench-repo"
pip install 'eegdash>=0.8.2'
```

- Keep both packages in **one** command, so pip takes `neuraltrain` from your fork too.
- `@emg-pose-cnn` selects the branch. A tag or a commit hash works there as well, and a
  commit hash gives an exactly reproducible install.
- For a private repository, use `git+ssh://git@github.com/YOUR_USERNAME/neuroai.git` with
  an SSH key set up on that machine.

### 7.2 Update after you push changes

This install is a snapshot, so it does not follow your branch automatically. After pushing:

```bash
pip install --force-reinstall --no-deps \
  "neuraltrain @ git+https://github.com/YOUR_USERNAME/neuroai.git@emg-pose-cnn#subdirectory=neuraltrain-repo" \
  "neuralbench @ git+https://github.com/YOUR_USERNAME/neuroai.git@emg-pose-cnn#subdirectory=neuralbench-repo"
```

- `--force-reinstall` is needed because the version number has not changed, so pip would
  otherwise consider the package already installed.
- `--no-deps` stops pip from reinstalling `torch` and every other dependency.
- Drop `--no-deps` once if your change added a new dependency.

### 7.3 Check that new files arrived

A non-editable install only contains the files the package's build settings include. After
adding a new YAML file, confirm it was installed:

```bash
python -c "import neuralbench, os; print(os.listdir(os.path.join(os.path.dirname(neuralbench.__file__), 'models')))"
```

If your file is missing from the list, use the editable install (option A) on that machine.

---

## 8. Keeping your fork in sync with upstream

The challenge code is still changing, so merge the original repository in periodically:

```bash
git fetch upstream
git checkout emg-pose-cnn
git merge upstream/main
# resolve conflicts if any, then:
git push
```

Then run `git pull` on the other machines. If the merge changed dependencies, also re-run
step 3.3 (or step 7.2 without `--no-deps`).

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `__file__` path contains `site-packages` | The PyPI copy is still installed | `pip uninstall`, then step 3.3 |
| `-m my_model` is not recognised | YAML is missing or misnamed | Check `neuralbench/models/my_model.yaml` exists in the installed package |
| Model name in the YAML is not found | Missing import in `neuraltrain/models/__init__.py` | Add the import line from section 5 |
| Bare `KeyError` at start-up | A required key is missing from `config.json` | Add all six required keys (section 4) |
| Data disappeared after a reboot | Run fell back to `/tmp/neuralbench` | Write `config.json` with permanent paths |
| `The NVIDIA driver on your system is too old` | `torch` build is newer than the driver | Install a matching build, for example `pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` |
| Changes not visible on another machine | Not pushed, or not pulled, or wrong branch | `git status` and `git branch` on both machines |
| Option B install ignores your latest push | Snapshot install | Re-run step 7.2 |

---

## 10. Quick reference

```bash
# First install on a machine (option A)
git clone https://github.com/YOUR_USERNAME/neuroai.git && cd neuroai
git checkout emg-pose-cnn
pip uninstall -y neuralbench neuraltrain
pip install -e neuraltrain-repo
pip install -e 'neuralbench-repo/.[dev]'
pip install 'eegdash>=0.8.2'

# Publish changes from the machine where you edit
git add -A && git commit -m "message" && git push

# Update any other machine
cd neuroai && git pull

# Pull in upstream changes
git fetch upstream && git merge upstream/main && git push
```

---

## Sources

- NeuralBench installation: https://facebookresearch.github.io/neuroai/neuralbench/install.html
- Adding a new model: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/adding_model/create_new_model.html
- Repository: https://github.com/facebookresearch/neuroai
