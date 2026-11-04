#!/usr/bin/env bash
# Read-only preflight check for running the neuroai_jf fork on Newton (UCF ARCC).
# Prints, one section at a time, everything a teammate needs to confirm before
# launching a *_newton.yaml run. Every step keeps going when it fails and says why.
#
# Usage (from anywhere on Newton):
#   bash <clone>/tutorial/newton_check.sh
#
# Note: the login node (evuser2) cannot import torch (process/memory limits), so
# section 2 only passes from a compute node, e.g. `srun -p short --pty bash`.
# Nothing here writes to disk or submits a job.

section() { printf '\n==== %s ====\n' "$1"; }
fail()    { printf 'FAILED: %s\n' "$1"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DATA_DIR=/lustre/fs1/home/ji045311/challenge_2026/Data
CACHE_DIR=/lustre/fs1/home/ji045311/challenge_2026/Cache
NEED_MIN=2880   # infra.timeout_min in st_pose_transformer{,_ghm}_newton.yaml

printf 'newton_check.sh  host=%s  user=%s  date=%s\n' "$(hostname)" "$(whoami)" "$(date '+%F %T')"

# ---------------------------------------------------------------- 1. conda envs
section "1. conda environments (module load anaconda/anaconda-2024.10)"
if module load anaconda/anaconda-2024.10 2>/tmp/newton_check_module.$$; then
  conda env list 2>&1 || fail "conda env list returned an error"
else
  fail "module load anaconda/anaconda-2024.10: $(tr '\n' ' ' < /tmp/newton_check_module.$$)"
fi
rm -f /tmp/newton_check_module.$$

# ---------------------------------------------------------------- 2. python packages
section "2. neuralbench / neuraltrain install (editable, inside the clone?)"
ENV=""
if command -v conda >/dev/null 2>&1; then
  for cand in neuroai challenge_2026; do
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$cand"; then ENV="$cand"; break; fi
  done
fi
if [ -z "$ENV" ]; then
  fail "neither 'neuroai' nor 'challenge_2026' is a conda env on this machine"
else
  echo "using env: $ENV"
  eval "$(conda shell.bash hook 2>/dev/null)" || fail "conda shell hook unavailable"
  if conda activate "$ENV" 2>/dev/null; then
    echo "python: $(command -v python)"
    python - <<'PY' 2>&1 || fail "import neuralbench / neuraltrain failed (on the login node this is expected: torch cannot load there; rerun from a compute node)"
import importlib
for name in ("neuralbench", "neuraltrain"):
    m = importlib.import_module(name)
    f = m.__file__
    kind = "site-packages (NOT editable)" if "site-packages" in f else "editable clone"
    print(f"{name}.__file__ = {f}   [{kind}]")
PY
    for pkg in neuralbench neuraltrain; do
      pip show "$pkg" 2>/dev/null | grep -E '^(Version|Editable project location)' | sed "s/^/$pkg: /" \
        || fail "pip show $pkg: not installed in $ENV"
    done
  else
    fail "conda activate $ENV failed"
  fi
fi

# ---------------------------------------------------------------- 3. clone
section "3. clone that holds this script: $REPO"
if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "branch:  $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "commit:  $(git -C "$REPO" log -1 --format='%h %ci %s')"
  echo "remote:  $(git -C "$REPO" remote get-url origin 2>/dev/null)"
  echo "behind/ahead origin: $(git -C "$REPO" rev-list --left-right --count '@{u}...HEAD' 2>/dev/null || echo 'unknown (run git fetch)')"
  dirty="$(git -C "$REPO" status --short 2>/dev/null | head -10)"
  [ -n "$dirty" ] && printf 'uncommitted changes:\n%s\n' "$dirty" || echo "working tree clean"
else
  fail "$REPO is not a git work tree (was this script copied out of the clone?)"
fi

# ---------------------------------------------------------------- 4. config.json
section "4. ~/.neuralbench/config.json"
CFG="${NEURALBENCH_CONFIG:-$HOME/.neuralbench/config.json}"
if [ -f "$CFG" ]; then
  echo "path: $CFG"
  cat "$CFG"; echo
  if command -v python >/dev/null 2>&1; then
    python - "$CFG" <<'PY' 2>&1 || fail "config.json is not valid JSON"
import json, sys
cfg = json.load(open(sys.argv[1]))
for k in ("DATA_DIR", "CACHE_DIR", "SAVE_DIR", "CLUSTER", "SLURM_PARTITION", "N_CPUS"):
    print(f"  {k:16s} = {cfg.get(k)!r}")
bad = [k for k, v in cfg.items() if isinstance(v, str) and "<NID>" in v]
if bad:
    print("  WARNING: placeholder <NID> still present in:", ", ".join(bad))
if cfg.get("CLUSTER") == "auto" and not cfg.get("SLURM_PARTITION"):
    print("  WARNING: SLURM_PARTITION is empty; the *_newton YAMLs expect 'normal'")
PY
  fi
else
  fail "$CFG does not exist (copy tutorial/newton_config.example.json there and fill in <NID>)"
fi

# ---------------------------------------------------------------- 5. shared data
section "5. read access to the shared Data / Cache directories"
for d in "$DATA_DIR" "$CACHE_DIR"; do
  echo "-- $d"
  if [ -d "$d" ]; then
    ls -ld "$d" 2>&1
    if [ -r "$d" ] && [ -x "$d" ]; then
      echo "readable: yes; first entries:"
      ls "$d" 2>&1 | head -8 | sed 's/^/    /'
    else
      fail "no read/execute permission on $d (ask the owner for g+rx or o+rx)"
    fi
  else
    fail "$d does not exist or is not visible from this node"
  fi
done
if [ -d "$DATA_DIR/Salter2024Emg2pose" ]; then
  echo "Salter2024Emg2pose: present ($(ls "$DATA_DIR/Salter2024Emg2pose" 2>/dev/null | wc -l) entries)"
else
  fail "$DATA_DIR/Salter2024Emg2pose not found; DATA_DIR must be the parent of the BIDS folder"
fi

# ---------------------------------------------------------------- 6. SLURM partitions
section "6. SLURM partitions normal / short (need MaxTime >= ${NEED_MIN} min)"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -p normal,short -o '%P %l %a %D %G' 2>&1 || fail "sinfo returned an error"
  for p in normal short; do
    mt="$(scontrol show partition "$p" 2>/dev/null | grep -o 'MaxTime=[^ ]*' | cut -d= -f2)"
    if [ -z "$mt" ]; then fail "partition $p not found"; continue; fi
    # MaxTime forms: UNLIMITED | D-HH:MM:SS | HH:MM:SS | MM:SS | MM
    mins="$(echo "$mt" | awk -F'[-:]' '
      /UNLIMITED|infinite/ {print "inf"; next}
      NF==4 {print $1*1440+$2*60+$3; next}
      NF==3 {print $1*60+$2; next}
      NF==2 {print $1; next}
      {print $1}')"
    if [ "$mins" = "inf" ] || [ "$mins" -ge "$NEED_MIN" ] 2>/dev/null; then
      echo "$p: MaxTime=$mt -> OK for timeout_min ${NEED_MIN}"
    else
      echo "$p: MaxTime=$mt (${mins} min) -> too short for timeout_min ${NEED_MIN}"
    fi
  done
else
  fail "sinfo not found (not a SLURM login/compute node)"
fi

printf '\nDone. Sections marked FAILED need attention before launching a *_newton run.\n'
