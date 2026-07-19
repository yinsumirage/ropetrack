# AGENTS.md

This repo is `ropetrack`, the implementation repo for the hand4D wrist bench
work. Treat `E:\Desktop\hand4D\now` as the external knowledge base and
`docs/knowledge/hand4d_now.md` as the local short version.

## Rules

- Keep third-party code isolated in `third_party/`. Do not edit submodules unless
  the user explicitly asks for a patch.
- Own only the outer code: data manifests, backend wrappers, evaluation,
  visualization, hard splits, rope labels, and experiment records.
- Benchmark first. Do data protocol and coordinate checks before training.
- Use one sample schema and one prediction schema for all backends.
- Do not commit raw data, processed data, checkpoints, predictions, metrics, or
  large figures.
- Prefer stdlib and existing code. Add dependencies only when a real script needs
  them.
- Use `docs/` for stable plans and knowledge summaries. Use `experience/` for
  experiment logs, environment notes, failures, fixes, and "do not repeat this"
  records.
- Before running an experiment or retrying a failed setup, read
  `experience/INDEX.md`. After any non-trivial experiment, environment fix, data
  finding, or submodule/Git failure, write a short note and add it to the index.
- Local machine is not the runtime target. Only run light checks here, such as
  small unit tests or static file checks.
- Put any throwaway local verification scripts under `.local_checks/`; that
  directory is ignored and must not be pushed.
- Real environment setup, data download, extraction, preprocessing, CUDA checks,
  model inference, and benchmark runs belong on the HPC cluster.
- Do not pin old PyTorch only to dodge checkpoint loading issues. First make
  `torch.load` behavior explicit in the smallest possible patch. Prefer
  `weights_only=True` for state-dict checkpoints; if a trusted upstream
  checkpoint truly requires object unpickling, use `weights_only=False` only in
  that isolated load site and document why.

## Current Research Line

Phases P0-P2 are closed and the formal release remains pinned by `RELEASE.md`.
The active experimental line is the frozen-token `DirectPoseHead`; its current
code/artifact/branch map is `docs/current-code-and-artifact-map.md` and its
authoritative no-leak result is `experience/0079_normal_joint_no_leak_final.md`.
Continue trusted data coverage, but do not promote the fixed triple mixture or
tune another mixture on the same final scores. Dense K16/K96, larger generic
temporal models, the tested simple natural-HOT3D state gates, and the global
orientation head are stopped directions. Physical rope sensing remains
unvalidated.

## HPC Rules

Treat `E:\Desktop\hpc` as the external HPC knowledge base. If these notes are
unclear or stale, re-read that folder before writing or running cluster scripts.

- SSH host is `hpc`; Slurm account is `engram`.
- HPC Miniforge root is `/public/home/guowt2512/miniforge3`; use
  `source /public/home/guowt2512/miniforge3/etc/profile.d/conda.sh && conda activate ropetrack`
  for RopeTrack checks.
- Before remote `git pull` or other Git network operations, set
  `http_proxy=http://hkuhpc.com:7999` and
  `https_proxy=http://hkuhpc.com:7999`.
- From Windows/PowerShell, run remote commands through an explicit login shell:
  `ssh hpc "bash -lc 'cd ~/project/ropetrack && ...'"`.
- Do not diagnose missing `conda`, `squeue`, `sbatch`, or modules from a bare
  `ssh hpc "command"` failure. First retry with `bash -lc`; for conda, also
  source `/public/home/guowt2512/miniforge3/etc/profile.d/conda.sh` in that same
  remote command.
- PowerShell expands `$...` before `ssh` inside double quotes. Escape remote
  shell variables as `` `$VAR`` or avoid inline `$` entirely. If a command needs
  nested quotes, `$`, here-docs, or multi-line Python, put it in a short remote
  shell script and run that script instead of fighting PowerShell quoting.
- In PowerShell, quote Git refs with braces, such as `git stash drop 'stash@{0}'`;
  unquoted `stash@{0}` is parsed by PowerShell before Git sees it.
- PowerShell here-strings may send CRLF to remote bash; if a streamed script
  fails with commands like `sort\r`, pipe through `tr -d '\r'` or run a saved
  `.sh` file.
- Default Slurm commands and scripts must include `-A engram`.
- Current practical partitions are `cpu` and `gpu`; do not confuse `-A engram`
  with `-p engram`.
- Login node is only for lightweight work: edit files, inspect queues, sync
  small code changes, and submit jobs.
- Do not run training, benchmark inference, large downloads, extraction,
  preprocessing, large compiles, or long Python jobs on the login node.
- Use CPU compute nodes for environment installation, package builds, downloads,
  extraction, and preprocessing.
- Use GPU nodes for CUDA checks, model inference, and GPU benchmark jobs.
- Prefer `sbatch` for real runs. Use `salloc`/`srun` only for short interactive
  debugging because local network drops can kill interactive jobs.
- GPU jobs must do real GPU work; do not keep GPUs idle with sleep loops,
  hidden keepalive processes, or long empty shells.
- CPU-only single-node jobs must not request more than 96 CPU cores.
- Any GPU job script should specify account, partition, CPU count, memory,
  GPU count, wall time, and log paths.
- Start with 1 GPU sanity checks. Do not request more than 8 GPUs without a
  recorded scaling/IO/memory justification.
- If a job is stuck, using no GPU, has bad data paths, or is downloading on a GPU
  allocation, cancel it rather than letting it violate idle-use rules.

## Current Remote Facts

- Treat `E:\Desktop\ropetrack` as the primary local worktree for this repo.
  Codex may also create detached worktrees under `C:\Users\gwt\.codex\worktrees`;
  before editing or committing, check `git worktree list` and prefer the primary
  worktree unless the user explicitly asks to keep an isolated Codex worktree.
- The stable HPC repo path is `~/project/ropetrack`. Do not record transient
  compute node names; Slurm may assign a different node.
- The repo lives under `~/project/ropetrack` on the HPC account, not under
  `/data`.
- Current data root: `/data/wentao/ropetrack`.
- Store benchmark outputs and Slurm logs under `/data/wentao/ropetrack/runs`,
  not inside `~/project/ropetrack`; keep temporary sbatch/debug scripts in the
  ignored repo-local `.local_checks/` directory.
- Current data under `/data/wentao/ropetrack`:
  - `pretrained_models`: shared checkpoint root. `pretrained_models` at the
    repo root on HPC is a symlink to this directory.
  - `FreiHAND`: extracted root is `/data/wentao/ropetrack/FreiHAND`; original
    archives and download-shell metadata were deleted after successful
    extraction.
  - `HO3D_v2_eval`: expanded eval set with `evaluation/`,
    `evaluation_xyz.json`, and `evaluation_verts.json`.
  - `HO3D_v3`: renamed from `HO-3D`; full zip was extracted and deleted. It now
    has `train/`, `evaluation/`, `train.txt`, `evaluation.txt`,
    `evaluation_xyz.json`, and `evaluation_verts.json`. The evaluation split has
    20137 samples and uses `.jpg` RGB images; train/evaluation sequences contain
    `rgb`, `depth`, and `meta`, and train sequences also have `seg`.
- HaMeR checkpoint files were moved under the shared `pretrained_models`, but
  `pretrained_models/hamer_ckpts/checkpoints/model_config.yaml` was observed as
  0 bytes on 2026-07-02. Re-check HaMeR assets before using that backend.
- The first benchmark anchor is still FreiHAND + HO3D v2. HO3D v3 now has full
  clean runs and may appear in the clean report, but label it explicitly as
  HO3D v3 rather than merging it into HO3D v2 numbers.
- Current working Python/PyTorch environment uses `torch==2.5.*`, so PyTorch 2.6
  checkpoint-loading changes are not the active issue unless the environment is
  upgraded later.
- Prefer WiLoR first for smoke inference. Treat HaMeR as a separate environment
  path because its `detectron2`, `mmcv`, `ViTPose`, and `chumpy` stack is
  brittle.

Minimal GPU job header:

```bash
#SBATCH -A engram
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
```

Minimal CPU setup/debug allocation:

```bash
salloc -A engram -p cpu -N 1 -c 4 --mem=16G -t 02:00:00
srun --pty bash
```

Minimal GPU check allocation:

```bash
salloc -A engram -p gpu -N 1 -c 4 --mem=32G --gres=gpu:1 -t 01:00:00
srun --pty bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Third-Party Backends

- `third_party/hamer`: https://github.com/geopavlakos/hamer
- `third_party/wilor`: https://github.com/rolpotamias/WiLoR

## Expected Data Interfaces

Hand-pose dataset adapters live in `ropetrack/datasets/hand_pose.py`. They
provide sample ids, image paths, GT bboxes, and protocol helpers shared by eval,
hard split generation, rope labels, and training data export.

Benchmark predictions are still written as the project-level `pred.json`
payload `[xyz_predictions, vertex_predictions]`, with run order recorded in
`run_meta.json`.
