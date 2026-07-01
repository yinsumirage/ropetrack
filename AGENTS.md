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

Start with FreiHAND + HO3D v2, reproduce clean HaMeR/WiLoR/AnyHand baselines,
then build hard mask/blur/crop/appearance splits. Only after clean and hard
benchmarks are credible, test fingertip-to-wrist rope distance as post-opt,
then as model input/loss.

## HPC Rules

Treat `E:\Desktop\hpc` as the external HPC knowledge base. If these notes are
unclear or stale, re-read that folder before writing or running cluster scripts.

- SSH host is `hpc`; Slurm account is `engram`.
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

- The stable HPC repo path is `~/project/ropetrack`. Do not record transient
  compute node names; Slurm may assign a different node.
- The repo lives under `~/project/ropetrack` on the HPC account, not under
  `/data`.
- Current data root: `/data/wentao/ropetrack`.
- Current downloaded-but-unprocessed data under `/data/wentao/ropetrack`
  includes `freihand` and `ho3dv3`.
- The first benchmark plan still targets FreiHAND + HO3D v2. Do not silently mix
  HO3D v3 into the first clean benchmark; if only v3 is available, document that
  protocol change before running scores.
- Current working Python/PyTorch environment uses `torch==2.5.*`, so PyTorch 2.6
  checkpoint-loading changes are not the active issue unless the environment is
  upgraded later.
- Prefer AnyHand/WiLoR first for smoke inference. Treat HaMeR as a separate
  environment path because its `detectron2`, `mmcv`, `ViTPose`, and `chumpy`
  stack is brittle.

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
- `third_party/anyhand`: https://github.com/chen-si-cs/AnyHand

## Expected Schemas

Core sample fields live in `src/ropetrack/schema.py`: `sample_id`, `dataset`,
`split`, `image_path`, `hand_side`, `bbox_xyxy`, optional camera and label paths.

Core prediction fields live there too: `sample_id`, `backend`,
predicted joints/vertices/MANO paths, optional projected joints, bbox, hand side,
confidence, and coordinate notes.
