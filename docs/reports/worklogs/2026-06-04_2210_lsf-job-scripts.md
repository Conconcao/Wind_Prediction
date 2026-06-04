# 2026-06-04 22:10 LSF Job Scripts

## Goal

Add server job scripts for the user's `LSF / bsub` cluster so the xinyang
full-run workflow can be submitted without manually translating the
existing Slurm scripts on the server.

## What I changed

- Added `jobs/lsf/` as the repository location for reusable `bsub` job scripts.
- Added `jobs/lsf/xinyang_build_store_full.lsf` for the full disk-backed
  xinyang store build.
- Added `jobs/lsf/xinyang_train_gwnet_full.lsf` for the full Graph
  WaveNet style training run from the saved store.
- Set both new scripts to use the persistent server root
  `/home3/s502024280003/Wind_Prediction`.
- Set both new scripts to activate the Conda environment
  `AIRU-WRF-torch` by default, while still falling back to `.venv` if
  needed.
- Updated the server-side docs so `bsub` / `LSF` is now the primary
  scheduler path for this cluster, while preserving the Slurm scripts as
  optional compatibility variants.
- Corrected the server layout document to show the actual uploaded file
  names `pre_QC_气象观测数据.xlsx` and `风机基本信息.csv`.

## Why I changed it

The user confirmed that the target cluster uses `bsub`, so relying only
on Slurm job scripts would create avoidable translation work and a higher
risk of submission mistakes. Adding explicit LSF scripts keeps the local
repo aligned with the real execution environment and makes the next
server step straightforward.

## Verification

- Checked the repository before editing and confirmed there were no
  existing `bsub` or `LSF` job scripts to reuse.
- Cross-checked the Python CLI arguments used by
  `build_window_store.py` and `train_gwnet_from_store.py` so the new LSF
  scripts call valid flags.
- Verified the required xinyang file names from the local data directory
  before updating the server layout document.

## Risks and next steps

- The exact LSF queue name and GPU resource syntax can vary by cluster.
  The scripts currently assume a common `gpu` queue and `#BSUB -gpu
  "num=1"` syntax; if the server uses a different convention, the first
  few `#BSUB` lines may need a small local edit.
- Before the first full run, the user should `git pull` on the server and
  optionally do a small smoke submission or an interactive smoke command
  to confirm that `AIRU-WRF-torch` activates correctly inside batch jobs.
