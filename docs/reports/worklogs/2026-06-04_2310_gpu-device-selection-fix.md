# 2026-06-04 23:10 GPU Device Selection Fix

## Goal

Fix the training scripts so server jobs submitted to GPU queues actually
use CUDA when a GPU is available.

## What I changed

- Updated `train_gwnet_from_store.py` to select
  `torch.device("cuda" if torch.cuda.is_available() else "cpu")` instead
  of always forcing CPU.
- Updated `train_tcn_from_store.py` with the same CUDA-aware device
  selection.
- Updated the local `GRU` and `TCN` training scripts to use the same
  logic for consistency.
- Added `torch.cuda.manual_seed_all(seed)` in each script when CUDA is
  available.
- Added the resolved device string to the saved training summary for the
  store-based `TCN` and `GraphWaveNet` runs.

## Why I changed it

The server training jobs request GPU resources, but the existing scripts
were hard-coded to `torch.device("cpu")`. That would waste GPU queue
resources and make the full xinyang training run much slower than
necessary.

## Verification

- Searched the training scripts for hard-coded `torch.device("cpu")`
  usage.
- Updated all four affected training entry points consistently.

## Risks and next steps

- This fix assumes the server PyTorch build has CUDA support in the
  active environment. If the environment is CPU-only, the scripts will
  safely fall back to CPU.
- The user should pull the updated code on the server before submitting
  the Graph WaveNet training job.
