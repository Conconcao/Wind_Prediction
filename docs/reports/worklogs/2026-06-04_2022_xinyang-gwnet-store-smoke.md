# 2026-06-04 Xinyang Graph WaveNet Store Smoke

## Goal

- Continue from the new disk-backed store path and connect it to the first
  graph-based primary model candidate.
- Keep the implementation grounded in the mature Graph WaveNet reference,
  but adapt it to the current `15-minute-ahead` one-step xinyang task.

## What I changed

### 1. Extended graph utilities

Updated:

- `15min-lead/wind/src/xinyang_wind15/graph.py`

Main additions:

- row-normalized adjacency helper
- Graph WaveNet support preparation helper

### 2. Added a Graph WaveNet style model

New file:

- `15min-lead/wind/src/xinyang_wind15/gwnet.py`

Design notes:

- inspired by the official Graph WaveNet repository structure
- keeps the core pieces:
  - temporal gated convolutions
  - graph convolution over fixed supports
  - adaptive adjacency
  - residual and skip paths
- simplified to the current one-step multi-turbine output setting

### 3. Added a store-based graph-model training script

New file:

- `15min-lead/wind/scripts/train_gwnet_from_store.py`

Behavior:

- reads the disk-backed feature store
- reads the saved distance adjacency
- builds normalized supports
- trains the graph model with the same evaluation pipeline used by the other deep models

### 4. Added tests

Updated:

- `15min-lead/wind/tests/test_model_shapes.py`

Added:

- Graph WaveNet style forward-shape coverage

## Why this design

- The project plan already identified a Graph WaveNet style model as the main
  advanced model direction.
- The new disk-backed store path made it practical to wire this model in
  without returning to the dense-window memory bottleneck.
- A distance-graph plus adaptive adjacency is a sensible first graph setup
  for one farm with known turbine coordinates.

## Verification

Fresh checks run after the edits:

- `pytest 15min-lead\\wind\\tests -q`
  - result: `8 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts`
  - result: passed
- Graph WaveNet style store smoke:
  - `python 15min-lead\\wind\\scripts\\train_gwnet_from_store.py --store-dir 15min-lead\\wind\\artifacts\\local_debug\\window_store_smoke_enriched_v1 --epochs 2 --batch-size 128 --output-dir 15min-lead\\wind\\artifacts\\local_debug\\gwnet_store_smoke_enriched_v1`

## Current result

From `artifacts/local_debug/gwnet_store_smoke_enriched_v1/metrics.csv`:

- val RMSE: `1.0980`
- test RMSE: `0.7939`

Interpretation:

- this is currently the best deep model in the local debug stack
- it improves over the current TCN and GRU debug runs
- it still does not beat persistence on the same subset

## Reference alignment

This implementation was guided primarily by:

- Graph WaveNet reference repo:
  `https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py`
- PyTorch `Conv2d` docs:
  `https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html`
- PyTorch `Dataset` / `DataLoader` docs:
  `https://docs.pytorch.org/docs/stable/data.html`
- NumPy `open_memmap` docs:
  `https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html`

## Current boundary

- This is still a debug-subset result on `4` turbines and `12000` timestamps.
- The graph supports currently use:
  - distance graph
  - transpose distance graph
  - optional learned adaptive graph
- The correlation graph mentioned in the experiment plan is not added yet.

## Recommended next step

- Use the current store path to prepare a full `46`-turbine graph-model run.
- Then write the corresponding server-side training job script for the graph model.
