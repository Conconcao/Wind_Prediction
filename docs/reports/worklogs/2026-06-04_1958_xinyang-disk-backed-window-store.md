# 2026-06-04 Xinyang Disk-Backed Window Store

## Goal

- Continue the xinyang `15-minute-ahead` deep-learning pipeline toward a
  full `46`-turbine workflow.
- Remove the biggest local scaling bottleneck from the current GRU/TCN
  path: dense materialization of every training window in memory.

## What I changed

### 1. Refactored window utilities

Updated:

- `15min-lead/wind/src/xinyang_wind15/windows.py`

Main changes:

- extracted shared feature/target array construction
- added vectorized valid-window index computation
- kept the existing dense-window builder working on top of the refactored logic

### 2. Added a disk-backed window store

New file:

- `15min-lead/wind/src/xinyang_wind15/window_store.py`

Main behavior:

- writes a time-major `feature_tensor.npy`
- writes a time-major `target_matrix.npy`
- writes `timestamps.npy`, `origin_indices.npy`, `target_indices.npy`, and
  `split_labels.npy`
- stores metadata in `metadata.json`

Important implementation detail:

- the store writes feature pivots one feature at a time into a `.npy`
  memmap instead of building all training windows eagerly

### 3. Added lazy-loading sequence dataset support

Updated:

- `15min-lead/wind/src/xinyang_wind15/sequence.py`

Main changes:

- added `WindowStoreDataset`
- added exact train-split standardization-stat computation from the disk-backed store
- extended `make_loader` so it can work with either in-memory arrays or a
  map-style dataset

### 4. Added new scripts

New files:

- `15min-lead/wind/scripts/build_window_store.py`
- `15min-lead/wind/scripts/train_tcn_from_store.py`

Purpose:

- `build_window_store.py` prepares the disk-backed store and adjacency files
- `train_tcn_from_store.py` proves that the TCN can train from lazy-loaded windows

### 5. Added tests

New file:

- `15min-lead/wind/tests/test_window_store.py`

Also updated:

- `15min-lead/wind/tests/test_pipeline_utils.py`

Purpose:

- verify store round-trip
- verify lazy dataset sample shapes
- verify the dense-window-size estimation helper

## Why this design

- The dense-window path duplicates the same time slices many times.
- For the enriched xinyang setup, that duplication becomes the main memory cost.
- A time-major feature store keeps only one copy of each timestamp-turbine-feature
  value and lets training cut windows lazily.
- That is a better base for the next stage, especially the graph model.

## Verification

Fresh checks run after the edits:

- `pytest 15min-lead\\wind\\tests -q`
  - result: `7 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts`
  - result: passed

Smoke runs:

- store build:
  - `python 15min-lead\\wind\\scripts\\build_window_store.py --max-turbines 4 --tail-timestamps 12000 --lookback-steps 32 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\window_store_smoke_enriched_v1`
- TCN from store:
  - `python 15min-lead\\wind\\scripts\\train_tcn_from_store.py --store-dir 15min-lead\\wind\\artifacts\\local_debug\\window_store_smoke_enriched_v1 --epochs 2 --batch-size 128 --output-dir 15min-lead\\wind\\artifacts\\local_debug\\tcn_store_smoke_enriched_v1`

## Current results

### Store smoke summary

From `artifacts/local_debug/window_store_smoke_enriched_v1/summary.json`:

- `feature_tensor_shape = [12000, 4, 181]`
- `target_matrix_shape = [12000, 4]`
- `n_valid_windows = 3827`
- `n_train = 535`
- `n_val = 2443`
- `n_test = 849`
- estimated store size: about `0.0325 GiB`

### TCN-from-store smoke

From `artifacts/local_debug/tcn_store_smoke_enriched_v1/metrics.csv`:

- val RMSE: `0.9934`
- test RMSE: `1.0319`

This run used only `2` epochs, so the point here is pipeline validation
rather than final performance. The important result is that lazy-loaded
training now works end to end.

## Scale comparison

Using the current enriched feature count `181`, lookback `32`, and the full
xinyang farm size:

- full-farm dense windows: about `33.93 GiB`
- full-farm disk-backed feature store: about `1.07 GiB`

This confirms that the store path removes the main storage duplication cost.

## Notes from debugging

- The first training attempt failed because I launched store building and
  store training in parallel, so training started before `metadata.json`
  had been written.
- Re-running the training step after the store completed resolved that issue.
- This was a workflow-order issue, not a code bug.

## Recommended next step

- Reuse this disk-backed store in the primary graph-model training path.
- Keep local smoke runs small, but use the new store path as the default
  structure for larger server-side deep-learning jobs.
