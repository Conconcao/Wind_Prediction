# 2026-06-04 Xinyang GWNet Correlation and Server Prep

## Goal

- Extend the current Graph WaveNet style path beyond distance-only supports.
- Prepare the first full-farm server-side experiment entry points now that the
  local graph-model path is validated.

## What I changed

### 1. Added correlation-based graph support

Updated:

- `15min-lead/wind/src/xinyang_wind15/graph.py`
- `15min-lead/wind/scripts/build_window_store.py`
- `15min-lead/wind/scripts/train_gwnet_from_store.py`

Main changes:

- added Pearson correlation adjacency construction from training-split SCADA only
- added support deduplication so symmetric supports are not duplicated unnecessarily
- added `support_mode` to the graph training script:
  - `distance`
  - `distance_correlation`
  - `auto`

### 2. Added server-side config and job scripts

New files:

- `15min-lead/wind/configs/splits/xinyang_7_2_1_server.yaml`
- `jobs/slurm/xinyang_build_store_full.slurm`
- `jobs/slurm/xinyang_train_gwnet_full.slurm`
- `15min-lead/wind/docs/xinyang_server_layout.md`

Purpose:

- define the expected server data paths explicitly
- make the first full-farm graph-model workflow reproducible
- keep the GPU request within the project rule by using `1` GPU per job

### 3. Added tests

Updated:

- `15min-lead/wind/tests/test_pipeline_utils.py`

Main additions:

- correlation adjacency smoke coverage
- support deduplication coverage

## Why this was the right next step

- The graph model was already the strongest deep model on the local debug subset.
- The experiment plan explicitly called out a correlation graph as a useful
  complement to the geographic graph.
- The server-side path is now worth formalizing because the local store and
  graph-model code are already proven end to end.

## Verification

Fresh checks run after the edits:

- `pytest 15min-lead\\wind\\tests -q`
  - result: `9 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts jobs\\slurm`
  - result: passed
- rebuilt store with correlation adjacency:
  - `python 15min-lead\\wind\\scripts\\build_window_store.py --max-turbines 4 --tail-timestamps 12000 --lookback-steps 32 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\window_store_smoke_enriched_v2`
- graph model with `distance_correlation` supports:
  - `python 15min-lead\\wind\\scripts\\train_gwnet_from_store.py --store-dir 15min-lead\\wind\\artifacts\\local_debug\\window_store_smoke_enriched_v2 --support-mode distance_correlation --epochs 2 --batch-size 128 --output-dir 15min-lead\\wind\\artifacts\\local_debug\\gwnet_store_smoke_enriched_v2`

## Current result

From `artifacts/local_debug/gwnet_store_smoke_enriched_v2/metrics.csv`:

- val RMSE: `1.0964`
- test RMSE: `0.7924`

Comparison with the previous distance-only graph smoke:

- distance-only test RMSE: `0.7939`
- distance + correlation test RMSE: `0.7924`

Interpretation:

- the improvement is small on this debug subset
- but the combined-support path is now working and ready for full-farm evaluation

## Server data upload target

Please place the required xinyang files in:

- `/s502024280003/gpfs/Wind_Prediction/data/xinyang/`

Optional later NWP file:

- `/s502024280003/gpfs/Wind_Prediction/data/open_nwp/gdex_gfs_xinyang_2025-07-01_2025-12-31_hourly.parquet`

## Recommended next step

- Run the full `46`-turbine store build on the server.
- Then launch the Graph WaveNet style full run with `distance_correlation`
  supports and compare it against the distance-only variant.
