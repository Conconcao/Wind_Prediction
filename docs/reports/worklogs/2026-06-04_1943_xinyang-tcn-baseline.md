# 2026-06-04 Xinyang 15-Min TCN Baseline

## Goal

- Continue the local deep-learning stack for the xinyang `15-minute-ahead`
  wind-speed task.
- Add a second mature deep-learning baseline after GRU, using a TCN design
  inspired by the established locuslab TCN implementation path.

## What I changed

### 1. Added shared sequence-training utilities

New file:

- `15min-lead/wind/src/xinyang_wind15/sequence.py`

Purpose:

- keep `WindowDataset`, normalization, evaluation, and checkpoint helpers
  in one place
- let GRU and TCN use the same training and metric logic
- reduce code duplication before adding more sequence models

### 2. Added the TCN model

New file:

- `15min-lead/wind/src/xinyang_wind15/tcn.py`

Main design:

- causal dilated `Conv1d`
- residual temporal blocks
- seq2one multi-turbine output head

Why this design:

- it follows the mature TCN pattern from the locuslab line
- it is a natural next baseline after GRU for short-horizon forecasting
- it matches the model shortlist already defined for this experiment

### 3. Added a local TCN training script

New file:

- `15min-lead/wind/scripts/train_tcn_baseline.py`

Behavior:

- uses the same feature-building path as the GRU script
- supports `--include-tower` and `--include-1min`
- trains on the same window tensors as the current GRU baseline
- writes summary, metrics, per-turbine metrics, history, and checkpoint files

### 4. Refactored the GRU path to use the shared utilities

Updated:

- `15min-lead/wind/src/xinyang_wind15/gru.py`
- `15min-lead/wind/scripts/train_gru_baseline.py`

Result:

- GRU and TCN now share the same standardization and evaluation code path
- this makes model-to-model comparison cleaner

### 5. Added model tests and updated docs

New test:

- `15min-lead/wind/tests/test_model_shapes.py`

Updated docs:

- `15min-lead/wind/README.md`
- `15min-lead/wind/references/implementation_sources.md`

### 6. Added a local memory guard for dense window materialization

Updated:

- `15min-lead/wind/src/xinyang_wind15/windows.py`
- `15min-lead/wind/scripts/train_gru_baseline.py`
- `15min-lead/wind/scripts/train_tcn_baseline.py`
- `15min-lead/wind/tests/test_pipeline_utils.py`

Purpose:

- estimate the memory footprint of dense window tensors before training
- stop local runs early when the projected materialized tensor size is too large
- make the current script behavior safer until a streaming full-farm path is built

## Why this was worth doing now

- The experiment plan already identified `GRU + TCN` as the first deep
  baselines before the primary graph model.
- Adding TCN now gives a stronger non-graph deep baseline for later
  comparison.
- Refactoring the shared training utilities now reduces rework before
  adding the spatio-temporal graph model.

## Verification

Fresh checks run after the edits:

- `pytest 15min-lead\\wind\\tests -q`
  - result: `6 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts`
  - result: passed
- Refreshed GRU smoke after the refactor:
  - `python 15min-lead\\wind\\scripts\\train_gru_baseline.py --epochs 2 --batch-size 128 --max-turbines 4 --tail-timestamps 12000 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\gru_smoke_enriched_v3`
- New TCN smoke:
  - `python 15min-lead\\wind\\scripts\\train_tcn_baseline.py --epochs 3 --batch-size 128 --max-turbines 4 --tail-timestamps 12000 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\tcn_smoke_enriched_v1`
- Guard verification runs:
  - `python 15min-lead\\wind\\scripts\\train_gru_baseline.py --epochs 1 --batch-size 128 --max-turbines 4 --tail-timestamps 12000 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\gru_smoke_guardcheck_v2`
  - `python 15min-lead\\wind\\scripts\\train_tcn_baseline.py --epochs 1 --batch-size 128 --max-turbines 4 --tail-timestamps 12000 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\tcn_smoke_guardcheck_v2`

## Current debug-subset results

### GRU re-check

From `artifacts/local_debug/gru_smoke_enriched_v3/metrics.csv`:

- val RMSE: `1.3431`
- test RMSE: `0.9928`

This matches the earlier GRU behavior, which is good evidence that the
refactor did not change the model path unintentionally.

### TCN smoke

From `artifacts/local_debug/tcn_smoke_enriched_v1/metrics.csv`:

- val RMSE: `0.8402`
- test RMSE: `0.8808`

Interpretation:

- TCN is clearly better than the current GRU on this enriched debug subset
- TCN is still weaker than persistence on the same subset, so it is not yet
  the best-performing method

### Guard verification

From the guard-check summaries:

- projected dense window size on the validated `4`-turbine debug subset:
  about `1.03 GiB`

This is acceptable for local smoke use, but it confirms that the current
dense-window design should not simply be scaled up blindly.

## Important boundary

- The current GRU and TCN scripts still materialize full window tensors in
  memory.
- That is acceptable for local debug subsets, but not yet the final design
  for full-farm `46`-turbine runs.
- A shorter tail subset such as `6000` timestamps can also lose the train
  split entirely, so local smoke subsets still need to span train/val/test
  together.

## Recommended next step

- Build a memory-safer full-farm deep-learning data path.
- Then move on to the primary graph model, reusing the shared sequence
  training utilities where possible.
