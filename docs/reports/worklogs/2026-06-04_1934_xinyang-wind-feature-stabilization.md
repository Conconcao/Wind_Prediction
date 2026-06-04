# 2026-06-04 Xinyang 15-Min Wind Feature Stabilization

## Goal

- Continue the `15min-lead/wind/` implementation by making the enriched
  local pipeline stable enough for the next deep-learning stage.
- Keep the workflow aligned with the project rule of testing code locally
  before preparing larger server-side runs.

## What I changed

### 1. Fixed the enriched feature pipeline

Updated:

- `15min-lead/wind/src/xinyang_wind15/loading.py`
- `15min-lead/wind/src/xinyang_wind15/features.py`

Main changes:

- Dropped tower feature columns that are structurally empty in the local
  xinyang tower file.
- Replaced feature backfilling with forward-only filling so the pipeline
  uses past information only.
- Added missing-indicator columns in one batch to avoid the previous
  pandas fragmentation warnings.
- Kept the 1-minute aggregate features and tower ablations integrated
  into both tabular and sequence pipelines.

### 2. Hardened the window and GRU path

Updated:

- `15min-lead/wind/src/xinyang_wind15/windows.py`
- `15min-lead/wind/scripts/train_gru_baseline.py`

Main changes:

- Added a clearer error when no valid windows can be generated because of
  missing features.
- Fixed the GRU input-dimension bug so the model now uses the actual
  enriched feature count instead of the original base feature count.

### 3. Synced lightweight documentation

Updated:

- `15min-lead/wind/README.md`
- `15min-lead/wind/references/implementation_sources.md`

Main changes:

- Recorded the now-working `--include-tower` and `--include-1min`
  options.
- Added the relevant pandas API references used for the forward-fill
  implementation.

## Why these changes were needed

- The previous enriched pipeline failed in two ways:
  - the unit test for grouped feature filling crashed
  - the GRU smoke run produced zero valid windows
- Root cause analysis showed that the tower dataset contained a block of
  all-null variables, and those columns forced every enriched window to
  contain missing values.
- The earlier backfill logic also had leakage risk because it could pull
  future information into earlier timestamps.

## Verification

Fresh checks run after the edits:

- `pytest 15min-lead\\wind\\tests -q`
  - result: `3 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts`
  - result: passed
- Enriched baseline smoke:
  - `python 15min-lead\\wind\\scripts\\run_local_baselines.py --max-turbines 4 --tail-timestamps 12000 --include-tower --include-1min --skip-sarima --output-dir 15min-lead\\wind\\artifacts\\local_debug\\baseline_smoke_enriched_v2`
- Enriched window smoke:
  - `python 15min-lead\\wind\\scripts\\build_window_dataset.py --max-turbines 4 --tail-timestamps 12000 --lookback-steps 32 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\window_smoke_enriched_v2`
- Enriched GRU smoke:
  - `python 15min-lead\\wind\\scripts\\train_gru_baseline.py --epochs 3 --batch-size 128 --max-turbines 4 --tail-timestamps 12000 --include-tower --include-1min --output-dir 15min-lead\\wind\\artifacts\\local_debug\\gru_smoke_enriched_v2`

## Current debug-subset results

### Enriched baseline

From `artifacts/local_debug/baseline_smoke_enriched_v2/baseline_metrics.csv`:

- persistence test RMSE: `0.5649`
- LightGBM test RMSE: `0.7003`

This is an improvement over the earlier non-enriched LightGBM debug run,
but persistence is still the strongest method on this local subset.

### Enriched window dataset

From `artifacts/local_debug/window_smoke_enriched_v2/summary.json`:

- `x_shape = [3827, 32, 4, 181]`
- `y_shape = [3827, 4]`
- `n_train = 535`
- `n_val = 2443`
- `n_test = 849`

### Enriched GRU

From `artifacts/local_debug/gru_smoke_enriched_v2/metrics.csv`:

- val RMSE: `1.3431`
- test RMSE: `0.9928`

The important result for this turn is that the enriched deep-learning
path now trains successfully end to end. Its accuracy is not yet good
enough to beat persistence on the debug subset.

## Risks and open points

- These smoke runs still use only `4` turbines and `12000` timestamps, so
  they are for pipeline validation rather than final model judgment.
- The enriched GRU likely needs full-farm training, feature selection,
  architecture tuning, and possibly a stronger spatio-temporal model.
- Tower humidity at `10 m` still has a small real missing share, which is
  tracked through the missing-indicator features.

## Recommended next step

- Run the same enriched pipeline on the full `46`-turbine xinyang farm.
- Then integrate the next mature deep-learning model, preferably TCN or
  Graph WaveNet, using the already exported window tensors and distance
  adjacency.
