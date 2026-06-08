# 2026-06-08 15:45 Xinyang Hub-WS 5974 Local Analysis

## Goal

Analyze the packaged local copy of the server-side xinyang joint
hub-height wind-speed-only Graph WaveNet run `5974` and compare it
against the earlier multivariate run `5934`.

## What I analyzed

- `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5974/metrics.json`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5974/summary.json`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5974/store_summary.json`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5974/training_history.csv`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5974/test_per_turbine.csv`
- comparison against the packaged `5934` run under
  `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/`

## Main findings

### 1. The run finished successfully and used GPU

- `summary.json` records `"device": "cuda"`.
- The run uses `ws_mean` only as the node feature and still trains
  end-to-end with the same `46`-turbine graph setup.

### 2. The hub-height-only run is cleaner experimentally, but worse in accuracy

- Validation metrics:
  - `RMSE = 0.4908`
  - `R2 = 0.7604`
  - macro `RMSE = 0.4888`
  - macro `R2 = 0.7511`
- Test metrics:
  - `RMSE = 0.5869`
  - `R2 = 0.4483`
  - macro `RMSE = 0.5612`
  - macro `R2 = 0.2489`

Compared with `5934`:

- `5934` test `RMSE = 0.5384` vs `5974` test `RMSE = 0.5869`
- `5934` test `R2 = 0.5357` vs `5974` test `R2 = 0.4483`
- `5934` test macro `RMSE = 0.5148` vs `5974` test macro `RMSE = 0.5612`
- `5934` test macro `R2 = 0.3655` vs `5974` test macro `R2 = 0.2489`

### 3. The valid-window problem did not improve

The most important negative result is that `5974` did **not** recover the
effective sample count.

- `5934` valid windows: `406`
- `5974` valid windows: `406`

Split counts are also unchanged:

- train: `173`
- val: `216`
- test: `17`

This means the bottleneck is not the wide feature set. The current
farm-wide valid-window logic is still the dominant constraint.

### 4. The feature reduction alone does not beat the multivariate run

- `5974` uses only `1` feature: `ws_mean`
- `5934` uses `181` features

Even after stripping the model down to pure hub-height wind speed, the
same `406` windows survive and the final accuracy becomes worse, not
better. This suggests that the multivariate inputs were still providing
useful signal on the surviving windows.

### 5. Per-turbine changes are mostly negative

- `8` turbines improved in test RMSE relative to `5934`
- `38` turbines became worse
- mean per-turbine RMSE change:
  `+0.0464` for `5974 - 5934`

Largest improvements:

- `S37`: `-0.0718`
- `S01`: `-0.0364`
- `S18`: `-0.0206`

Largest degradations:

- `S46`: `+0.1715`
- `S30`: `+0.1584`
- `S45`: `+0.1251`
- `S41`: `+0.1117`

### 6. Training behavior is stable but not materially better

- best validation epoch: `17`
- best validation RMSE: `0.4908`
- final epoch `30` validation RMSE: `0.4971`

This is stable and trainable, but still weaker than the earlier
multivariate run.

## Why these findings matter

`5974` is a useful control experiment because it shows that pure joint
hub-height wind-speed modeling is feasible and operationally simpler.
However, it does not solve the underlying data-efficiency issue and does
not outperform the earlier multivariate run on the current surviving
window set.

The current evidence points to this priority order:

1. fix the valid-window construction
2. regenerate stores with a healthier number of train/val/test windows
3. only then re-evaluate `hub_ws_only` versus wider feature sets

## Verification

- Loaded and compared the packaged JSON and CSV outputs for both runs.
- Computed test-metric deltas per turbine between `5974` and `5934`.
- Confirmed that the store summary keeps the same valid-window counts
  despite the feature reduction.

## Recommended next steps

1. Audit and relax the farm-wide valid-window rule in
   `xinyang_wind15/windows.py`.
2. Regenerate both the `hub_ws_only` and multivariate stores after that
   fix.
3. Re-run the same Graph WaveNet configuration on both feature settings
   so the comparison is made on a realistic sample size.
4. Add a strong persistence baseline on the corrected test windows before
   drawing final conclusions about model value.
