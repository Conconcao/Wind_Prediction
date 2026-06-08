# 2026-06-08 15:22 Xinyang GWNet 5934 Local Analysis

## Goal

Analyze the packaged local copy of the server-side xinyang `15-minute`
Graph WaveNet run `5934` and summarize what is reliable, what is risky,
and what should happen next.

## What I analyzed

- `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/metrics.json`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/summary.json`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/store_summary.json`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/training_history.csv`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/val_per_turbine.csv`
- `15min-lead/wind/results/remote_runs/xinyang_gwnet_5934/test_per_turbine.csv`

## Main findings

### 1. The run finished successfully and used GPU

- `summary.json` records `"device": "cuda"`.
- The packaged result includes the expected metrics, history, and
  per-turbine outputs.

### 2. Validation looks strong, but test is much weaker

- Validation:
  - `RMSE = 0.4467`
  - `R2 = 0.8015`
  - macro `RMSE = 0.4450`
  - macro `R2 = 0.7935`
- Test:
  - `RMSE = 0.5384`
  - `R2 = 0.5357`
  - macro `RMSE = 0.5148`
  - macro `R2 = 0.3655`

This is not a failure, but it is a meaningful generalization drop from
validation to test.

### 3. The biggest problem is the extremely small number of valid windows

- `store_summary.json` reports:
  - `feature_tensor_shape[0] = 34208` timestamps
  - `n_valid_windows = 406`
  - `n_train = 173`
  - `n_val = 216`
  - `n_test = 17`

With `lookback_steps = 32` and `horizon_steps = 1`, a nearly complete
year of `15-minute` timestamps should normally yield far more than `406`
valid windows. The final test set has only `17` windows, which is far
too small for a stable full-farm evaluation.

### 4. The train/val/test window ratios are inconsistent with the intended 7:2:1 split

- Intended split: `70% / 20% / 10%`
- Actual valid-window counts:
  - train: `173` (`42.6%`)
  - val: `216` (`53.2%`)
  - test: `17` (`4.2%`)

This strongly suggests that missing-value filtering, rather than the
chronological split rule itself, is dominating which windows survive.

### 5. The current valid-window logic is probably too strict

`xinyang_wind15/windows.py` marks a timestamp invalid if **any**
feature/turbine entry at that time is `NaN`, via:

- `invalid_feature_steps = np.isnan(feature_tensor).any(axis=(1, 2))`

That means one missing feature anywhere in the `46 x 181` feature grid
invalidates that timestamp for every turbine and for every candidate
window touching it. With tower features and `1-minute` aggregates
included, this can easily collapse the usable sample count.

### 6. Training behavior is reasonable, but early stopping would help

- `training_history.csv` has `30` epochs.
- Best validation RMSE occurred at epoch `16`: `0.4467`.
- Final epoch `30` validation RMSE was `0.4593`.

The model does not appear to explode, but the best checkpoint arrives
well before the final epoch, so explicit early stopping would be
appropriate.

### 7. Test performance is heterogeneous across turbines

- Mean per-turbine test RMSE: `0.5148`
- Std of per-turbine test RMSE: `0.1595`
- Mean per-turbine test R2: `0.3655`
- Two turbines have negative test `R2`

Worst test RMSE turbines:

- `S44`: `RMSE = 0.9206`, `R2 = 0.2172`
- `S10`: `RMSE = 0.9093`, `R2 = -0.2665`
- `S04`: `RMSE = 0.7950`, `R2 = -0.0213`

Best test RMSE turbines:

- `S29`: `RMSE = 0.3007`, `R2 = 0.5477`
- `S01`: `RMSE = 0.3014`, `R2 = 0.4519`
- `S32`: `RMSE = 0.3227`, `R2 = 0.4505`

### 8. Validation-to-test degradation is concentrated in a few turbines

Largest RMSE increases from validation to test:

- `S44`: `+0.4968`
- `S10`: `+0.3774`
- `S04`: `+0.3757`
- `S35`: `+0.3105`

This suggests the poor aggregate test result is not uniformly shared
across all turbines.

## Why these findings matter

The `5934` run is useful because it proves the full server path works and
the model can train on GPU end to end. However, the current test metric
should not yet be treated as the definitive score for the xinyang
Graph WaveNet experiment, because the effective test set is too small
and the current valid-window rule is likely discarding most of the year.

## Verification

- Loaded and summarized the packaged JSON and CSV outputs locally.
- Re-read `xinyang_wind15/window_store.py` and
  `xinyang_wind15/windows.py` to trace how valid windows are computed.
- Computed local descriptive statistics for:
  - validation/test per-turbine metrics
  - training history
  - validation-to-test per-turbine gaps

## Recommended next steps

1. Relax or redesign the valid-window rule so a single missing feature at
   one turbine does not invalidate the entire farm-wide timestamp.
2. Regenerate the xinyang store and confirm that valid windows recover to
   a scale consistent with the intended `7:2:1` chronological split.
3. Re-run the same `GraphWaveNet` configuration after the window-count
   issue is fixed before making architecture-level conclusions.
4. Add a full-run persistence baseline and compare against it on the same
   corrected test windows.
5. Add early stopping based on validation RMSE to avoid unnecessary late
   epochs once the best checkpoint has already been reached.
