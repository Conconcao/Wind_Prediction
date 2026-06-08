# 2026-06-08 Window Validity And Masked-Target Fix

## What I changed

- Updated the disk-backed window-store path to stop discarding a full window
  whenever a single turbine target is missing at the forecast timestamp.
- Added causal feature filling inside
  `15min-lead/wind/src/xinyang_wind15/window_store.py`.
  - Turbine-level features such as `ws_mean` are forward-filled per turbine.
  - Timestamp-level tower features are broadcast across turbines at the same timestamp.
  - Missing-indicator columns are preserved and filled consistently.
- Added `target_mask.npy` to the window store and kept `target_matrix.npy`
  with raw missing targets intact.
- Extended `compute_valid_window_indices` so the store can keep a window when
  the target step still has enough observed turbines.
- Added `--min-target-coverage` to
  `15min-lead/wind/scripts/build_window_store.py`.
  - Current default is `0.85`.
  - On the `46`-turbine xinyang farm this means at least `40` turbine targets
    must be observed at the forecast step.
- Updated `WindowStoreDataset`, standardization, evaluation, and the
  `TCN/GWNet` store-training scripts to use explicit target masks.
  - Loss is now computed only on observed targets.
  - Validation and test metrics are also computed only on observed targets.
- Made `load_window_store` backward-compatible with older stores that do not
  yet contain `target_mask.npy`.
- Added unit coverage for masked targets and causal filling in
  `15min-lead/wind/tests/test_window_store.py`.
- Updated `15min-lead/wind/README.md` with the new window-validity behavior.

## Why I changed it

- Local analysis of runs `5934` and `5974` showed that both experiments were
  being throttled by the same bottleneck: only `406` valid windows survived
  after strict all-turbine validity filtering.
- A direct inspection of the xinyang `ws_mean` pivot showed:
  - only `9995 / 34208` timestamps are fully complete across all `46` turbines
  - after causal forward fill, feature availability is no longer the main issue
  - the real bottleneck is requiring every turbine target to be present at the
    forecast step
- This meant that changing feature width alone could not fix the sample-size
  problem. The validity rule itself had to change.

## Local verification

- `pytest 15min-lead\\wind\\tests -q`
  - result: `14 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts jobs\\lsf`
  - result: passed
- End-to-end masked-store smoke:
  - built `15min-lead/wind/artifacts/local_debug/window_store_mask_smoke`
  - trained `15min-lead/wind/artifacts/local_debug/gwnet_store_mask_smoke_fast`
  - verified that the new masked dataset path trains and evaluates successfully
- Full-farm `hub_ws_only` store rebuild:
  - output: `15min-lead/wind/artifacts/local_debug/xinyang_store_hubws_joint_validfix`
  - valid windows increased from `406` to `27899`
  - split counts became `18926 / 6032 / 2941`
  - minimum valid target count per kept window: `40`
  - mean valid target count per kept window: `44.43`

## Practical next step

- Rebuild the server-side xinyang store with the updated code.
- Rerun both:
  - the `hub_ws_only` joint model
  - the wider multivariate GWNet setup
- Compare the new results against `5934` and `5974` only after those reruns,
  because the old scores were produced under the overly strict validity rule.
