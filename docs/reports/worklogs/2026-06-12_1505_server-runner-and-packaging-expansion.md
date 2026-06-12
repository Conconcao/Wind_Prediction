# 2026-06-12 Server Runner And Packaging Expansion

## What I changed

- Generalized remote result packaging so it can now handle:
  - store-backed joint runs
  - dense-window single-turbine runs without a store summary
- Updated:
  - `15min-lead/wind/src/xinyang_wind15/remote_results.py`
  - `15min-lead/wind/scripts/package_remote_run.py`
  - `15min-lead/wind/tests/test_remote_results.py`
- Added `LSF` scripts for single-turbine controls:
  - `jobs/lsf/xinyang_train_gru_single_turbine.lsf`
  - `jobs/lsf/xinyang_train_tcn_single_turbine.lsf`
- Updated `15min-lead/wind/README.md` with:
  - single-turbine server submission examples
  - packaging examples for dense-window runs

## Why I changed it

- The previous packaging path was tuned for `Graph WaveNet`-style
  store-backed joint runs and assumed every packaged run had a
  `store_summary.json`.
- That assumption breaks for the new single-turbine `GRU/TCN` controls,
  because those controls currently use the dense-window scripts rather
  than the disk-backed store path.
- Without this change, the user could run single-turbine controls on the
  server, but the result-push-back workflow would be inconsistent and
  more manual than the joint-model workflow.

## Verification

- `pytest 15min-lead\\wind\\tests -q`
  - result: `18 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts jobs\\lsf`
  - result: passed

## Practical outcome

We now have one consistent repo workflow for:

- `Graph WaveNet`
- `AGCRN`
- `MTGNN`
- `ModernTCN`
- single-turbine `GRU`
- single-turbine `TCN`

Joint models can package with store summaries, while single-turbine
dense-window runs can package with `--skip-store-summary`.
