# 2026-06-08 16:05 Joint Hub-Height 46-Turbine Setup

## Goal

Turn the user's chosen direction, joint modeling across all `46` xinyang
turbines using hub-height wind speed only, into a concrete and repeatable
experiment path.

## What I changed

- Added `xinyang_wind15/feature_presets.py` with named presets:
  - `default_multivariate`
  - `hub_ws_only`
  - `scada_core`
- Updated `scripts/build_window_store.py` so the store builder can use
  `--feature-preset hub_ws_only` instead of relying on manual feature
  lists.
- Added `jobs/lsf/xinyang_build_store_hubws_joint.lsf` for the full-farm
  disk-backed store using `ws_mean` only.
- Added `jobs/lsf/xinyang_train_gwnet_hubws_joint.lsf` for the
  corresponding full-farm Graph WaveNet training run.
- Extended `scripts/package_remote_run.py` and
  `xinyang_wind15/remote_results.py` with a configurable `--log-stem` so
  packaging also works for non-default job names such as
  `xinyang_gwnet_hubws`.
- Added tests for:
  - feature preset resolution
  - remote run packaging manifest behavior with log stems
- Updated `15min-lead/wind/README.md` and `jobs/README.md` with the new
  joint hub-height setup and packaging command.

## Why I changed it

The previous full server run used a very wide multivariate feature set and
collapsed to too few valid windows. A full-farm joint model with only
hub-height wind speed is a better next experiment because it:

- preserves the spatial joint-modeling structure across `46` turbines
- avoids the missing-value explosion caused by wide tower and `1-minute`
  feature blocks
- gives a cleaner baseline for comparing graph models against
  persistence, SARIMA, GRU, and TCN

## Verification

- Ran `pytest 15min-lead\wind\tests -q`, which passed with `13` tests.
- Ran `python -m compileall 15min-lead\wind\src 15min-lead\wind\scripts jobs\lsf`,
  which completed successfully.

## Risks and next steps

- This setup still uses the current valid-window rule, so if there are
  any residual `ws_mean` gaps at the farm-wide time grid level, valid
  window count may still shrink, though it should be much healthier than
  the multivariate run.
- After pushing this change, the next server action should be:
  - pull latest code
  - submit `xinyang_build_store_hubws_joint.lsf`
  - submit `xinyang_train_gwnet_hubws_joint.lsf`
  - package the result with the `--log-stem xinyang_gwnet_hubws` command
