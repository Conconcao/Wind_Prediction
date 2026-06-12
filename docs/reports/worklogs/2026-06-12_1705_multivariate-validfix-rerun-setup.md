# 2026-06-12 Multivariate Validfix Rerun Setup

## What I changed

- Rebuilt the full xinyang multivariate store locally with:
  - `default_multivariate`
  - `include_tower = true`
  - `include_1min = true`
  - repaired masked-target validity logic
- Added dedicated `LSF` rerun scripts so the repaired multivariate
  experiment does not overwrite older pre-fix artifacts:
  - `jobs/lsf/xinyang_build_store_full_validfix.lsf`
  - `jobs/lsf/xinyang_train_gwnet_full_validfix.lsf`
- Updated `15min-lead/wind/README.md` to document the restored window
  counts and the new rerun entry points.

## Why I changed it

- The next recommended experiment after the strong `5976` result is a
  fair rerun of the wider multivariate `GWNet` setup under the repaired
  validity rule.
- The older `xinyang_store_full` / `gwnet_full_run` naming is tied to the
  pre-fix history and could cause confusion or accidental overwrites.
- A dedicated `validfix` naming path makes the comparison much cleaner.

## Local verification

Rebuilt local multivariate store:

- output:
  `15min-lead/wind/artifacts/local_debug/xinyang_store_full_validfix`
- key summary:
  - `181` features
  - `27899` valid windows
  - `18926 / 6032 / 2941` train/val/test windows
  - `min_target_count = 40`
  - `valid_target_count_mean = 44.43`
  - estimated store size about `1.07 GiB`

This confirms that the repaired logic restores the multivariate path to a
healthy sample size and makes it directly comparable with the `5976`
`hub_ws_only` run.

## Practical next step

On the server:

1. `git pull`
2. `bsub < jobs/lsf/xinyang_build_store_full_validfix.lsf`
3. `bsub < jobs/lsf/xinyang_train_gwnet_full_validfix.lsf`

After the run finishes, package it into a dedicated result folder so it
can be compared cleanly against `xinyang_gwnet_hubws_5976`.
