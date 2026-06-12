# 2026-06-11 Xinyang Hub-WS 5976 Local Analysis

## What I reviewed

- Pulled the latest remote result package:
  `15min-lead/wind/results/remote_runs/xinyang_gwnet_hubws_5976/`
- Compared it against:
  - `xinyang_gwnet_hubws_5974`
  - `xinyang_gwnet_5934`

## Key result

This is the first `hub_ws_only` joint-model result produced after the
window-validity fix, and it is the first one that can be treated as a
meaningful full-farm score.

Test metrics from `metrics.json`:

- `RMSE = 0.5313`
- `MAE = 0.3888`
- `R2 = 0.9383`
- macro `RMSE = 0.5309`
- macro `R2 = 0.9377`

Validation metrics:

- `RMSE = 0.5167`
- `R2 = 0.9302`

The validation/test gap is small, so this run looks stable rather than
severely overfit.

## Why this run matters

The previous `5974` hub-height-only run was bottlenecked by the old
window-validity rule and had only:

- `406` valid windows total
- `17` test windows

This new `5976` run uses the repaired store logic and now has:

- `27899` valid windows total
- `18926 / 6032 / 2941` train/val/test windows
- `131102` observed turbine-target pairs in the test split
- test target coverage of about `96.9%`

This means `5976` is much more trustworthy than `5974`, and also much
more trustworthy than `5934`, because the earlier runs were scored on a
tiny and unstable test sample.

## Comparison to earlier runs

Compared with `5974`:

- test `RMSE` improved from `0.5869` to `0.5313`
- absolute `RMSE` gain: `0.0556`
- relative `RMSE` gain: about `9.47%`
- test `R2` improved from `0.4483` to `0.9383`

Compared with `5934`:

- test `RMSE` improved from `0.5384` to `0.5313`
- absolute `RMSE` gain: `0.0071`
- relative `RMSE` gain: about `1.32%`
- test `R2` improved from `0.5357` to `0.9383`

The `5934` comparison is not apples-to-apples, because `5934` was still
running under the old strict validity rule and only had `17` test
windows. I therefore treat `5976` as the new reference run rather than
as a simple incremental improvement over `5934`.

## Per-turbine notes

Largest `RMSE` improvements versus `5974`:

- `S44`: `1.0154 -> 0.5041`
- `S10`: `0.9343 -> 0.5460`
- `S46`: `0.8521 -> 0.4878`
- `S30`: `0.8326 -> 0.5269`
- `S04`: `0.8540 -> 0.5512`

Worst turbines in `5976` by test RMSE are still fairly tight:

- `S41`: `0.5801`
- `S36`: `0.5773`
- `S35`: `0.5708`
- `S40`: `0.5706`
- `S39`: `0.5668`

Even these turbines still have test `R2` values around `0.93+`, so there
is no obvious catastrophic node failure in this run.

## Training notes

- Best validation RMSE occurred at epoch `20`
- Best validation RMSE: `0.5167`
- Training was run for `30` epochs total

This suggests that adding explicit early stopping would be worthwhile,
but it is not an urgent blocker because the selected checkpoint already
comes from the best validation epoch.

## Conclusion

The repaired window-store logic worked as intended. The `5976`
`hub_ws_only` joint Graph WaveNet run is now a credible baseline and
should replace `5974` as the main reference for the pure time-series
joint-model line.

## Recommended next step

- Rebuild and rerun the wider multivariate `5934`-style experiment under
  the same repaired validity logic.
- Then compare:
  - `hub_ws_only joint GWNet`
  - `multivariate GWNet`

That next comparison will be much fairer than anything based on the old
`406`-window runs.
