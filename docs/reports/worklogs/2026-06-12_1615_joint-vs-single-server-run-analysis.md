# 2026-06-12 Joint Vs Single Server Run Analysis

## What I reviewed

Pulled and analyzed the newly packaged server-side result folders:

- `xinyang_gwnet_hubws_5976`
- `xinyang_agcrn_hubws_6049`
- `xinyang_mtgnn_hubws_6050`
- `xinyang_moderntcn_hubws_6051`
- `xinyang_gru_single_s29_6052`
- `xinyang_tcn_single_s29_6053`

## Main findings

### 1. Joint-model ranking on the repaired `hub_ws_only` full-farm store

Using the common `27899`-window store and masked-target evaluation:

1. `Graph WaveNet`
   - test `RMSE = 0.5313`
   - test `R2 = 0.9383`

2. `AGCRN`
   - test `RMSE = 0.5467`
   - test `R2 = 0.9347`

3. `MTGNN`
   - test `RMSE = 0.5506`
   - test `R2 = 0.9338`

4. `ModernTCN`
   - test `RMSE = 0.5886`
   - test `R2 = 0.9243`

Conclusion:

- `Graph WaveNet` remains the strongest joint model in the current
  comparison.
- `AGCRN` is a competitive second-best baseline and the closest challenger.
- `MTGNN` is slightly behind `AGCRN`, but still strong.
- `ModernTCN` is weaker than the graph models, which supports the idea
  that spatial joint modeling is genuinely helping.

### 2. Single-turbine controls versus joint spatial modeling

Single-turbine `S29` controls:

- `GRU` single:
  - test `RMSE = 0.5788`
  - test `R2 = 0.9254`
- `TCN` single:
  - test `RMSE = 0.5772`
  - test `R2 = 0.9258`

Joint `Graph WaveNet` on turbine `S29`:

- test `RMSE = 0.4847`

This gives a direct S29 gain of roughly:

- `0.0942` RMSE better than single-turbine `GRU`
- `0.0925` RMSE better than single-turbine `TCN`

Conclusion:

- For `S29`, joint spatial modeling provides a clear and practically
  meaningful gain over pure single-turbine time-series prediction.

### 3. Per-turbine comparison against Graph WaveNet

Relative to `Graph WaveNet`:

- `AGCRN` beats it on `6` turbines, but loses on the remaining `40`
- `MTGNN` beats it on `0` turbines
- `ModernTCN` beats it on `0` turbines

Average RMSE gaps versus `Graph WaveNet`:

- `AGCRN`: `+0.0152`
- `MTGNN`: `+0.0191`
- `ModernTCN`: `+0.0573`

This reinforces the overall ranking:

- `GWNet > AGCRN > MTGNN > ModernTCN`

## Data consistency

All joint-model runs use the same repaired store summary:

- `n_valid_windows = 27899`
- `n_train / n_val / n_test = 18926 / 6032 / 2941`
- `min_target_count = 40`
- `valid_target_count_mean = 44.43`

That makes this comparison much fairer than the earlier `406`-window runs.

## Note on duplicated AGCRN folder

There are two AGCRN result folders:

- `xinyang_agcrn_hubws_6049`
- `xinyang_agcrn_hubws_6050`

`6050` appears to be a duplicate packaging of the same AGCRN outputs:

- identical `metrics.json`
- identical `training_history.csv`
- identical best epoch (`11`)

So only `6049` should be treated as the actual AGCRN experiment result.

## Practical conclusion

At this point the evidence supports two statements:

1. `Graph WaveNet` is the best current model on the repaired `hub_ws_only`
   joint setup.
2. Spatial joint modeling provides a real gain over pure single-turbine
   time-series baselines, at least for the tested turbine `S29`.

## Recommended next step

- Re-run the wider multivariate store path with the repaired validity rule
  and compare it directly against `xinyang_gwnet_hubws_5976`.
- If the multivariate model does not beat `5976` clearly, then `hub_ws_only`
  joint modeling may already be the most efficient mainline configuration.
