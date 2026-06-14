# Xinyang 15-Min Wind Speed Forecasting

This workspace is for the first round of `15-minute-ahead` wind speed forecasting
experiments on the `xinyang` wind farm.

## Scope

- Primary task: causal deterministic forecasting of turbine-level wind speed
  `15 minutes ahead`
- Site: `xinyang`
- Main data source: `C:\Users\caosh\Desktop\WTPC\data\xinyang`
- Split rule: chronological `7:2:1`

## Layout

- `docs/`: literature notes and experiment plan
- `configs/`: split and model config drafts
- `scripts/`: upcoming local data prep, training, and evaluation scripts
- `references/`: paper list and reference notes
- `artifacts/`: local debug outputs and future experiment artifacts

## Current status

- Literature review and experiment design completed
- Data split boundaries fixed
- Baseline code implemented and smoke-tested locally
- GRU baseline implemented and smoke-tested locally on a debug subset
- TCN baseline implemented and smoke-tested locally on an enriched debug subset
- 1-minute aggregate features and tower-feature ablations wired into the local pipeline
- Enriched window generation and GRU smoke training now run end to end on a debug subset
- Disk-backed window-store path implemented for larger deep-learning runs
- Graph WaveNet style local training path implemented on top of the disk-backed store
- Correlation support graph and server full-run templates prepared for the graph model
- Window-store validity logic updated to use causal feature filling plus masked targets
- AGCRN, MTGNN, and ModernTCN experiment entry points added
- Single-turbine GRU/TCN control runs can now be launched by turbine id
- Single-turbine CfC control run is now available for liquid-network comparison

## Available scripts

- `scripts/run_local_baselines.py`
  - runs persistence, seasonal persistence, LightGBM, and farm-mean ARIMA
  - supports `--include-tower` and `--include-1min`
- `scripts/build_window_dataset.py`
  - builds compressed window tensors for future GRU / graph experiments
  - exports distance adjacency for graph models
- `scripts/build_window_store.py`
  - writes a time-major disk-backed feature store plus valid window indices
  - intended for larger deep-learning runs where dense window materialization is too expensive
  - supports feature presets such as `default_multivariate`, `hub_ws_only`, `direction_wd_only`, `direction_wd_yaw`, `direction_wd_yaw_error`, and `scada_core`
  - now supports `--min-target-coverage` to avoid discarding windows when only a subset of turbine targets is missing
  - now also supports `--include-derived-core` and `--include-spatial-context` for direct-derived physics features
- `scripts/train_gru_baseline.py`
  - trains a local multi-turbine GRU baseline on windowed data
  - supports the same `--include-tower` and `--include-1min` feature ablations
- `scripts/train_tcn_baseline.py`
  - trains a local multi-turbine TCN baseline on the same windowed inputs
  - follows the causal dilated-convolution pattern from the mature locuslab TCN line
- `scripts/train_cfc_baseline.py`
  - trains a local CfC baseline using the official `ncps` implementation
  - intended first for single-turbine controls before any graph-plus-CfC hybrid work
- `scripts/train_tcn_from_store.py`
  - trains the TCN baseline from the disk-backed store via lazy window loading
- `scripts/train_gwnet_from_store.py`
  - trains a Graph WaveNet style model from the disk-backed store and saved adjacency
  - supports `distance` or `distance + correlation` fixed graph supports
  - can also append a wind-direction-driven dynamic support from the latest lookback step
- `scripts/train_agcrn_from_store.py`
  - trains an AGCRN-style adaptive graph recurrent model from the disk-backed store
- `scripts/train_mtgnn_from_store.py`
  - trains an MTGNN-style learned-graph temporal model from the disk-backed store
- `scripts/train_moderntcn_from_store.py`
  - trains a ModernTCN-style non-graph temporal model from the disk-backed store
  - currently intended for `hub_ws_only` stores with one feature per turbine
- `scripts/package_remote_run.py`
  - packages server-side `GraphWaveNet` summary files and tailed logs into a repo-safe folder
  - excludes large binaries and raw store arrays so the packaged result can be committed

## Latest local debug results

- Enriched baseline smoke run:
  `artifacts/local_debug/baseline_smoke_enriched_v2/`
- Enriched window dataset smoke run:
  `artifacts/local_debug/window_smoke_enriched_v2/`
- Enriched GRU smoke run:
  `artifacts/local_debug/gru_smoke_enriched_v2/`
- Enriched TCN smoke run:
  `artifacts/local_debug/tcn_smoke_enriched_v1/`
- Disk-backed store smoke run:
  `artifacts/local_debug/window_store_smoke_enriched_v1/`
- Disk-backed TCN smoke run:
  `artifacts/local_debug/tcn_store_smoke_enriched_v1/`
- Disk-backed Graph WaveNet style smoke run:
  `artifacts/local_debug/gwnet_store_smoke_enriched_v1/`
- Correlation-support Graph WaveNet style smoke run:
  `artifacts/local_debug/gwnet_store_smoke_enriched_v2/`

On the current `4`-turbine debug subset, persistence is still the
strongest baseline. The enriched GRU path now trains successfully, but
it still underperforms persistence and needs full-data training and
tuning before model conclusions should be drawn. The new TCN baseline is
stronger than the current GRU on this subset, but it still does not beat
persistence. The new Graph WaveNet style path is currently the strongest
deep model on the same subset, but it also still trails persistence. The
distance-plus-correlation graph variant is a small improvement over the
distance-only graph run on the current debug subset.

## Local execution note

The current neural-training scripts materialize full window tensors in
memory. That is acceptable for local smoke subsets such as `4` turbines
and `12000` timestamps, but not yet ideal for full-farm local runs.
Use local subsets for debugging first, then move larger deep-learning
runs to the server workflow after local validation. The current GRU and
TCN scripts also include a dense-window memory estimate guard to catch
obviously oversized local runs early.

The new disk-backed store path keeps a single time-major tensor on disk
and slices windows lazily at training time. On the current enriched
feature set with lookback `32`, the estimated size difference for the
full `46`-turbine xinyang farm is roughly:

- dense materialized windows: about `33.93 GiB`
- disk-backed feature store: about `1.07 GiB`

The current store/training path also treats target availability with an
explicit mask instead of requiring every turbine target to be present at
every forecast step. On the full-farm `hub_ws_only` setup, this raises
the valid-window count from the earlier `406`-window bottleneck to
`27899` windows when `--min-target-coverage 0.85` is used.

The repaired validity logic also restores the wider multivariate store.
Locally rebuilding the `include_tower + include_1min` full store with
the current code yields:

- `181` features
- `27899` valid windows
- `18926 / 6032 / 2941` train/val/test windows

To avoid overwriting older pre-fix multivariate artifacts, use:

- `jobs/lsf/xinyang_build_store_full_validfix.lsf`
- `jobs/lsf/xinyang_train_gwnet_full_validfix.lsf`

On the current enriched `4`-turbine debug subset, the deep-model ordering is:

- `Graph WaveNet style + distance/correlation supports` test RMSE: about `0.7924`
- `Graph WaveNet style + distance support` test RMSE: about `0.7939`
- `TCN` test RMSE: about `0.8808`
- `GRU` test RMSE: about `0.9928`
- `persistence` test RMSE: about `0.5649`

## Server layout

Recommended persistent repo root on the server:

- `/home3/s502024280003/Wind_Prediction`

Please upload the required xinyang data to:

- `/home3/s502024280003/Wind_Prediction/data/xinyang/`

Do not store the repository, datasets, logs, model outputs, or other
persistent files under `/s502024280003/gpfs/`. That directory is only a
temporary job workspace and may be cleaned automatically after `7` days.

Supporting server-side files:

- `15min-lead/wind/docs/xinyang_server_layout.md`
- `15min-lead/wind/configs/splits/xinyang_7_2_1_server.yaml`
- `jobs/lsf/xinyang_build_store_full.lsf`
- `jobs/lsf/xinyang_train_gwnet_full.lsf`
- `jobs/lsf/xinyang_build_store_full_validfix.lsf`
- `jobs/lsf/xinyang_train_gwnet_full_validfix.lsf`
- `jobs/lsf/xinyang_build_store_full_derived_ablation.lsf`
- `jobs/lsf/xinyang_train_gwnet_full_derived_ablation.lsf`
- `jobs/lsf/xinyang_build_store_direction_ablation.lsf`
- `jobs/lsf/xinyang_train_gwnet_direction_ablation.lsf`
- `jobs/lsf/xinyang_build_store_direct_derived_ablation.lsf`
- `jobs/lsf/xinyang_train_gwnet_direct_derived_ablation.lsf`
- `jobs/slurm/xinyang_build_store_full.slurm`
- `jobs/slurm/xinyang_train_gwnet_full.slurm`

## Packaging remote results

After a server-side `GraphWaveNet` run finishes, package the lightweight
result summary back into the repo before pushing:

```bash
python 15min-lead/wind/scripts/package_remote_run.py --job-id 5934
git add 15min-lead/wind/results/remote_runs/xinyang_gwnet_5934
git commit -m "Add xinyang gwnet run 5934 summaries"
git push
```

The packaging script copies metrics, training history, per-turbine CSVs,
the store summary, and the last log lines into
`15min-lead/wind/results/remote_runs/<run_name>/`. It intentionally does
not copy large files such as `gwnet_baseline.pt` or the disk-backed store
arrays.

## Recommended pure joint-time-series run

For the next xinyang baseline-comparison round, the most targeted setup
is joint modeling across all `46` turbines using only hub-height wind
speed history:

- node feature: `ws_mean` only
- target: next-step `ws_mean`
- topology: `distance + correlation` supports
- no tower features
- no `1-minute` aggregate features

Server-side `LSF` entry points for this setup:

- `jobs/lsf/xinyang_build_store_hubws_joint.lsf`
- `jobs/lsf/xinyang_train_gwnet_hubws_joint.lsf`
- `jobs/lsf/xinyang_train_agcrn_hubws_joint.lsf`
- `jobs/lsf/xinyang_train_mtgnn_hubws_joint.lsf`
- `jobs/lsf/xinyang_train_moderntcn_hubws_joint.lsf`
- `jobs/lsf/xinyang_train_gru_single_turbine.lsf`
- `jobs/lsf/xinyang_train_tcn_single_turbine.lsf`
- `jobs/lsf/xinyang_train_cfc_single_turbine.lsf`

## Direction and yaw ablations

The shortest targeted follow-up after the current `hub_ws_only` and full
multivariate runs is a compact direction/yaw ladder on the same joint
`GWNet` path:

- `D0`: `hub_ws_only`
- `D1`: `direction_wd_only`
- `D2`: `direction_wd_yaw`
- `D3`: `direction_wd_yaw_error`
- `D4`: `direction_wd_yaw_error` plus dynamic directional support

Preset contents:

- `direction_wd_only`: `ws_mean + wd_sin + wd_cos`
- `direction_wd_yaw`: `D1 + nacelle_sin + nacelle_cos`
- `direction_wd_yaw_error`: `D2 + yaw_error_sin + yaw_error_cos + yaw_error_abs`

The new `LSF` scripts are parameterized so you can switch among these
ablations without editing files. Example submissions:

```bash
FEATURE_PRESET=direction_wd_only STORE_TAG=xinyang_store_direction_wd bsub < jobs/lsf/xinyang_build_store_direction_ablation.lsf
STORE_TAG=xinyang_store_direction_wd RUN_TAG=gwnet_direction_wd_run bsub < jobs/lsf/xinyang_train_gwnet_direction_ablation.lsf

FEATURE_PRESET=direction_wd_yaw STORE_TAG=xinyang_store_direction_wd_yaw bsub < jobs/lsf/xinyang_build_store_direction_ablation.lsf
STORE_TAG=xinyang_store_direction_wd_yaw RUN_TAG=gwnet_direction_wd_yaw_run bsub < jobs/lsf/xinyang_train_gwnet_direction_ablation.lsf

FEATURE_PRESET=direction_wd_yaw_error STORE_TAG=xinyang_store_direction_wd_yaw_error bsub < jobs/lsf/xinyang_build_store_direction_ablation.lsf
STORE_TAG=xinyang_store_direction_wd_yaw_error RUN_TAG=gwnet_direction_wd_yaw_error_run bsub < jobs/lsf/xinyang_train_gwnet_direction_ablation.lsf

STORE_TAG=xinyang_store_direction_wd_yaw_error RUN_TAG=gwnet_direction_wd_yaw_error_dyn_run DYNAMIC_DIRECTIONAL_SUPPORT=1 DIRECTION_SUPPORT_SOURCE=wd_sincos bsub < jobs/lsf/xinyang_train_gwnet_direction_ablation.lsf
```

## Direct-derived feature blocks

Two new optional feature blocks are now wired into both
`build_window_store.py` and `build_window_dataset.py`:

- `--include-derived-core`
  - `derived_ti_15m`, `derived_gust_factor_15m`, `derived_gust_excess_15m`
  - tower-profile shear and veer features such as
    `profile_shear_alpha_10m_125m`, `profile_veer_10m_125m_*`
  - hub-versus-tower mismatch features such as
    `hub_tower_ws_125m_delta`, `hub_tower_wd_125m_*`
- `--include-spatial-context`
  - direction-aware upwind context features based on current wind direction
    and turbine layout
  - examples:
    `ctx_upwind_ws_mean`, `ctx_upwind_power_mean`,
    `ctx_upwind_ws_gap`, `ctx_upwind_nearest_dist_km`

These blocks are meant for the next gain-attribution round on top of the
current `GWNet` line. A compact local smoke example is:

```bash
python 15min-lead/wind/scripts/build_window_store.py \
  --output-dir 15min-lead/wind/artifacts/local_debug/xinyang_store_derived_ctx_smoke \
  --feature-preset scada_core \
  --include-tower \
  --include-derived-core \
  --include-spatial-context \
  --max-turbines 4 \
  --tail-timestamps 64
```

## Server-side direct-derived ablations

For the next full-farm `GWNet` gain-attribution round, use the new
parameterized `LSF` pair:

- `jobs/lsf/xinyang_build_store_direct_derived_ablation.lsf`
- `jobs/lsf/xinyang_train_gwnet_direct_derived_ablation.lsf`

Recommended store and run tags:

- `xinyang_store_derived_core` / `gwnet_derived_core_run`
- `xinyang_store_derived_ctx` / `gwnet_derived_ctx_run`

Example submissions:

```bash
FEATURE_PRESET=scada_core STORE_TAG=xinyang_store_derived_core INCLUDE_TOWER=1 INCLUDE_DERIVED_CORE=1 INCLUDE_SPATIAL_CONTEXT=0 bsub < jobs/lsf/xinyang_build_store_direct_derived_ablation.lsf
STORE_TAG=xinyang_store_derived_core RUN_TAG=gwnet_derived_core_run bsub < jobs/lsf/xinyang_train_gwnet_direct_derived_ablation.lsf

FEATURE_PRESET=scada_core STORE_TAG=xinyang_store_derived_ctx INCLUDE_TOWER=1 INCLUDE_DERIVED_CORE=1 INCLUDE_SPATIAL_CONTEXT=1 bsub < jobs/lsf/xinyang_build_store_direct_derived_ablation.lsf
STORE_TAG=xinyang_store_derived_ctx RUN_TAG=gwnet_derived_ctx_run bsub < jobs/lsf/xinyang_train_gwnet_direct_derived_ablation.lsf
```

After each training job finishes, package the result back into the repo
with one line:

```bash
python 15min-lead/wind/scripts/package_remote_run.py --job-id <jobid> --run-name xinyang_gwnet_derived_core_<jobid> --train-dir 15min-lead/wind/artifacts/server_runs/gwnet_derived_core_run --store-dir 15min-lead/wind/artifacts/server_runs/xinyang_store_derived_core --log-stem xinyang_gwnet_direct_derived
python 15min-lead/wind/scripts/package_remote_run.py --job-id <jobid> --run-name xinyang_gwnet_derived_ctx_<jobid> --train-dir 15min-lead/wind/artifacts/server_runs/gwnet_derived_ctx_run --store-dir 15min-lead/wind/artifacts/server_runs/xinyang_store_derived_ctx --log-stem xinyang_gwnet_direct_derived
```

## Full-plus-derived ablations

The next highest-value check after `6073/6075` is whether the same
direct-derived blocks still add marginal value on top of the current
strong multivariate `6058` line. Use:

- `jobs/lsf/xinyang_build_store_full_derived_ablation.lsf`
- `jobs/lsf/xinyang_train_gwnet_full_derived_ablation.lsf`

These scripts fix the `6058` backbone:

- `feature_preset=default_multivariate`
- `include_tower=1`
- `include_1min=1`
- `include_derived_core=1`

and let `spatial_context` switch on or off.

Recommended submissions:

```bash
STORE_TAG=xinyang_store_full_derived_core INCLUDE_SPATIAL_CONTEXT=0 bsub < jobs/lsf/xinyang_build_store_full_derived_ablation.lsf
STORE_TAG=xinyang_store_full_derived_core RUN_TAG=gwnet_full_derived_core_run bsub < jobs/lsf/xinyang_train_gwnet_full_derived_ablation.lsf

STORE_TAG=xinyang_store_full_derived_ctx INCLUDE_SPATIAL_CONTEXT=1 bsub < jobs/lsf/xinyang_build_store_full_derived_ablation.lsf
STORE_TAG=xinyang_store_full_derived_ctx RUN_TAG=gwnet_full_derived_ctx_run bsub < jobs/lsf/xinyang_train_gwnet_full_derived_ablation.lsf
```

After the training jobs finish, package them with:

```bash
python 15min-lead/wind/scripts/package_remote_run.py --job-id <jobid> --run-name xinyang_gwnet_full_derived_core_<jobid> --train-dir 15min-lead/wind/artifacts/server_runs/gwnet_full_derived_core_run --store-dir 15min-lead/wind/artifacts/server_runs/xinyang_store_full_derived_core --log-stem xinyang_gwnet_full_derived
python 15min-lead/wind/scripts/package_remote_run.py --job-id <jobid> --run-name xinyang_gwnet_full_derived_ctx_<jobid> --train-dir 15min-lead/wind/artifacts/server_runs/gwnet_full_derived_ctx_run --store-dir 15min-lead/wind/artifacts/server_runs/xinyang_store_full_derived_ctx --log-stem xinyang_gwnet_full_derived
```

## Single-turbine pure time-series controls

Use the existing dense-window GRU and TCN scripts with `--turbine-id` to
run one-turbine controls and check whether joint spatial modeling adds
clear value for a selected machine. Example:

```bash
python 15min-lead/wind/scripts/train_gru_baseline.py \
  --turbine-id S29 \
  --feature-columns ws_mean \
  --max-turbines 1

python 15min-lead/wind/scripts/train_tcn_baseline.py \
  --turbine-id S29 \
  --feature-columns ws_mean \
  --max-turbines 1

python 15min-lead/wind/scripts/train_cfc_baseline.py \
  --turbine-id S29 \
  --feature-columns ws_mean \
  --max-turbines 1
```

Server-side `LSF` single-turbine controls default to `S29`, but you can
override the turbine id at submit time:

```bash
TURBINE_ID=S29 bsub < jobs/lsf/xinyang_train_gru_single_turbine.lsf
TURBINE_ID=S29 bsub < jobs/lsf/xinyang_train_tcn_single_turbine.lsf
TURBINE_ID=S29 bsub < jobs/lsf/xinyang_train_cfc_single_turbine.lsf
```

## Packaging non-GWNet runs

The remote packaging script now also supports runs that do not have a
disk-backed store summary, such as single-turbine dense-window GRU/TCN
controls. Example:

```bash
python 15min-lead/wind/scripts/package_remote_run.py \
  --job-id <jobid> \
  --run-name xinyang_gru_single_s29_<jobid> \
  --train-dir 15min-lead/wind/artifacts/server_runs/gru_single_S29 \
  --log-stem xinyang_gru_single \
  --skip-store-summary

python 15min-lead/wind/scripts/package_remote_run.py \
  --job-id <jobid> \
  --run-name xinyang_cfc_single_s29_<jobid> \
  --train-dir 15min-lead/wind/artifacts/server_runs/cfc_single_S29 \
  --log-stem xinyang_cfc_single \
  --skip-store-summary
```

After the training job finishes, package the run with:

```bash
python 15min-lead/wind/scripts/package_remote_run.py \
  --job-id <jobid> \
  --run-name xinyang_gwnet_hubws_<jobid> \
  --train-dir 15min-lead/wind/artifacts/server_runs/gwnet_hubws_joint_run \
  --store-dir 15min-lead/wind/artifacts/server_runs/xinyang_store_hubws_joint \
  --log-stem xinyang_gwnet_hubws
```

## Next implementation steps

1. Run the full `46`-turbine store build and Graph WaveNet style training on the server
2. Compare `distance` versus `distance + correlation` supports on the full run
3. Add full-length ARIMA or seasonal ARIMA comparison where runtime is acceptable
4. Consider richer graph ablations beyond the current Pearson correlation support
