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
  - supports feature presets such as `default_multivariate`, `hub_ws_only`, and `scada_core`
  - now supports `--min-target-coverage` to avoid discarding windows when only a subset of turbine targets is missing
- `scripts/train_gru_baseline.py`
  - trains a local multi-turbine GRU baseline on windowed data
  - supports the same `--include-tower` and `--include-1min` feature ablations
- `scripts/train_tcn_baseline.py`
  - trains a local multi-turbine TCN baseline on the same windowed inputs
  - follows the causal dilated-convolution pattern from the mature locuslab TCN line
- `scripts/train_tcn_from_store.py`
  - trains the TCN baseline from the disk-backed store via lazy window loading
- `scripts/train_gwnet_from_store.py`
  - trains a Graph WaveNet style model from the disk-backed store and saved adjacency
  - supports `distance` or `distance + correlation` fixed graph supports
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
