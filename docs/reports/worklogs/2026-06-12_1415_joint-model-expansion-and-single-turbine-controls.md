# 2026-06-12 Joint Model Expansion And Single-Turbine Controls

## What I changed

- Added three new joint-model experiment paths under
  `15min-lead/wind/src/xinyang_wind15/` and `scripts/`:
  - `AGCRN`:
    - `src/xinyang_wind15/agcrn.py`
    - `scripts/train_agcrn_from_store.py`
  - `MTGNN`:
    - `src/xinyang_wind15/mtgnn.py`
    - `scripts/train_mtgnn_from_store.py`
  - `ModernTCN`:
    - `src/xinyang_wind15/modern_tcn.py`
    - `scripts/train_moderntcn_from_store.py`
- All three reuse the existing disk-backed store, masked-target loss, and
  evaluation path.
- Extended `load_scada_15min` and `load_scada_1min` to support explicit
  `turbine_ids`, so one-turbine control experiments can be launched
  cleanly from the same data-loading code.
- Added `--turbine-id` to:
  - `scripts/train_gru_baseline.py`
  - `scripts/train_tcn_baseline.py`
- Added `LSF` scripts for the new joint hub-height-only runs:
  - `jobs/lsf/xinyang_train_agcrn_hubws_joint.lsf`
  - `jobs/lsf/xinyang_train_mtgnn_hubws_joint.lsf`
  - `jobs/lsf/xinyang_train_moderntcn_hubws_joint.lsf`
- Updated:
  - `tests/test_model_shapes.py`
  - `references/implementation_sources.md`
  - `configs/models/model_shortlist.yaml`
  - `15min-lead/wind/README.md`

## Why I changed it

- The user wanted to try `AGCRN`, `MTGNN`, and `ModernTCN` in addition to
  the existing `Graph WaveNet` line.
- The user also wanted two pure time-series single-turbine controls to
  judge whether spatial joint modeling provides meaningful gains.
- Reusing the existing store path keeps the joint-model comparison fair:
  all joint models can now be trained on the same `hub_ws_only` store and
  the same repaired valid-window logic.
- Reusing the existing dense-window `GRU/TCN` scripts for one turbine is
  the cleanest way to create single-machine controls without forcing the
  single-turbine experiments through the full-farm target-coverage rule.

## Source alignment

- `AGCRN` implementation was aligned to the official repo structure:
  `AGCRN.py`, `AGCRNCell.py`, `AGCN.py`
- `MTGNN` implementation was aligned to the official repo structure:
  `net.py`, `layer.py`
- `ModernTCN` implementation was aligned to the official short-term repo
  structure:
  `ModernTCN.py`, `ModernTCN_Layer.py`

All source links were added to `references/implementation_sources.md`.

## Local verification

- `pytest 15min-lead\\wind\\tests -q`
  - result: `17 passed`
- `python -m compileall 15min-lead\\wind\\src 15min-lead\\wind\\scripts jobs\\lsf`
  - result: passed

## Smoke runs

### Joint models on `4` turbines, `hub_ws_only`, store-backed

Store used:
- `15min-lead/wind/artifacts/local_debug/window_store_mask_smoke`

1. `AGCRN`
- output: `artifacts/local_debug/agcrn_store_smoke_v1`
- test `RMSE = 1.9944`
- test `R2 = 0.0708`

2. `MTGNN`
- output: `artifacts/local_debug/mtgnn_store_smoke_v1`
- test `RMSE = 1.1215`
- test `R2 = 0.7062`

3. `ModernTCN`
- output: `artifacts/local_debug/moderntcn_store_smoke_v1`
- test `RMSE = 1.3879`
- test `R2 = 0.5500`

These are only `1`-epoch CPU smoke checks. They confirm the training
chains work; they do not rank the models conclusively.

### Single-turbine pure time-series controls

Selected turbine:
- `S29`

1. `GRU` single-turbine control
- output: `artifacts/local_debug/gru_single_s29_smoke_v1`
- test `RMSE = 1.1702`
- test `R2 = 0.6950`

2. `TCN` single-turbine control
- output: `artifacts/local_debug/tcn_single_s29_smoke_v1`
- test `RMSE = 1.0485`
- test `R2 = 0.7551`

These also are only `1`-epoch local smoke checks, but they confirm that
the single-turbine control path is live and directly comparable in
principle with the joint-model line.

## Practical next step

Recommended server-side order:

1. `AGCRN` on the repaired full-farm `hub_ws_only` store
2. `MTGNN` on the same store
3. `ModernTCN` on the same store
4. One selected single-turbine `GRU` and `TCN` control, ideally on the
   same turbine later chosen for qualitative comparison

The most meaningful comparison after those runs will be:

- `Graph WaveNet` joint
- `AGCRN` joint
- `MTGNN` joint
- `ModernTCN` joint
- `GRU` single-turbine
- `TCN` single-turbine

That should make the value of spatial joint modeling much easier to
judge.
