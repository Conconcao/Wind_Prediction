# Xinyang Server Layout

## Recommended persistent server repo root

- `/home3/s502024280003/Wind_Prediction`

Do not place the repository, datasets, logs, model checkpoints, or
experiment outputs under `/s502024280003/gpfs/`. That directory is only
the temporary job workspace and may be cleaned automatically after `7`
days.

## Recommended data upload target

Please place the xinyang files here:

- `/home3/s502024280003/Wind_Prediction/data/xinyang/`

Required files:

- `ALL_TURBINES_15min_202501-202512_QC2.parquet`
- `ALL_TURBINES_1min_202501-202512.parquet`
- `pre_QC_气象观测数据.xlsx`
- `风机基本信息.csv`

Optional for later ablations:

- `/home3/s502024280003/Wind_Prediction/data/open_nwp/gdex_gfs_xinyang_2025-07-01_2025-12-31_hourly.parquet`

## First server-side jobs

- build the full disk-backed store:
  `jobs/lsf/xinyang_build_store_full.lsf`
- train the Graph WaveNet style model:
  `jobs/lsf/xinyang_train_gwnet_full.lsf`

## Notes

- Both scripts request `1` GPU, which stays within the project rule of at most `2` GPUs per job.
- The current LSF scripts assume the server scheduler is `bsub` / `LSF`.
- The current LSF scripts try to activate the `AIRU-WRF-torch` Conda environment by default.
- Adjust the queue name, GPU resource syntax, or environment activation section if the server uses a different local convention.
