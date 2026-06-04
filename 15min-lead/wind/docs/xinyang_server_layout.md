# Xinyang Server Layout

## Recommended server repo root

- `/s502024280003/gpfs/Wind_Prediction`

## Recommended data upload target

Please place the xinyang files here:

- `/s502024280003/gpfs/Wind_Prediction/data/xinyang/`

Required files:

- `ALL_TURBINES_15min_202501-202512_QC2.parquet`
- `ALL_TURBINES_1min_202501-202512.parquet`
- `pre_QC_姘旇薄瑙傛祴鏁版嵁.xlsx`
- `椋庢満鍩烘湰淇℃伅.csv`

Optional for later ablations:

- `/s502024280003/gpfs/Wind_Prediction/data/open_nwp/gdex_gfs_xinyang_2025-07-01_2025-12-31_hourly.parquet`

## First server-side jobs

- build the full disk-backed store:
  `jobs/slurm/xinyang_build_store_full.slurm`
- train the Graph WaveNet style model:
  `jobs/slurm/xinyang_train_gwnet_full.slurm`

## Notes

- Both scripts request `1` GPU, which stays within the project rule of at most `2` GPUs per job.
- Adjust the environment activation section in the job scripts if the server uses a different Python or Conda setup.
