# Job Notes

- `jobs/slurm/` stores reusable Slurm job scripts.
- `jobs/templates/` stores reusable job templates.
- The server working root is `/s502024280003/gpfs`.
- Each job should request at most `2` GPUs.
- Job scripts should always make input paths, output paths, log paths, and environment activation explicit.
- Current xinyang graph-model jobs:
  - `jobs/slurm/xinyang_build_store_full.slurm`
  - `jobs/slurm/xinyang_train_gwnet_full.slurm`
- Recommended server repo root:
  - `/s502024280003/gpfs/Wind_Prediction`
- Recommended xinyang data upload folder:
  - `/s502024280003/gpfs/Wind_Prediction/data/xinyang/`
