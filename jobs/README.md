# Job Notes

- `jobs/lsf/` stores reusable LSF job scripts for `bsub`-based clusters.
- `jobs/slurm/` stores reusable Slurm job scripts.
- `jobs/templates/` stores reusable job templates.
- The persistent server root should be `/home3/s502024280003`.
- Do not store the repository, datasets, logs, checkpoints, or experiment outputs under `/s502024280003/gpfs/`; that path is only a temporary job workspace and may be cleaned automatically after `7` days.
- Each job should request at most `2` GPUs.
- Job scripts should always make input paths, output paths, log paths, and environment activation explicit.
- Current xinyang graph-model jobs for `bsub` / `LSF`:
  - `jobs/lsf/xinyang_build_store_full.lsf`
  - `jobs/lsf/xinyang_train_gwnet_full.lsf`
- Optional cross-cluster Slurm variants:
  - `jobs/slurm/xinyang_build_store_full.slurm`
  - `jobs/slurm/xinyang_train_gwnet_full.slurm`
- Recommended server repo root:
  - `/home3/s502024280003/Wind_Prediction`
- Recommended xinyang data upload folder:
  - `/home3/s502024280003/Wind_Prediction/data/xinyang/`
