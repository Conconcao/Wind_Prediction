# 2026-06-04 22:35 LSF Conda Nounset Fix

## Goal

Fix the batch-script environment activation failure reported by the user
on the `LSF / bsub` cluster.

## What I changed

- Updated `jobs/lsf/xinyang_build_store_full.lsf` so the script starts
  with `set -eo pipefail` instead of `set -euo pipefail`.
- Wrapped both the `conda activate` branch and the `.venv` activation
  branch with `set +u` before activation and `set -u` after activation.
- Applied the same change to
  `jobs/lsf/xinyang_train_gwnet_full.lsf` so both server jobs use the
  same robust activation behavior.

## Why I changed it

The user reported this batch error:

`geotiff-deactivate.sh: line 5: _CONDA_SET_GEOTIFF_CSV: unbound variable`

That indicates the shell was running with `nounset` enabled during Conda
activation or deactivation, and one of the environment hook scripts
expected an unset variable to be acceptable. Temporarily disabling `-u`
around environment activation keeps the rest of the job script strict
while avoiding this common Conda hook failure mode.

## Verification

- Re-read both `LSF` scripts after editing to confirm `set +u` / `set -u`
  now wraps the activation block.
- Kept the rest of the job parameters, paths, and Python commands
  unchanged so the fix is isolated to environment setup.

## Risks and next steps

- If the cluster uses a nonstandard `bsub` queue or GPU syntax, the next
  failure could still come from the scheduler directives rather than from
  environment activation.
- The failed job should be resubmitted after pulling the updated scripts
  on the server.
