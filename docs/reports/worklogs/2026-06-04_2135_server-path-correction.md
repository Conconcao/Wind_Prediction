# 2026-06-04 21:35 Server Path Correction

## Goal

Correct the server path convention after the user clarified that
`/s502024280003/gpfs/` is only a temporary job workspace and that
persistent files must be stored under `/home3/s502024280003/`.

## What I changed

- Updated the xinyang `15min` workspace README to use
  `/home3/s502024280003/Wind_Prediction` as the persistent server repo
  root and `/home3/s502024280003/Wind_Prediction/data/xinyang/` as the
  data upload target.
- Updated `15min-lead/wind/docs/xinyang_server_layout.md` with the same
  persistent-path convention and an explicit note that `gpfs` is
  temporary and auto-cleaned.
- Updated `jobs/README.md` to document the corrected server storage
  policy.
- Updated `15min-lead/wind/configs/splits/xinyang_7_2_1_server.yaml` so
  the explicit server-side dataset paths now point to `/home3`.
- Updated both Slurm scripts,
  `jobs/slurm/xinyang_build_store_full.slurm` and
  `jobs/slurm/xinyang_train_gwnet_full.slurm`, so their project root and
  log paths now live under `/home3/s502024280003/Wind_Prediction`.
- Added a local-only correction note to `AGENTS.md` so the server path
  rule is preserved in future collaboration.

## Why I changed it

The earlier `gpfs`-based server layout would have risked losing logs,
generated stores, trained model outputs, and uploaded data after
automatic cleanup. Moving all persistent paths to `home3` aligns the
project with the actual server storage policy and avoids accidental data
loss.

## Verification

- Searched the repository for `gpfs`, `/s502024280003`, and `home3` to
  identify affected files before editing.
- Re-checked the updated path-bearing files after editing to confirm the
  active server-facing docs, config, and Slurm scripts now point to
  `/home3/s502024280003/`.

## Risks and next steps

- Historical worklogs still contain the earlier `gpfs` path because they
  record what was written at that time. Future execution should follow
  the corrected `home3` convention instead of those older notes.
- Before the first server submission, the user should clone or pull the
  repo under `/home3/s502024280003/Wind_Prediction` and upload xinyang
  data under `/home3/s502024280003/Wind_Prediction/data/xinyang/`.
