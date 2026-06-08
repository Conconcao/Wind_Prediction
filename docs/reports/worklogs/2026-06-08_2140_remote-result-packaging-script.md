# 2026-06-08 21:40 Remote Result Packaging Script

## Goal

Reduce the friction of bringing server-side experiment results back to
the local machine by packaging repo-safe summaries directly inside the
repository, so the user can push from the server and pull locally.

## What I changed

- Added `15min-lead/wind/src/xinyang_wind15/remote_results.py` with
  reusable helpers for packaging a completed remote run.
- Added `15min-lead/wind/scripts/package_remote_run.py` as the CLI entry
  point for server-side packaging.
- Added `15min-lead/wind/tests/test_remote_results.py` to verify that the
  packaging flow copies the expected files, tails logs correctly, and
  excludes large binaries such as `gwnet_baseline.pt`.
- Added `15min-lead/wind/results/remote_runs/.gitkeep` to establish the
  versionable destination for packaged server summaries.
- Updated `15min-lead/wind/README.md` with the packaging command and the
  expected result directory.
- Updated `transfer/README.md` so lightweight result summaries now
  prefer Git-based synchronization when safe.
- Updated the local-only `AGENTS.md` rules to reflect the preferred
  workflow for versionable remote results.

## Why I changed it

The manual copy list for every completed server run was repetitive and
error-prone. A dedicated packaging script turns the server-to-local sync
into a short, repeatable workflow while still protecting the repository
from large arrays, checkpoints, and raw data.

## Verification

- Added an automated test for the packaging helper with synthetic run
  outputs and logs.
- Ran `pytest 15min-lead\wind\tests -q`, which passed with `11` tests.
- Ran `python -m compileall 15min-lead\wind\src 15min-lead\wind\scripts`,
  which completed successfully.

## Risks and next steps

- The current script is intentionally focused on the xinyang `GraphWaveNet`
  run layout and log naming. If later experiments use different output
  directories or job-name conventions, the CLI defaults may need to be
  extended.
- After this script is pushed, the user can run it directly on the
  server with `python 15min-lead/wind/scripts/package_remote_run.py --job-id <jobid>`.
