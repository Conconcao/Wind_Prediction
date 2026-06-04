# 2026-06-04 22:50 Unicode Path Fallback

## Goal

Fix the server-side failure caused by mojibake in configured Chinese file
names for the xinyang tower and turbine-metadata inputs.

## What I changed

- Added fallback path resolution in
  `15min-lead/wind/src/xinyang_wind15/loading.py`.
- The loaders for `15min` SCADA, `1min` SCADA, tower meteorology, and
  turbine metadata now first try the configured path, then search the
  same directory for a known filename pattern if the configured path is
  missing.
- Added a regression test in
  `15min-lead/wind/tests/test_pipeline_utils.py` that simulates mojibake
  filenames such as `pre_QC_姘旇薄瑙傛祴鏁版嵁.xlsx` and confirms the
  loader resolves them back to the actual files
  `pre_QC_气象观测数据.xlsx` and `风机基本信息.csv`.

## Why I changed it

The server job failed because the configured path for the tower Excel
file was a mojibake variant of the true filename. Even if the repo
contains the correct UTF-8 config locally, different terminals or copy
paths can still reintroduce this class of issue. Resolving by directory
and known pattern makes the pipeline much more tolerant.

## Verification

- Re-read the active server YAML locally and confirmed the intended UTF-8
  filenames are correct there.
- Added a dedicated automated test that exercises the fallback resolver
  against mojibake-like paths.

## Risks and next steps

- The fallback logic assumes the xinyang data directory does not contain
  multiple conflicting files matching the same semantic pattern.
- After pulling the updated code on the server, the failed store job
  should be resubmitted.
