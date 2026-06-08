# Transfer Workflow

- Files that need to be uploaded to the server should be staged under
  `transfer/to_hpc/payload/`.
- Large outputs pulled back from the server can be staged under
  `transfer/from_hpc/` before optional archival into `artifacts/from_hpc/`.
- For lightweight experiment summaries that are safe to version, prefer
  packaging them directly inside the repo and pushing them through Git.
- The current xinyang `15min` workflow supports this via
  `15min-lead/wind/scripts/package_remote_run.py`, which packages only
  metrics, summaries, per-turbine CSVs, and tailed logs while excluding
  large arrays and model checkpoints.
- Update the corresponding worklog before uploading or synchronizing
  server artifacts, and record what was transferred and why.
