from __future__ import annotations

import json
from pathlib import Path

from xinyang_wind15.remote_results import RemoteRunPackageSpec, package_remote_run


def test_package_remote_run_collects_expected_summary_files(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts" / "server_runs" / "gwnet_full_run"
    store_dir = tmp_path / "artifacts" / "server_runs" / "xinyang_store_full"
    log_dir = tmp_path / "logs"
    output_root = tmp_path / "results" / "remote_runs"
    train_dir.mkdir(parents=True)
    store_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    for file_name in (
        "metrics.json",
        "metrics.csv",
        "training_history.csv",
        "val_per_turbine.csv",
        "test_per_turbine.csv",
        "summary.json",
    ):
        (train_dir / file_name).write_text(f"content for {file_name}\n", encoding="utf-8")

    (store_dir / "summary.json").write_text('{"store": true}\n', encoding="utf-8")
    (train_dir / "gwnet_baseline.pt").write_bytes(b"large-binary")
    (log_dir / "5934.xinyang_gwnet_full.out").write_text(
        "\n".join(f"out-{i}" for i in range(5)) + "\n",
        encoding="utf-8",
    )
    (log_dir / "5934.xinyang_gwnet_full.err").write_text(
        "\n".join(f"err-{i}" for i in range(5)) + "\n",
        encoding="utf-8",
    )

    manifest = package_remote_run(
        RemoteRunPackageSpec(
            run_name="xinyang_gwnet_5934",
            job_id="5934",
            train_dir=train_dir,
            store_dir=store_dir,
            log_dir=log_dir,
            output_root=output_root,
            max_log_lines=2,
        )
    )

    run_dir = output_root / "xinyang_gwnet_5934"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "store_summary.json").exists()
    assert (run_dir / "5934.xinyang_gwnet_full.out").read_text(encoding="utf-8") == "out-3\nout-4\n"
    assert (run_dir / "5934.xinyang_gwnet_full.err").read_text(encoding="utf-8") == "err-3\nerr-4\n"
    assert not (run_dir / "gwnet_baseline.pt").exists()

    manifest_path = run_dir / "manifest.json"
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["run_name"] == "xinyang_gwnet_5934"
    assert any(item.endswith("gwnet_baseline.pt") for item in loaded_manifest["excluded_large_files"])
    assert manifest["output_dir"] == str(run_dir)
