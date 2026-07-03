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
        "val_predictions.csv",
        "test_predictions.csv",
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
    assert not (run_dir / "val_predictions.csv").exists()
    assert not (run_dir / "test_predictions.csv").exists()
    assert (run_dir / "5934.xinyang_gwnet_full.out").read_text(encoding="utf-8") == "out-3\nout-4\n"
    assert (run_dir / "5934.xinyang_gwnet_full.err").read_text(encoding="utf-8") == "err-3\nerr-4\n"
    assert not (run_dir / "gwnet_baseline.pt").exists()

    manifest_path = run_dir / "manifest.json"
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["run_name"] == "xinyang_gwnet_5934"
    assert loaded_manifest["log_stem"] == "xinyang_gwnet_full"
    assert any(item.endswith("val_predictions.csv") for item in loaded_manifest["excluded_large_files"])
    assert any(item.endswith("test_predictions.csv") for item in loaded_manifest["excluded_large_files"])
    assert any(item.endswith("gwnet_baseline.pt") for item in loaded_manifest["excluded_large_files"])
    assert manifest["output_dir"] == str(run_dir)


def test_package_remote_run_can_skip_store_summary_for_dense_runs(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts" / "server_runs" / "gru_single_run"
    log_dir = tmp_path / "logs"
    output_root = tmp_path / "results" / "remote_runs"
    train_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    for file_name in (
        "metrics.json",
        "metrics.csv",
        "training_history.csv",
        "val_per_turbine.csv",
        "test_per_turbine.csv",
        "val_predictions.csv",
        "test_predictions.csv",
        "summary.json",
    ):
        (train_dir / file_name).write_text(f"content for {file_name}\n", encoding="utf-8")

    (log_dir / "6001.xinyang_gru_single.out").write_text("ok\n", encoding="utf-8")
    (log_dir / "6001.xinyang_gru_single.err").write_text("", encoding="utf-8")

    manifest = package_remote_run(
        RemoteRunPackageSpec(
            run_name="xinyang_gru_single_6001",
            job_id="6001",
            train_dir=train_dir,
            store_dir=None,
            log_dir=log_dir,
            output_root=output_root,
            log_stem="xinyang_gru_single",
            include_store_summary=False,
        )
    )

    run_dir = output_root / "xinyang_gru_single_6001"
    assert (run_dir / "metrics.json").exists()
    assert not (run_dir / "store_summary.json").exists()
    loaded_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["store_summary"] is None
    assert manifest["store_dir"] is None


def test_package_remote_run_can_include_predictions_when_requested(tmp_path: Path) -> None:
    train_dir = tmp_path / "artifacts" / "server_runs" / "huaian_seq_run"
    store_dir = tmp_path / "artifacts" / "server_runs" / "huaian_store"
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
        "val_predictions.csv",
        "test_predictions.csv",
        "summary.json",
    ):
        (train_dir / file_name).write_text(f"content for {file_name}\n", encoding="utf-8")

    (store_dir / "summary.json").write_text('{"store": true}\n', encoding="utf-8")
    (log_dir / "6301.huaian_1min_lstm.out").write_text("ok\n", encoding="utf-8")
    (log_dir / "6301.huaian_1min_lstm.err").write_text("", encoding="utf-8")

    manifest = package_remote_run(
        RemoteRunPackageSpec(
            run_name="huaian_lstm_1min_raw_ws_6301",
            job_id="6301",
            train_dir=train_dir,
            store_dir=store_dir,
            log_dir=log_dir,
            output_root=output_root,
            log_stem="huaian_1min_lstm",
            include_predictions=True,
        )
    )

    run_dir = output_root / "huaian_lstm_1min_raw_ws_6301"
    assert (run_dir / "val_predictions.csv").exists()
    assert (run_dir / "test_predictions.csv").exists()
    loaded_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert not any(item.endswith("val_predictions.csv") for item in loaded_manifest["excluded_large_files"])
    assert not any(item.endswith("test_predictions.csv") for item in loaded_manifest["excluded_large_files"])
    assert manifest["output_dir"] == str(run_dir)
