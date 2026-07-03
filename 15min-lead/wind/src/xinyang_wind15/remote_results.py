"""Utilities for packaging server-side run artifacts into repo-safe summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TRAIN_FILES: tuple[str, ...] = (
    "metrics.json",
    "metrics.csv",
    "training_history.csv",
    "val_per_turbine.csv",
    "test_per_turbine.csv",
    "summary.json",
)

PREDICTION_FILES: tuple[str, ...] = (
    "val_predictions.csv",
    "test_predictions.csv",
)

DEFAULT_MODEL_ARTIFACTS: tuple[str, ...] = (
    "gwnet_baseline.pt",
    "agcrn_baseline.pt",
    "mtgnn_baseline.pt",
    "moderntcn_baseline.pt",
    "gru_baseline.pt",
    "tcn_baseline.pt",
    "cfc_baseline.pt",
)


@dataclass(frozen=True)
class RemoteRunPackageSpec:
    run_name: str
    job_id: str
    train_dir: Path
    store_dir: Path | None
    log_dir: Path
    output_root: Path
    max_log_lines: int = 200
    train_files: tuple[str, ...] = DEFAULT_TRAIN_FILES
    include_predictions: bool = False
    log_stem: str = "xinyang_gwnet_full"
    include_store_summary: bool = True
    model_artifacts: tuple[str, ...] = DEFAULT_MODEL_ARTIFACTS


def tail_lines(path: str | Path, max_lines: int) -> str:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines <= 0:
        return ""
    selected = lines[-max_lines:]
    text = "\n".join(selected)
    if lines:
        text += "\n"
    return text


def _copy_required_files(
    src_dir: Path,
    dest_dir: Path,
    file_names: Iterable[str],
) -> list[dict[str, str]]:
    copied = []
    for file_name in file_names:
        src = src_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing required result file: {src}")
        dest = dest_dir / file_name
        dest.write_bytes(src.read_bytes())
        copied.append({"source": str(src), "target": str(dest)})
    return copied


def package_remote_run(spec: RemoteRunPackageSpec) -> dict[str, object]:
    train_dir = Path(spec.train_dir)
    store_dir = None if spec.store_dir is None else Path(spec.store_dir)
    log_dir = Path(spec.log_dir)
    output_dir = Path(spec.output_root) / spec.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = tuple(spec.train_files)
    if spec.include_predictions:
        train_files = tuple([*train_files, *PREDICTION_FILES])
    copied_train = _copy_required_files(train_dir, output_dir, train_files)

    store_summary_manifest = None
    if spec.include_store_summary:
        if store_dir is None:
            raise ValueError("include_store_summary=True requires store_dir to be provided.")
        store_summary_src = store_dir / "summary.json"
        if not store_summary_src.exists():
            raise FileNotFoundError(f"Missing required store summary: {store_summary_src}")
        store_summary_dest = output_dir / "store_summary.json"
        store_summary_dest.write_bytes(store_summary_src.read_bytes())
        store_summary_manifest = {
            "source": str(store_summary_src),
            "target": str(store_summary_dest),
        }

    out_log_src = log_dir / f"{spec.job_id}.{spec.log_stem}.out"
    err_log_src = log_dir / f"{spec.job_id}.{spec.log_stem}.err"
    if not out_log_src.exists():
        raise FileNotFoundError(f"Missing job stdout log: {out_log_src}")
    if not err_log_src.exists():
        raise FileNotFoundError(f"Missing job stderr log: {err_log_src}")

    out_log_dest = output_dir / out_log_src.name
    err_log_dest = output_dir / err_log_src.name
    out_log_dest.write_text(
        tail_lines(out_log_src, spec.max_log_lines),
        encoding="utf-8",
    )
    err_log_dest.write_text(
        tail_lines(err_log_src, spec.max_log_lines),
        encoding="utf-8",
    )

    manifest = {
        "run_name": spec.run_name,
        "job_id": spec.job_id,
        "train_dir": str(train_dir),
        "store_dir": None if store_dir is None else str(store_dir),
        "log_dir": str(log_dir),
        "output_dir": str(output_dir),
        "max_log_lines": int(spec.max_log_lines),
        "log_stem": spec.log_stem,
        "copied_train_files": copied_train,
        "store_summary": store_summary_manifest,
        "tailed_logs": [
            {"source": str(out_log_src), "target": str(out_log_dest)},
            {"source": str(err_log_src), "target": str(err_log_dest)},
        ],
        "excluded_large_files": [
            *(
                []
                if spec.include_predictions
                else [str(train_dir / file_name) for file_name in PREDICTION_FILES]
            ),
            *[str(train_dir / file_name) for file_name in spec.model_artifacts],
            *(
                []
                if store_dir is None
                else [
                    str(store_dir / "feature_tensor.npy"),
                    str(store_dir / "target_matrix.npy"),
                    str(store_dir / "origin_indices.npy"),
                    str(store_dir / "target_indices.npy"),
                    str(store_dir / "target_mask.npy"),
                    str(store_dir / "split_labels.npy"),
                    str(store_dir / "distance_adjacency.npy"),
                    str(store_dir / "correlation_adjacency.npy"),
                ]
            ),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
