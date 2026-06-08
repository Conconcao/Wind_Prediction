"""Disk-backed window-store utilities for larger deep-learning experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .settings import SplitBounds
from .splits import assign_split_labels
from .windows import compute_valid_window_indices


TIMESTAMP_LEVEL_FEATURES = {
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
}


def _is_timestamp_level_feature(feature_name: str) -> bool:
    return feature_name in TIMESTAMP_LEVEL_FEATURES or feature_name.startswith("tower_")


def _materialize_feature_pivot(
    scada_df: pd.DataFrame,
    *,
    feature_name: str,
    timestamps_index: pd.Index,
    turbine_order: list[str],
) -> pd.DataFrame:
    pivot = (
        scada_df.pivot(index="timestamp", columns="turbine_id", values=feature_name)
        .reindex(index=timestamps_index, columns=turbine_order)
    )

    if _is_timestamp_level_feature(feature_name):
        pivot = pivot.ffill(axis=1).bfill(axis=1)
        if feature_name.endswith("_missing"):
            pivot = pivot.fillna(1.0)
        return pivot

    if feature_name.endswith("_missing"):
        return pivot.fillna(1.0)

    return pivot.ffill()


def write_window_store(
    scada_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    lookback_steps: int,
    horizon_steps: int,
    split_bounds: SplitBounds,
    output_dir: str | Path,
    min_target_coverage: float = 1.0,
) -> dict[str, Any]:
    """Write a time-major feature store and valid window index metadata to disk."""
    if not 0.0 < float(min_target_coverage) <= 1.0:
        raise ValueError("min_target_coverage must be in the interval (0, 1].")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    turbine_order = sorted(scada_df["turbine_id"].unique())
    timestamps = sorted(scada_df["timestamp"].unique())
    timestamps_index = pd.Index(timestamps)
    n_timestamps = len(timestamps)
    n_turbines = len(turbine_order)
    n_features = len(feature_columns)

    feature_path = out_dir / "feature_tensor.npy"
    target_path = out_dir / "target_matrix.npy"
    feature_tensor = open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_timestamps, n_turbines, n_features),
    )
    target_matrix = open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_timestamps, n_turbines),
    )

    for feature_idx, feature in enumerate(feature_columns):
        pivot = _materialize_feature_pivot(
            scada_df,
            feature_name=feature,
            timestamps_index=timestamps_index,
            turbine_order=turbine_order,
        )
        feature_tensor[:, :, feature_idx] = pivot.to_numpy(dtype=np.float32)

    target_pivot = (
        scada_df.pivot(index="timestamp", columns="turbine_id", values=target_column)
        .reindex(index=timestamps_index, columns=turbine_order)
    )
    target_mask = (~target_pivot.isna()).to_numpy(dtype=bool)
    target_matrix[:, :] = target_pivot.to_numpy(dtype=np.float32)
    feature_tensor.flush()
    target_matrix.flush()
    np.save(out_dir / "target_mask.npy", target_mask)

    feature_tensor_ro = np.load(feature_path, mmap_mode="r")
    target_matrix_ro = np.load(target_path, mmap_mode="r")
    min_target_count = int(math.ceil(float(min_target_coverage) * n_turbines))
    origin_indices, target_indices = compute_valid_window_indices(
        feature_tensor_ro,
        target_matrix_ro,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        target_mask=target_mask,
        min_target_count=min_target_count,
    )
    target_times = np.asarray(timestamps_index[target_indices], dtype="datetime64[ns]")
    split_labels = assign_split_labels(
        pd.Series(pd.to_datetime(target_times)),
        split_bounds,
    ).to_numpy(dtype="U8")

    timestamps_array = np.asarray(timestamps_index, dtype="datetime64[ns]")
    np.save(out_dir / "timestamps.npy", timestamps_array)
    np.save(out_dir / "origin_indices.npy", origin_indices)
    np.save(out_dir / "target_indices.npy", target_indices)
    np.save(out_dir / "split_labels.npy", split_labels)

    valid_target_counts = (
        target_mask[target_indices].sum(axis=1).astype(np.int64)
        if len(target_indices) > 0
        else np.empty(0, dtype=np.int64)
    )
    metadata = {
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "turbine_order": turbine_order,
        "lookback_steps": int(lookback_steps),
        "horizon_steps": int(horizon_steps),
        "min_target_coverage": float(min_target_coverage),
        "min_target_count": int(min_target_count),
        "feature_tensor_shape": [int(n_timestamps), int(n_turbines), int(n_features)],
        "target_matrix_shape": [int(n_timestamps), int(n_turbines)],
        "target_mask_shape": [int(n_timestamps), int(n_turbines)],
        "n_valid_windows": int(len(origin_indices)),
        "n_train": int((split_labels == "train").sum()),
        "n_val": int((split_labels == "val").sum()),
        "n_test": int((split_labels == "test").sum()),
        "valid_target_count_min": int(valid_target_counts.min()) if len(valid_target_counts) else 0,
        "valid_target_count_mean": float(valid_target_counts.mean()) if len(valid_target_counts) else 0.0,
        "valid_target_count_max": int(valid_target_counts.max()) if len(valid_target_counts) else 0,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_window_store(
    store_dir: str | Path,
    *,
    mmap_mode: str = "r",
) -> dict[str, Any]:
    """Load a previously saved window store."""
    store_path = Path(store_dir)
    metadata = json.loads((store_path / "metadata.json").read_text(encoding="utf-8"))
    target_matrix = np.load(store_path / "target_matrix.npy", mmap_mode=mmap_mode)
    target_mask_path = store_path / "target_mask.npy"
    if target_mask_path.exists():
        target_mask = np.load(target_mask_path, mmap_mode=mmap_mode)
    else:
        target_mask = ~np.isnan(np.asarray(target_matrix))
    return {
        "feature_tensor": np.load(store_path / "feature_tensor.npy", mmap_mode=mmap_mode),
        "target_matrix": target_matrix,
        "target_mask": target_mask,
        "timestamps": np.load(store_path / "timestamps.npy", mmap_mode=mmap_mode),
        "origin_indices": np.load(store_path / "origin_indices.npy", mmap_mode=mmap_mode),
        "target_indices": np.load(store_path / "target_indices.npy", mmap_mode=mmap_mode),
        "split_labels": np.load(store_path / "split_labels.npy", mmap_mode=mmap_mode),
        "metadata": metadata,
    }


def estimate_window_store_bytes(
    *,
    n_timestamps: int,
    n_turbines: int,
    n_features: int,
    dtype_bytes: int = 4,
) -> int:
    """Estimate bytes required by the disk-backed feature and target store."""
    feature_bytes = n_timestamps * n_turbines * n_features * dtype_bytes
    target_bytes = n_timestamps * n_turbines * dtype_bytes
    return int(feature_bytes + target_bytes)
