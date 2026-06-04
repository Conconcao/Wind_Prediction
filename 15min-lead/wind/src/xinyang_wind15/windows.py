"""Window builders for future deep-learning and graph models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .settings import SplitBounds
from .splits import assign_split_labels


def estimate_dense_window_bytes(
    *,
    n_timestamps: int,
    n_turbines: int,
    n_features: int,
    lookback_steps: int,
    horizon_steps: int,
    dtype_bytes: int = 4,
) -> int:
    """Estimate bytes required by dense materialized window tensors."""
    n_windows = max(0, n_timestamps - lookback_steps - horizon_steps + 1)
    x_bytes = n_windows * lookback_steps * n_turbines * n_features * dtype_bytes
    y_bytes = n_windows * n_turbines * dtype_bytes
    return int(x_bytes + y_bytes)


def build_feature_target_arrays(
    scada_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
) -> dict[str, object]:
    turbine_order = sorted(scada_df["turbine_id"].unique())
    timestamps = sorted(scada_df["timestamp"].unique())
    n_timestamps = len(timestamps)
    n_turbines = len(turbine_order)
    n_features = len(feature_columns)

    feature_tensor = np.empty(
        (n_timestamps, n_turbines, n_features),
        dtype=np.float32,
    )
    for feature_idx, feature in enumerate(feature_columns):
        pivot = (
            scada_df.pivot(index="timestamp", columns="turbine_id", values=feature)
            .reindex(index=timestamps, columns=turbine_order)
        )
        feature_tensor[:, :, feature_idx] = pivot.to_numpy(dtype=np.float32)

    target_pivot = (
        scada_df.pivot(index="timestamp", columns="turbine_id", values=target_column)
        .reindex(index=timestamps, columns=turbine_order)
    )
    target_matrix = target_pivot.to_numpy(dtype=np.float32)

    return {
        "feature_tensor": feature_tensor,
        "target_matrix": target_matrix,
        "timestamps": np.asarray(timestamps, dtype="datetime64[ns]"),
        "turbine_order": turbine_order,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
    }


def compute_valid_window_indices(
    feature_tensor: np.ndarray,
    target_matrix: np.ndarray,
    *,
    lookback_steps: int,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_timestamps = int(feature_tensor.shape[0])
    if n_timestamps != int(target_matrix.shape[0]):
        raise ValueError("feature_tensor and target_matrix must share the time axis length.")
    candidate_origins = np.arange(
        lookback_steps - 1,
        n_timestamps - horizon_steps,
        dtype=np.int64,
    )
    if candidate_origins.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    invalid_feature_steps = np.isnan(feature_tensor).any(axis=(1, 2)).astype(np.int64)
    invalid_target_steps = np.isnan(target_matrix).any(axis=1)
    invalid_prefix = np.concatenate(
        [np.zeros(1, dtype=np.int64), invalid_feature_steps.cumsum()],
    )
    window_starts = candidate_origins - lookback_steps + 1
    invalid_counts = (
        invalid_prefix[candidate_origins + 1] - invalid_prefix[window_starts]
    )
    target_indices = candidate_origins + horizon_steps
    valid_mask = (invalid_counts == 0) & (~invalid_target_steps[target_indices])
    return candidate_origins[valid_mask], target_indices[valid_mask]


def build_spatiotemporal_windows(
    scada_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    lookback_steps: int,
    horizon_steps: int,
    split_bounds: SplitBounds,
) -> dict[str, object]:
    arrays = build_feature_target_arrays(
        scada_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    feature_tensor = arrays["feature_tensor"]
    target_matrix = arrays["target_matrix"]
    timestamps = arrays["timestamps"]
    turbine_order = arrays["turbine_order"]
    valid_origins, target_indices = compute_valid_window_indices(
        feature_tensor,
        target_matrix,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
    )

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    origin_times: list[pd.Timestamp] = []
    target_times: list[pd.Timestamp] = []

    for origin_idx, target_idx in zip(valid_origins, target_indices, strict=False):
        start_idx = origin_idx - lookback_steps + 1
        x_window = feature_tensor[start_idx : origin_idx + 1]
        y_step = target_matrix[target_idx]
        x_list.append(x_window)
        y_list.append(y_step)
        origin_times.append(pd.Timestamp(timestamps[origin_idx]))
        target_times.append(pd.Timestamp(timestamps[target_idx]))

    if not x_list:
        missing_share = (
            scada_df.loc[:, list(feature_columns)]
            .isna()
            .mean()
            .sort_values(ascending=False)
        )
        top_missing = missing_share.loc[missing_share > 0].head(10).to_dict()
        raise ValueError(
            "No valid windows were generated because every candidate window contained "
            f"missing values. Top feature missing shares: {top_missing}"
        )

    x = np.stack(x_list, axis=0)
    y = np.stack(y_list, axis=0)
    split_labels = assign_split_labels(pd.Series(target_times), split_bounds).to_numpy()

    return {
        "x": x,
        "y": y,
        "origin_times": np.asarray(origin_times, dtype="datetime64[ns]"),
        "target_times": np.asarray(target_times, dtype="datetime64[ns]"),
        "split": split_labels,
        "turbine_order": turbine_order,
        "feature_columns": list(feature_columns),
        "lookback_steps": lookback_steps,
        "horizon_steps": horizon_steps,
    }


def save_window_bundle(bundle: dict[str, object], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "windows.npz",
        x=bundle["x"],
        y=bundle["y"],
        origin_times=bundle["origin_times"],
        target_times=bundle["target_times"],
        split=bundle["split"],
    )
    metadata = {
        "turbine_order": bundle["turbine_order"],
        "feature_columns": bundle["feature_columns"],
        "lookback_steps": bundle["lookback_steps"],
        "horizon_steps": bundle["horizon_steps"],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
