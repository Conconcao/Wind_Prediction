"""Shared sequence-model utilities for window-based deep learning baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import regression_summary


class WindowDataset(Dataset):
    """Simple in-memory dataset for spatio-temporal windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class WindowStoreDataset(Dataset):
    """Map-style lazy dataset backed by a time-major feature store on disk."""

    def __init__(
        self,
        feature_tensor: np.ndarray,
        target_matrix: np.ndarray,
        origin_indices: np.ndarray,
        *,
        lookback_steps: int,
        horizon_steps: int,
        stats: StandardizationStats | None = None,
    ) -> None:
        self.feature_tensor = feature_tensor
        self.target_matrix = target_matrix
        self.origin_indices = np.asarray(origin_indices, dtype=np.int64)
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)
        self.stats = stats

    def __len__(self) -> int:
        return int(len(self.origin_indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin_idx = int(self.origin_indices[idx])
        start_idx = origin_idx - self.lookback_steps + 1
        target_idx = origin_idx + self.horizon_steps
        x = np.array(
            self.feature_tensor[start_idx : origin_idx + 1],
            dtype=np.float32,
            copy=True,
        )
        y = np.array(
            self.target_matrix[target_idx],
            dtype=np.float32,
            copy=True,
        )
        if self.stats is not None:
            x = (
                (x - self.stats.x_mean[0]) / self.stats.x_std[0]
            ).astype(np.float32, copy=False)
            y = (
                (y - self.stats.y_mean[0]) / self.stats.y_std[0]
            ).astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass(frozen=True)
class StandardizationStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def compute_standardization_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> StandardizationStats:
    x_mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    x_std = x_train.std(axis=(0, 1, 2), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    return StandardizationStats(
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )


def standardize_arrays(
    x: np.ndarray,
    y: np.ndarray | None,
    stats: StandardizationStats,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_scaled = (x - stats.x_mean) / stats.x_std
    y_scaled = None
    if y is not None:
        y_scaled = (y - stats.y_mean) / stats.y_std
    return x_scaled.astype(np.float32), None if y_scaled is None else y_scaled.astype(np.float32)


def inverse_targets(
    y_scaled: np.ndarray,
    stats: StandardizationStats,
) -> np.ndarray:
    return (y_scaled * stats.y_std) + stats.y_mean


def make_loader(
    x: np.ndarray | Dataset,
    y: np.ndarray | None = None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset: Dataset
    if isinstance(x, Dataset):
        dataset = x
    else:
        if y is None:
            raise ValueError("y is required when make_loader is called with array inputs.")
        dataset = WindowDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def compute_standardization_stats_from_store(
    feature_tensor: np.ndarray,
    target_matrix: np.ndarray,
    origin_indices: np.ndarray,
    *,
    lookback_steps: int,
    horizon_steps: int,
    chunk_size: int = 32,
) -> StandardizationStats:
    origin_indices = np.asarray(origin_indices, dtype=np.int64)
    if origin_indices.size == 0:
        raise ValueError("Cannot compute standardization stats from an empty training split.")

    n_features = int(feature_tensor.shape[-1])
    n_turbines = int(target_matrix.shape[-1])
    sum_x = np.zeros((1, 1, 1, n_features), dtype=np.float64)
    sum_x2 = np.zeros((1, 1, 1, n_features), dtype=np.float64)
    sum_y = np.zeros((1, n_turbines), dtype=np.float64)
    sum_y2 = np.zeros((1, n_turbines), dtype=np.float64)
    count_x = 0
    count_y = 0

    for start in range(0, len(origin_indices), chunk_size):
        chunk_origins = origin_indices[start : start + chunk_size]
        x_chunk = np.stack(
            [
                np.asarray(
                    feature_tensor[origin_idx - lookback_steps + 1 : origin_idx + 1],
                    dtype=np.float32,
                )
                for origin_idx in chunk_origins
            ],
            axis=0,
        )
        y_chunk = np.asarray(
            target_matrix[chunk_origins + horizon_steps],
            dtype=np.float32,
        )
        x_chunk64 = x_chunk.astype(np.float64, copy=False)
        y_chunk64 = y_chunk.astype(np.float64, copy=False)
        sum_x += x_chunk64.sum(axis=(0, 1, 2), keepdims=True)
        sum_x2 += np.square(x_chunk64).sum(axis=(0, 1, 2), keepdims=True)
        sum_y += y_chunk64.sum(axis=0, keepdims=True)
        sum_y2 += np.square(y_chunk64).sum(axis=0, keepdims=True)
        count_x += int(np.prod(x_chunk.shape[:3]))
        count_y += int(y_chunk.shape[0])

    x_mean = sum_x / count_x
    x_var = np.maximum((sum_x2 / count_x) - np.square(x_mean), 0.0)
    x_std = np.sqrt(x_var)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    y_mean = sum_y / count_y
    y_var = np.maximum((sum_y2 / count_y) - np.square(y_mean), 0.0)
    y_std = np.sqrt(y_var)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)

    return StandardizationStats(
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )


def evaluate_window_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    stats: StandardizationStats,
    turbine_order: list[str],
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    all_preds = []
    all_truth = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            all_preds.append(pred)
            all_truth.append(y_batch.numpy())
    y_pred_scaled = np.concatenate(all_preds, axis=0)
    y_true_scaled = np.concatenate(all_truth, axis=0)
    y_pred = inverse_targets(y_pred_scaled, stats)
    y_true = inverse_targets(y_true_scaled, stats)

    per_turbine_rows = []
    for idx, turbine_id in enumerate(turbine_order):
        metrics = regression_summary(
            pd.Series(y_true[:, idx]),
            pd.Series(y_pred[:, idx]),
        )
        per_turbine_rows.append({"turbine_id": turbine_id, **metrics})
    per_turbine_df = pd.DataFrame(per_turbine_rows)
    overall = regression_summary(
        pd.Series(y_true.reshape(-1)),
        pd.Series(y_pred.reshape(-1)),
    )
    overall["mae_macro"] = float(per_turbine_df["mae"].mean())
    overall["rmse_macro"] = float(per_turbine_df["rmse"].mean())
    overall["r2_macro"] = float(per_turbine_df["r2"].mean())
    overall["split"] = split_name
    return overall, per_turbine_df


def save_checkpoint(model: nn.Module, output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
