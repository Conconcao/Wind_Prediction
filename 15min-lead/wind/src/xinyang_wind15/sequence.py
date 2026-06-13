"""Shared sequence-model utilities for window-based deep learning baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
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
        target_mask: np.ndarray | None = None,
        stats: StandardizationStats | None = None,
    ) -> None:
        self.feature_tensor = feature_tensor
        self.target_matrix = target_matrix
        self.origin_indices = np.asarray(origin_indices, dtype=np.int64)
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)
        self.target_mask = target_mask
        self.stats = stats

    def __len__(self) -> int:
        return int(len(self.origin_indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if self.target_mask is None:
            y_mask = np.ones_like(y, dtype=bool)
        else:
            y_mask = np.array(
                self.target_mask[target_idx],
                dtype=bool,
                copy=True,
            )
        y_filled = np.where(y_mask, y, 0.0).astype(np.float32, copy=False)
        if self.stats is not None:
            x = (
                (x - self.stats.x_mean[0]) / self.stats.x_std[0]
            ).astype(np.float32, copy=False)
            y_filled = np.where(y_mask, y, self.stats.y_mean[0]).astype(np.float32, copy=False)
            y_filled = (
                (y_filled - self.stats.y_mean[0]) / self.stats.y_std[0]
            ).astype(np.float32, copy=False)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_filled),
            torch.from_numpy(y_mask),
        )


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
    target_mask: np.ndarray | None = None,
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
    count_y = np.zeros((1, n_turbines), dtype=np.float64)

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
        if target_mask is None:
            y_mask_chunk = np.ones_like(y_chunk, dtype=bool)
        else:
            y_mask_chunk = np.asarray(
                target_mask[chunk_origins + horizon_steps],
                dtype=bool,
            )
        y_chunk64 = np.where(y_mask_chunk, y_chunk, 0.0).astype(np.float64, copy=False)
        sum_x += x_chunk64.sum(axis=(0, 1, 2), keepdims=True)
        sum_x2 += np.square(x_chunk64).sum(axis=(0, 1, 2), keepdims=True)
        sum_y += y_chunk64.sum(axis=0, keepdims=True)
        sum_y2 += np.square(y_chunk64).sum(axis=0, keepdims=True)
        count_x += int(np.prod(x_chunk.shape[:3]))
        count_y += y_mask_chunk.sum(axis=0, keepdims=True, dtype=np.float64)

    x_mean = sum_x / count_x
    x_var = np.maximum((sum_x2 / count_x) - np.square(x_mean), 0.0)
    x_std = np.sqrt(x_var)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    y_mean = np.divide(sum_y, count_y, out=np.zeros_like(sum_y), where=count_y > 0)
    y_var = np.maximum(
        np.divide(sum_y2, count_y, out=np.zeros_like(sum_y2), where=count_y > 0)
        - np.square(y_mean),
        0.0,
    )
    y_std = np.sqrt(y_var)
    y_std = np.where((count_y < 2) | (y_std < 1e-6), 1.0, y_std)

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
    supports_builder: Callable[[torch.Tensor], list[torch.Tensor] | None] | None = None,
    target_timestamps: np.ndarray | pd.Series | list[object] | None = None,
    origin_timestamps: np.ndarray | pd.Series | list[object] | None = None,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame] | tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model.eval()
    all_preds = []
    all_truth = []
    all_masks = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x_batch, y_batch, y_mask_batch = batch
            else:
                x_batch, y_batch = batch
                y_mask_batch = torch.ones_like(y_batch, dtype=torch.bool)
            x_batch = x_batch.to(device)
            if supports_builder is None:
                pred = model(x_batch).cpu().numpy()
            else:
                extra_supports = supports_builder(x_batch)
                pred = model(x_batch, extra_supports=extra_supports).cpu().numpy()
            all_preds.append(pred)
            all_truth.append(y_batch.numpy())
            all_masks.append(y_mask_batch.numpy().astype(bool, copy=False))
    y_pred_scaled = np.concatenate(all_preds, axis=0)
    y_true_scaled = np.concatenate(all_truth, axis=0)
    y_mask = np.concatenate(all_masks, axis=0).astype(bool, copy=False)
    y_pred = inverse_targets(y_pred_scaled, stats)
    y_true = inverse_targets(y_true_scaled, stats)

    per_turbine_rows = []
    for idx, turbine_id in enumerate(turbine_order):
        turbine_mask = y_mask[:, idx]
        metrics = regression_summary(
            pd.Series(y_true[turbine_mask, idx]),
            pd.Series(y_pred[turbine_mask, idx]),
        )
        per_turbine_rows.append({"turbine_id": turbine_id, **metrics})
    per_turbine_df = pd.DataFrame(per_turbine_rows)
    observed_mask = y_mask.reshape(-1)
    overall = regression_summary(
        pd.Series(y_true.reshape(-1)[observed_mask]),
        pd.Series(y_pred.reshape(-1)[observed_mask]),
    )
    overall["mae_macro"] = float(per_turbine_df["mae"].mean(skipna=True))
    overall["rmse_macro"] = float(per_turbine_df["rmse"].mean(skipna=True))
    overall["r2_macro"] = float(per_turbine_df["r2"].mean(skipna=True))
    overall["split"] = split_name
    if return_predictions:
        prediction_df = build_window_prediction_frame(
            y_true,
            y_pred,
            y_mask,
            turbine_order=turbine_order,
            split_name=split_name,
            target_timestamps=target_timestamps,
            origin_timestamps=origin_timestamps,
        )
        return overall, per_turbine_df, prediction_df
    return overall, per_turbine_df


def build_window_prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_mask: np.ndarray,
    *,
    turbine_order: list[str],
    split_name: str,
    target_timestamps: np.ndarray | pd.Series | list[object] | None,
    origin_timestamps: np.ndarray | pd.Series | list[object] | None = None,
) -> pd.DataFrame:
    if target_timestamps is None:
        raise ValueError("target_timestamps are required when exporting prediction frames.")

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_mask = np.asarray(y_mask, dtype=bool)
    target_ts = pd.to_datetime(np.asarray(target_timestamps))
    if len(target_ts) != y_true.shape[0]:
        raise ValueError(
            "target_timestamps length must match the number of prediction rows. "
            f"Got timestamps={len(target_ts)}, predictions={y_true.shape[0]}."
        )

    origin_ts = None
    if origin_timestamps is not None:
        origin_ts = pd.to_datetime(np.asarray(origin_timestamps))
        if len(origin_ts) != y_true.shape[0]:
            raise ValueError(
                "origin_timestamps length must match the number of prediction rows. "
                f"Got timestamps={len(origin_ts)}, predictions={y_true.shape[0]}."
            )

    rows: list[pd.DataFrame] = []
    for turbine_idx, turbine_id in enumerate(turbine_order):
        turbine_mask = y_mask[:, turbine_idx]
        if not np.any(turbine_mask):
            continue
        block = pd.DataFrame(
            {
                "turbine_id": turbine_id,
                "target_timestamp": target_ts[turbine_mask],
                "split": split_name,
                "y_true": y_true[turbine_mask, turbine_idx],
                "y_pred": y_pred[turbine_mask, turbine_idx],
            }
        )
        if origin_ts is not None:
            block.insert(1, "origin_timestamp", origin_ts[turbine_mask])
        rows.append(block)

    if not rows:
        columns = ["turbine_id", "target_timestamp", "split", "y_true", "y_pred"]
        if origin_ts is not None:
            columns.insert(1, "origin_timestamp")
        return pd.DataFrame(columns=columns)

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["turbine_id", "target_timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )


def masked_mean_loss(loss_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(loss_tensor.device, dtype=loss_tensor.dtype)
    valid_count = mask.sum()
    if valid_count.item() <= 0:
        raise ValueError("Masked loss received a batch with no valid targets.")
    return (loss_tensor * mask).sum() / valid_count


def save_checkpoint(model: nn.Module, output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
