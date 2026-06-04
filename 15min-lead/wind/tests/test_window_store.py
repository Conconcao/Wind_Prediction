from __future__ import annotations

import pandas as pd

from xinyang_wind15.sequence import (
    WindowStoreDataset,
    compute_standardization_stats_from_store,
)
from xinyang_wind15.settings import SplitBounds
from xinyang_wind15.window_store import load_window_store, write_window_store


def make_synthetic_scada() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=12, freq="15min")
    rows = []
    for turbine_id, offset in [("T01", 0.0), ("T02", 1.0)]:
        for idx, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "turbine_id": turbine_id,
                    "timestamp": timestamp,
                    "ws_mean": idx + offset,
                    "ws_max": idx + offset + 0.2,
                    "ws_min": idx + offset - 0.2,
                    "ws_std": 0.1,
                    "power_mean": 10.0 + idx,
                    "power_max": 10.5 + idx,
                    "power_min": 9.5 + idx,
                    "power_std": 0.2,
                    "nacelle_mean": 180.0,
                    "nacelle_max": 182.0,
                    "nacelle_min": 178.0,
                    "nacelle_std": 0.5,
                    "wd_mean": 270.0,
                    "wd_max": 275.0,
                    "wd_min": 265.0,
                    "wd_std": 1.0,
                    "cnt_raw": 15,
                }
            )
    return pd.DataFrame(rows)


def test_window_store_round_trip_and_dataset(tmp_path) -> None:
    scada = make_synthetic_scada()
    bounds = SplitBounds(
        train_start=pd.Timestamp("2025-01-01 00:45:00"),
        train_end=pd.Timestamp("2025-01-01 01:30:00"),
        val_start=pd.Timestamp("2025-01-01 01:45:00"),
        val_end=pd.Timestamp("2025-01-01 02:00:00"),
        test_start=pd.Timestamp("2025-01-01 02:15:00"),
        test_end=pd.Timestamp("2025-01-01 02:45:00"),
    )
    metadata = write_window_store(
        scada,
        feature_columns=["ws_mean", "power_mean"],
        target_column="ws_mean",
        lookback_steps=3,
        horizon_steps=1,
        split_bounds=bounds,
        output_dir=tmp_path,
    )
    assert metadata["feature_tensor_shape"] == [12, 2, 2]
    assert metadata["n_valid_windows"] == 9

    store = load_window_store(tmp_path, mmap_mode="r")
    train_origins = store["origin_indices"][store["split_labels"] == "train"]
    stats = compute_standardization_stats_from_store(
        store["feature_tensor"],
        store["target_matrix"],
        train_origins,
        lookback_steps=3,
        horizon_steps=1,
        chunk_size=2,
    )
    dataset = WindowStoreDataset(
        store["feature_tensor"],
        store["target_matrix"],
        train_origins,
        lookback_steps=3,
        horizon_steps=1,
        stats=stats,
    )
    x, y = dataset[0]
    assert tuple(x.shape) == (3, 2, 2)
    assert tuple(y.shape) == (2,)
