from __future__ import annotations

import pandas as pd

from xinyang_wind15.features import build_supervised_frame, build_timestep_feature_frame
from xinyang_wind15.graph import build_correlation_adjacency, build_graph_wavenet_supports
from xinyang_wind15.loading import build_scada_1min_aggregates
from xinyang_wind15.settings import SplitBounds
from xinyang_wind15.windows import build_spatiotemporal_windows, estimate_dense_window_bytes


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


def test_build_supervised_frame_assigns_target_and_split() -> None:
    scada = make_synthetic_scada()
    bounds = SplitBounds(
        train_start=pd.Timestamp("2025-01-01 00:15:00"),
        train_end=pd.Timestamp("2025-01-01 01:30:00"),
        val_start=pd.Timestamp("2025-01-01 01:45:00"),
        val_end=pd.Timestamp("2025-01-01 02:00:00"),
        test_start=pd.Timestamp("2025-01-01 02:15:00"),
        test_end=pd.Timestamp("2025-01-01 02:45:00"),
    )
    frame = build_supervised_frame(
        scada,
        split_bounds=bounds,
        horizon_steps=1,
    )
    t01 = frame.loc[frame["turbine_id"] == "T01"].reset_index(drop=True)
    assert t01.loc[0, "y_true"] == 1.0
    assert t01.loc[0, "split"] == "train"
    assert t01.loc[8, "split"] == "test"


def test_build_spatiotemporal_windows_shapes() -> None:
    scada = make_synthetic_scada()
    bounds = SplitBounds(
        train_start=pd.Timestamp("2025-01-01 00:45:00"),
        train_end=pd.Timestamp("2025-01-01 01:30:00"),
        val_start=pd.Timestamp("2025-01-01 01:45:00"),
        val_end=pd.Timestamp("2025-01-01 02:00:00"),
        test_start=pd.Timestamp("2025-01-01 02:15:00"),
        test_end=pd.Timestamp("2025-01-01 02:45:00"),
    )
    bundle = build_spatiotemporal_windows(
        scada,
        feature_columns=["ws_mean", "power_mean"],
        target_column="ws_mean",
        lookback_steps=3,
        horizon_steps=1,
        split_bounds=bounds,
    )
    assert bundle["x"].shape[1:] == (3, 2, 2)
    assert bundle["y"].shape[1] == 2


def test_build_scada_1min_aggregates_and_merge() -> None:
    rows = []
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=20, freq="1min")
    for idx, timestamp in enumerate(timestamps):
        rows.append(
            {
                "turbine_id": "T01",
                "timestamp": timestamp,
                "ws": float(idx),
                "power": float(idx + 10),
                "nacelle_angle": float(100 + idx),
                "wd": 270.0,
                "longitude_deg": 120.0,
                "latitude_deg": 33.0,
            }
        )
    one_min = pd.DataFrame(rows)
    origin_timestamps = [
        pd.Timestamp("2025-01-01 00:14:00"),
        pd.Timestamp("2025-01-01 00:19:00"),
    ]
    agg = build_scada_1min_aggregates(
        one_min,
        origin_timestamps=origin_timestamps,
        windows=(15,),
    )
    assert set(agg["timestamp"]) == set(origin_timestamps)

    row = agg.loc[agg["timestamp"] == pd.Timestamp("2025-01-01 00:14:00")].iloc[0]
    assert row["m1_ws_15m_last"] == 14.0
    assert row["m1_ws_15m_min"] == 0.0
    assert row["m1_ws_15m_max"] == 14.0
    assert row["m1_ws_15m_ramp"] == 14.0

    scada = make_synthetic_scada().loc[lambda df: df["turbine_id"] == "T01"].copy()
    scada = scada.loc[scada["timestamp"].isin(origin_timestamps)].copy()
    merged = build_timestep_feature_frame(scada, one_min_agg=agg)
    assert "m1_ws_15m_mean" in merged.columns
    assert merged["m1_ws_15m_mean"].notna().all()


def test_estimate_dense_window_bytes() -> None:
    estimated = estimate_dense_window_bytes(
        n_timestamps=12,
        n_turbines=2,
        n_features=3,
        lookback_steps=4,
        horizon_steps=1,
    )
    expected_windows = 12 - 4 - 1 + 1
    expected = (expected_windows * 4 * 2 * 3 * 4) + (expected_windows * 2 * 4)
    assert estimated == expected


def test_correlation_adjacency_and_support_deduplication() -> None:
    scada = make_synthetic_scada()
    adjacency = build_correlation_adjacency(
        scada,
        ["T01", "T02"],
        value_column="ws_mean",
        min_periods=3,
    )
    assert adjacency.shape == (2, 2)
    assert float(adjacency[0, 0]) == 1.0
    assert float(adjacency[1, 1]) == 1.0

    supports = build_graph_wavenet_supports(adjacency)
    assert len(supports) == 1
