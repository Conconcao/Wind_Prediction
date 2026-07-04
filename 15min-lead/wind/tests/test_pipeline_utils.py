from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from xinyang_wind15.features import build_supervised_frame, build_timestep_feature_frame
from xinyang_wind15.graph import (
    build_bearing_matrix,
    build_correlation_adjacency,
    build_directional_supports_torch,
    build_graph_wavenet_supports,
)
from xinyang_wind15.loading import (
    _resolve_input_path,
    aggregate_scada_1min_to_15min,
    build_scada_1min_aggregates,
)
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


def make_synthetic_tower_wide() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=12, freq="15min")
    rows = []
    for timestamp in timestamps:
        rows.append(
            {
                "timestamp": timestamp,
                "tower_ws_10m": 5.0,
                "tower_ws_70m": 7.0,
                "tower_ws_125m": 8.0,
                "tower_wd_10m": 260.0,
                "tower_wd_70m": 265.0,
                "tower_wd_125m": 270.0,
                "tower_temperature_10m": 20.0,
                "tower_temperature_125m": 18.0,
                "tower_pressure_10m": 1000.0,
                "tower_pressure_125m": 985.0,
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


def test_aggregate_scada_1min_to_15min_builds_canonical_frame() -> None:
    one_min = pd.DataFrame(
        [
            {
                "turbine_id": "T01",
                "timestamp": pd.Timestamp("2025-01-01 00:00:00"),
                "ws": 1.0,
                "power": 10.0,
                "wd": 350.0,
                "nacelle_angle": 5.0,
            },
            {
                "turbine_id": "T01",
                "timestamp": pd.Timestamp("2025-01-01 00:01:00"),
                "ws": 3.0,
                "power": 12.0,
                "wd": 10.0,
                "nacelle_angle": 355.0,
            },
            {
                "turbine_id": "T01",
                "timestamp": pd.Timestamp("2025-01-01 00:14:00"),
                "ws": 5.0,
                "power": 14.0,
                "wd": 0.0,
                "nacelle_angle": 0.0,
            },
            {
                "turbine_id": "T01",
                "timestamp": pd.Timestamp("2025-01-01 00:15:00"),
                "ws": 7.0,
                "power": 16.0,
                "wd": 90.0,
                "nacelle_angle": 90.0,
            },
        ]
    )
    aggregated = aggregate_scada_1min_to_15min(one_min)
    assert list(aggregated["timestamp"]) == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 00:15:00"),
    ]

    first = aggregated.iloc[0]
    assert first["ws_mean"] == pytest.approx(3.0)
    assert first["ws_max"] == pytest.approx(5.0)
    assert first["ws_min"] == pytest.approx(1.0)
    assert first["cnt_raw"] == pytest.approx(3.0)
    assert first["wd_mean"] == pytest.approx(0.0)
    assert first["nacelle_mean"] == pytest.approx(0.0)


def test_build_timestep_feature_frame_adds_direction_and_yaw_features() -> None:
    scada = make_synthetic_scada()
    frame = build_timestep_feature_frame(scada)
    assert {"wd_sin", "wd_cos", "nacelle_sin", "nacelle_cos"}.issubset(frame.columns)
    assert {"yaw_error_sin", "yaw_error_cos", "yaw_error_abs", "yaw_error_deg"}.issubset(
        frame.columns
    )
    first_row = frame.iloc[0]
    assert first_row["yaw_error_deg"] == pytest.approx(90.0)
    assert first_row["yaw_error_abs"] == pytest.approx(90.0)


def test_build_timestep_feature_frame_adds_derived_core_features() -> None:
    scada = make_synthetic_scada()
    tower_wide = make_synthetic_tower_wide()
    frame = build_timestep_feature_frame(
        scada,
        tower_wide=tower_wide,
        include_derived_core=True,
    )
    first_row = frame.loc[
        (frame["turbine_id"] == "T02") & (frame["timestamp"] == pd.Timestamp("2025-01-01 00:00:00"))
    ].iloc[0]
    expected_alpha_10_125 = math.log(8.0 / 5.0) / math.log(125.0 / 10.0)
    expected_alpha_70_125 = math.log(8.0 / 7.0) / math.log(125.0 / 70.0)
    assert first_row["derived_ti_15m"] == pytest.approx(0.1)
    assert first_row["derived_gust_factor_15m"] == pytest.approx(1.2)
    assert first_row["derived_gust_excess_15m"] == pytest.approx(0.2)
    assert first_row["derived_ws_range_15m"] == pytest.approx(0.4)
    assert first_row["profile_shear_alpha_10m_125m"] == pytest.approx(expected_alpha_10_125)
    assert first_row["profile_shear_alpha_70m_125m"] == pytest.approx(expected_alpha_70_125)
    assert first_row["profile_veer_10m_125m_abs"] == pytest.approx(10.0)
    assert first_row["profile_veer_70m_125m_abs"] == pytest.approx(5.0)
    assert first_row["profile_temperature_delta_125m_10m"] == pytest.approx(-2.0)
    assert first_row["profile_pressure_delta_125m_10m"] == pytest.approx(-15.0)
    assert first_row["hub_tower_ws_125m_delta"] == pytest.approx(-7.0)
    assert first_row["hub_tower_wd_125m_abs"] == pytest.approx(0.0)
    assert "profile_shear_alpha_10m_125m_missing" in frame.columns
    assert first_row["profile_shear_alpha_10m_125m_missing"] == 0


def test_build_timestep_feature_frame_adds_spatial_context_features() -> None:
    scada = make_synthetic_scada()
    turbine_meta = pd.DataFrame(
        [
            {"turbine_id": "T01", "longitude_deg": 120.0, "latitude_deg": 33.0},
            {"turbine_id": "T02", "longitude_deg": 120.01, "latitude_deg": 33.0},
        ]
    )
    frame = build_timestep_feature_frame(
        scada,
        turbine_meta=turbine_meta,
        include_spatial_context=True,
        spatial_direction_sigma_deg=20.0,
        spatial_distance_scale_km=2.0,
    )
    t01_row = frame.loc[
        (frame["turbine_id"] == "T01") & (frame["timestamp"] == pd.Timestamp("2025-01-01 00:00:00"))
    ].iloc[0]
    t02_row = frame.loc[
        (frame["turbine_id"] == "T02") & (frame["timestamp"] == pd.Timestamp("2025-01-01 00:00:00"))
    ].iloc[0]
    assert t02_row["ctx_upwind_count"] == pytest.approx(1.0)
    assert t02_row["ctx_upwind_ws_mean"] == pytest.approx(0.0)
    assert t02_row["ctx_upwind_power_mean"] == pytest.approx(10.0)
    assert t02_row["ctx_upwind_ws_gap"] == pytest.approx(1.0)
    assert t02_row["ctx_upwind_weight_sum"] > 0.0
    assert t02_row["ctx_upwind_nearest_dist_km"] > 0.0
    assert t01_row["ctx_upwind_ws_mean"] == pytest.approx(0.0)
    assert t01_row["ctx_upwind_power_mean"] == pytest.approx(10.0)
    assert t01_row["ctx_upwind_ws_gap"] == pytest.approx(0.0)
    assert t01_row["ctx_upwind_nearest_dist_km"] == pytest.approx(0.0)
    assert t01_row["ctx_upwind_count"] == pytest.approx(0.0)



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


def test_bearing_matrix_and_directional_supports() -> None:
    turbine_meta = pd.DataFrame(
        [
            {"turbine_id": "T01", "longitude_deg": 120.0, "latitude_deg": 33.0},
            {"turbine_id": "T02", "longitude_deg": 120.01, "latitude_deg": 33.0},
        ]
    )
    bearing_matrix = build_bearing_matrix(turbine_meta, ["T01", "T02"])
    assert bearing_matrix.shape == (2, 2)
    assert float(bearing_matrix[0, 1]) == pytest.approx(90.0, abs=5.0)

    supports = build_directional_supports_torch(
        torch.tensor([[270.0, 270.0]], dtype=torch.float32),
        bearing_matrix_deg=torch.tensor(bearing_matrix, dtype=torch.float32),
        base_adjacency=torch.ones((2, 2), dtype=torch.float32),
        sigma_deg=20.0,
        include_transpose=True,
    )
    assert len(supports) == 2
    assert tuple(supports[0].shape) == (1, 2, 2)
    assert float(supports[0][0, 0, 1]) > float(supports[0][0, 1, 0])


def test_resolve_input_path_recovers_from_mojibake_like_names(tmp_path) -> None:
    tower_file = tmp_path / "pre_QC_气象观测数据.xlsx"
    turbine_meta_file = tmp_path / "风机基本信息.csv"
    tower_file.write_bytes(b"placeholder")
    turbine_meta_file.write_text("turbine_id,longitude_deg,latitude_deg\nT01,120.0,33.0\n", encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="Input path not found for tower_met"):
        resolved_tower = _resolve_input_path(
            tmp_path / "pre_QC_姘旇薄瑙傛祴鏁版嵁.xlsx",
            kind="tower_met",
        )
    with pytest.warns(RuntimeWarning, match="Input path not found for turbine_meta"):
        resolved_meta = _resolve_input_path(
            tmp_path / "椋庢満鍩烘湰淇℃伅.csv",
            kind="turbine_meta",
        )

    assert resolved_tower == tower_file
    assert resolved_meta == turbine_meta_file
