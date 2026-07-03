from __future__ import annotations

import pandas as pd

from xinyang_wind15.loading import load_turbine_metadata
from xinyang_wind15.raw_one_min import (
    build_ratio_split_bounds,
    build_raw_one_min_feature_frame,
    merge_direction_15min_snapshots,
    resolve_feature_columns,
)


def test_build_ratio_split_bounds_uses_chronological_712() -> None:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=10, freq="min")
    bounds = build_ratio_split_bounds(
        timestamps,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    assert bounds.train_start == pd.Timestamp("2025-01-01 00:00:00")
    assert bounds.train_end == pd.Timestamp("2025-01-01 00:06:00")
    assert bounds.val_start == pd.Timestamp("2025-01-01 00:07:00")
    assert bounds.val_end == pd.Timestamp("2025-01-01 00:07:00")
    assert bounds.test_start == pd.Timestamp("2025-01-01 00:08:00")
    assert bounds.test_end == pd.Timestamp("2025-01-01 00:09:00")


def test_build_raw_one_min_feature_frame_keeps_ws_only_by_default() -> None:
    frame = pd.DataFrame(
        {
            "turbine_id": ["F01", "F01"],
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "ws": [5.0, 5.5],
            "wd": [None, None],
        }
    )
    out = build_raw_one_min_feature_frame(frame)
    assert list(out.columns) == ["turbine_id", "timestamp", "ws"]


def test_merge_direction_15min_snapshots_uses_latest_snapshot_per_turbine() -> None:
    scada = pd.DataFrame(
        {
            "turbine_id": ["F01", "F01", "F01", "F02"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:15:00",
                    "2025-01-01 00:16:00",
                    "2025-01-01 00:29:00",
                    "2025-01-01 00:16:00",
                ]
            ),
            "ws": [5.0, 5.1, 5.2, 6.0],
        }
    )
    direction = pd.DataFrame(
        {
            "turbine_id": ["F01", "F01", "F02"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:15:00",
                    "2025-01-01 00:30:00",
                    "2025-01-01 00:15:00",
                ]
            ),
            "wd_mean": [200.0, 210.0, 180.0],
            "wd_std": [10.0, 11.0, 8.0],
            "nacelle_mean": [198.0, 212.0, 179.0],
        }
    )

    merged = merge_direction_15min_snapshots(scada, direction)

    assert float(merged.loc[0, "wd_mean"]) == 200.0
    assert float(merged.loc[1, "wd_mean"]) == 200.0
    assert float(merged.loc[2, "wd_mean"]) == 200.0
    assert float(merged.loc[3, "wd_mean"]) == 180.0


def test_build_raw_one_min_feature_frame_derives_directional_snapshot_features() -> None:
    frame = pd.DataFrame(
        {
            "turbine_id": ["F01"],
            "timestamp": pd.to_datetime(["2025-01-01 00:15:00"]),
            "ws": [5.0],
            "wd_mean": [200.0],
            "wd_std": [12.0],
            "nacelle_mean": [190.0],
            "nacelle_std": [3.0],
        }
    )

    out = build_raw_one_min_feature_frame(frame, include_direction_if_available=True)

    expected_columns = {
        "ws",
        "wd_mean",
        "wd_std",
        "wd_sin",
        "wd_cos",
        "nacelle_mean",
        "nacelle_std",
        "nacelle_sin",
        "nacelle_cos",
        "yaw_error_deg",
        "yaw_error_sin",
        "yaw_error_cos",
        "yaw_error_abs",
    }
    assert expected_columns.issubset(set(out.columns))
    assert abs(float(out.loc[0, "yaw_error_deg"]) - 10.0) < 1e-6


def test_resolve_feature_columns_rejects_missing_features() -> None:
    frame = pd.DataFrame(
        {
            "turbine_id": ["F01"],
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "ws": [5.0],
        }
    )
    try:
        resolve_feature_columns(frame, requested=["ws", "wd_sin"])
    except ValueError as exc:
        assert "wd_sin" in str(exc)
    else:
        raise AssertionError("Expected resolve_feature_columns to fail on missing raw features.")


def test_load_turbine_metadata_filters_summary_by_site(tmp_path) -> None:
    meta_path = tmp_path / "风机基本信息汇总.csv"
    pd.DataFrame(
        {
            "site": ["huaian", "xinyang"],
            "turbine_id": ["F01", "S01"],
            "longitude_deg": [118.1, 114.2],
            "latitude_deg": [33.6, 32.1],
        }
    ).to_csv(meta_path, index=False, encoding="utf-8-sig")

    out = load_turbine_metadata(meta_path, site="huaian")

    assert list(out["turbine_id"]) == ["F01"]
    assert list(out["site"]) == ["huaian"]
