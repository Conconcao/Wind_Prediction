from __future__ import annotations

import pandas as pd

from xinyang_wind15.raw_one_min import (
    build_ratio_split_bounds,
    build_raw_one_min_feature_frame,
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
