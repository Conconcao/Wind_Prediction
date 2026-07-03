from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from xinyang_wind15.power_curve_bridge import (
    HUAIAN_G5_CURVE_FEATURE_COLUMNS,
    HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS,
    HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_FEATURE_COLUMNS,
    filter_predictions_to_reference_keys,
    load_ws_prediction_frames,
    merge_predictions_with_reference_frame,
)


def test_load_ws_prediction_frames_combines_files(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "turbine_id": ["T01"],
            "target_timestamp": ["2025-01-01 00:15:00"],
            "y_pred": [5.0],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "turbine_id": ["T02"],
            "target_timestamp": ["2025-01-01 00:15:00"],
            "y_pred": [6.0],
        }
    ).to_csv(second, index=False)

    combined = load_ws_prediction_frames([first, second])

    assert len(combined) == 2
    assert combined["turbine_id"].tolist() == ["T01", "T02"]


def test_load_ws_prediction_frames_rejects_duplicate_keys(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    payload = {
        "turbine_id": ["T01"],
        "target_timestamp": ["2025-01-01 00:15:00"],
        "y_pred": [5.0],
    }
    pd.DataFrame(payload).to_csv(first, index=False)
    pd.DataFrame(payload).to_csv(second, index=False)

    with pytest.raises(ValueError, match="Duplicated wind-speed prediction rows"):
        load_ws_prediction_frames([first, second])


def test_merge_predictions_with_reference_frame_keeps_extra_columns() -> None:
    ws_predictions = pd.DataFrame(
        {
            "turbine_id": ["T01", "T02"],
            "target_timestamp": pd.to_datetime(["2025-01-01 00:15:00", "2025-01-01 00:30:00"]),
            "ws_pred": [5.2, 6.1],
        }
    )
    reference_frames = {
        "val": pd.DataFrame(
            {
                "turbine_id": ["T01"],
                "time": pd.to_datetime(["2025-01-01 00:15:00"]),
                "ws": [5.0],
                "power": [1000.0],
                "power_norm": [0.4],
                "wd_sin": [0.5],
            }
        ),
        "test": pd.DataFrame(
            {
                "turbine_id": ["T02"],
                "time": pd.to_datetime(["2025-01-01 00:30:00"]),
                "ws": [6.0],
                "power": [1200.0],
                "power_norm": [0.48],
                "wd_sin": [0.6],
            }
        ),
    }

    joined = merge_predictions_with_reference_frame(
        ws_predictions,
        reference_frames,
        extra_columns=["wd_sin"],
    )

    assert list(joined["actual_split"]) == ["val", "test"]
    assert list(joined["power_true_kw"]) == [1000.0, 1200.0]
    assert list(joined["power_true_norm"]) == [0.4, 0.48]
    assert list(joined["wd_sin"]) == [0.5, 0.6]


def test_huaian_g5_curve_feature_set_is_minimal_directional() -> None:
    assert len(HUAIAN_G5_CURVE_FEATURE_COLUMNS) == 17
    assert "wd_sin" in HUAIAN_G5_CURVE_FEATURE_COLUMNS
    assert "yaw_err_deg" in HUAIAN_G5_CURVE_FEATURE_COLUMNS
    assert not any(name.startswith("hist_") for name in HUAIAN_G5_CURVE_FEATURE_COLUMNS)


def test_huaian_latest_history_no_direction_feature_set_matches_intent() -> None:
    assert "ws" in HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS
    assert "turbine_id" in HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS
    assert any(name.startswith("hist_") for name in HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS)
    assert not any(
        key in name
        for name in HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS
        for key in ["wd_", "yaw", "wake_"]
    )


def test_huaian_latest_history_past_direction_only_feature_set_matches_intent() -> None:
    assert "hist_yaw_err_abs_lag_1" in HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_FEATURE_COLUMNS
    assert "hist_wd_sector_mode_ratio_8" in HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_FEATURE_COLUMNS
    assert not any(
        name in HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_FEATURE_COLUMNS
        for name in ["wd_sin", "wd_cos", "yaw_sin", "yaw_cos", "wd_sector_12"]
    )


def test_filter_predictions_to_reference_keys_keeps_overlap_only() -> None:
    ws_predictions = pd.DataFrame(
        {
            "turbine_id": ["T01", "T01", "T01"],
            "target_timestamp": pd.to_datetime(
                ["2025-01-01 00:15:00", "2025-01-01 00:30:00", "2025-01-01 00:45:00"]
            ),
            "ws_pred": [5.0, 5.1, 5.2],
        }
    )
    reference_frames = {
        "val": pd.DataFrame(
            {
                "turbine_id": ["T01"],
                "time": pd.to_datetime(["2025-01-01 00:30:00"]),
                "power_norm": [0.4],
            }
        ),
        "test": pd.DataFrame(
            {
                "turbine_id": ["T01"],
                "time": pd.to_datetime(["2025-01-01 00:45:00"]),
                "power_norm": [0.5],
            }
        ),
    }

    filtered, meta = filter_predictions_to_reference_keys(ws_predictions, reference_frames)

    assert filtered["target_timestamp"].tolist() == list(pd.to_datetime(["2025-01-01 00:30:00", "2025-01-01 00:45:00"]))
    assert meta["prediction_rows_input"] == 3
    assert meta["prediction_rows_retained"] == 2
    assert meta["prediction_rows_dropped"] == 1
    assert meta["missing_reference_rows"] == 0


def test_filter_predictions_to_reference_keys_rejects_missing_reference_rows() -> None:
    ws_predictions = pd.DataFrame(
        {
            "turbine_id": ["T01"],
            "target_timestamp": pd.to_datetime(["2025-01-01 00:30:00"]),
            "ws_pred": [5.1],
        }
    )
    reference_frames = {
        "val": pd.DataFrame(
            {
                "turbine_id": ["T01", "T01"],
                "time": pd.to_datetime(["2025-01-01 00:30:00", "2025-01-01 00:45:00"]),
                "power_norm": [0.4, 0.5],
            }
        )
    }

    with pytest.raises(ValueError, match="do not fully cover"):
        filter_predictions_to_reference_keys(ws_predictions, reference_frames)
