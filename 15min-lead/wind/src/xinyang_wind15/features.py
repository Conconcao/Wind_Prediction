"""Feature engineering for xinyang wind-speed experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .settings import SplitBounds
from .splits import assign_split_labels


DEFAULT_LAG_SPEC: dict[str, list[int]] = {
    "ws_mean": [1, 2, 4, 8, 96],
    "power_mean": [1, 4, 96],
    "ws_std": [1, 4],
    "wd_mean": [1, 4],
    "nacelle_mean": [1, 4],
    "cnt_raw": [1],
}


def add_calendar_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    hour_float = ts.dt.hour + ts.dt.minute / 60.0
    hour_angle = 2.0 * np.pi * hour_float / 24.0
    day_angle = 2.0 * np.pi * ts.dt.dayofyear / 366.0
    out["hour_sin"] = np.sin(hour_angle)
    out["hour_cos"] = np.cos(hour_angle)
    out["doy_sin"] = np.sin(day_angle)
    out["doy_cos"] = np.cos(day_angle)
    return out


def add_directional_features(
    df: pd.DataFrame,
    source_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()
    radians = np.deg2rad(out[source_col] % 360.0)
    out[f"{prefix}_sin"] = np.sin(radians)
    out[f"{prefix}_cos"] = np.cos(radians)
    return out


def _add_missing_indicators(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    if not feature_cols:
        return frame
    missing_flags = pd.DataFrame(
        {
            f"{col}_missing": frame[col].isna().astype("int8")
            for col in feature_cols
        },
        index=frame.index,
    )
    return pd.concat([frame, missing_flags], axis=1)


def merge_tower_features(
    frame: pd.DataFrame,
    tower_wide: pd.DataFrame,
    *,
    origin_timestamp_col: str,
) -> pd.DataFrame:
    out = frame.merge(
        tower_wide,
        left_on=origin_timestamp_col,
        right_on="timestamp",
        how="left",
        suffixes=("", "_tower"),
    )
    if "timestamp_tower" in out.columns:
        out = out.drop(columns=["timestamp_tower"])
    tower_feature_cols = [col for col in out.columns if col.startswith("tower_")]
    return _add_missing_indicators(out, tower_feature_cols)


def merge_scada_1min_features(
    frame: pd.DataFrame,
    one_min_agg: pd.DataFrame,
    *,
    origin_timestamp_col: str,
) -> pd.DataFrame:
    out = frame.merge(
        one_min_agg,
        left_on=["turbine_id", origin_timestamp_col],
        right_on=["turbine_id", "timestamp"],
        how="left",
        suffixes=("", "_1min"),
    )
    if "timestamp_1min" in out.columns:
        out = out.drop(columns=["timestamp_1min"])
    one_min_feature_cols = [col for col in out.columns if col.startswith("m1_")]
    return _add_missing_indicators(out, one_min_feature_cols)


def _impute_prefixed_columns(
    frame: pd.DataFrame,
    *,
    prefix: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    value_cols = [
        col
        for col in frame.columns
        if col.startswith(prefix) and not col.endswith("_missing")
    ]
    if not value_cols:
        return frame
    out = frame.copy()
    # Only carry past information forward; backfilling would leak future values.
    if group_col is None:
        out[value_cols] = out[value_cols].ffill()
    else:
        out[value_cols] = out.groupby(group_col, sort=False)[value_cols].ffill()
    return out


def build_timestep_feature_frame(
    scada_df: pd.DataFrame,
    *,
    tower_wide: pd.DataFrame | None = None,
    one_min_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = scada_df.copy()
    out = out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    out = add_calendar_features(out, "timestamp")
    out = add_directional_features(out, "wd_mean", "wd")
    out = add_directional_features(out, "nacelle_mean", "nacelle")
    if tower_wide is not None:
        out = merge_tower_features(out, tower_wide, origin_timestamp_col="timestamp")
        out = _impute_prefixed_columns(out, prefix="tower_", group_col="turbine_id")
    if one_min_agg is not None:
        out = merge_scada_1min_features(out, one_min_agg, origin_timestamp_col="timestamp")
        out = _impute_prefixed_columns(out, prefix="m1_", group_col="turbine_id")
    return out


def build_supervised_frame(
    scada_df: pd.DataFrame,
    *,
    split_bounds: SplitBounds,
    horizon_steps: int = 1,
    lag_spec: Mapping[str, Sequence[int]] | None = None,
    include_current_features: Sequence[str] | None = None,
    tower_wide: pd.DataFrame | None = None,
    one_min_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if lag_spec is None:
        lag_spec = DEFAULT_LAG_SPEC
    if include_current_features is None:
        include_current_features = [
            "ws_mean",
            "ws_max",
            "ws_min",
            "ws_std",
            "power_mean",
            "power_std",
            "nacelle_mean",
            "nacelle_std",
            "wd_mean",
            "wd_std",
            "cnt_raw",
        ]

    out = scada_df.copy()
    out = out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    grouped = out.groupby("turbine_id", sort=False)

    out["origin_timestamp"] = out["timestamp"]
    out["target_timestamp"] = grouped["timestamp"].shift(-horizon_steps)
    out["y_true"] = grouped["ws_mean"].shift(-horizon_steps)

    for col in include_current_features:
        out[f"cur_{col}"] = out[col]

    for col, lags in lag_spec.items():
        for lag in sorted(set(int(x) for x in lags)):
            out[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

    out = add_calendar_features(out, "target_timestamp")
    out = add_directional_features(out, "wd_mean", "cur_wd")
    out = add_directional_features(out, "nacelle_mean", "cur_nacelle")

    if tower_wide is not None:
        out = merge_tower_features(
            out,
            tower_wide,
            origin_timestamp_col="origin_timestamp",
        )
        out = _impute_prefixed_columns(out, prefix="tower_", group_col="turbine_id")
    if one_min_agg is not None:
        out = merge_scada_1min_features(
            out,
            one_min_agg,
            origin_timestamp_col="origin_timestamp",
        )
        out = _impute_prefixed_columns(out, prefix="m1_", group_col="turbine_id")

    out["split"] = assign_split_labels(out["target_timestamp"], split_bounds)
    return out


def feature_columns_for_lightgbm(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "origin_timestamp",
        "target_timestamp",
        "split",
        "y_true",
        "ws_mean",
        "ws_max",
        "ws_min",
        "ws_std",
        "power_mean",
        "power_max",
        "power_min",
        "power_std",
        "nacelle_mean",
        "nacelle_max",
        "nacelle_min",
        "nacelle_std",
        "wd_mean",
        "wd_max",
        "wd_min",
        "wd_std",
        "cnt_raw",
    }
    return [col for col in frame.columns if col not in excluded]
