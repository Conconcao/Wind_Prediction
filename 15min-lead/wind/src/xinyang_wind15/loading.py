"""Data loading and normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

from .schema import (
    RAW_SCADA_15MIN_TO_CANONICAL,
    RAW_SCADA_1MIN_TO_CANONICAL,
    RAW_TOWER_TO_CANONICAL,
    TOWER_WIDE_VARIABLES,
)


_FALLBACK_PATTERNS: dict[str, tuple[str, ...]] = {
    "scada_15min": ("ALL_TURBINES_15min_*.parquet",),
    "scada_1min": ("ALL_TURBINES_1min_*.parquet",),
    "tower_met": ("pre_QC*.xlsx", "*气象*观测*.xlsx"),
    "turbine_meta": ("*风机基本信息*.csv",),
}


def _resolve_input_path(path: str | Path, *, kind: str) -> Path:
    resolved = Path(path)
    if resolved.exists():
        return resolved

    parent = resolved.parent
    if not parent.exists():
        raise FileNotFoundError(f"Missing input path and parent directory: {resolved}")

    for pattern in _FALLBACK_PATTERNS.get(kind, ()):
        candidates = sorted(candidate for candidate in parent.glob(pattern) if candidate.is_file())
        if candidates:
            replacement = candidates[0]
            warnings.warn(
                f"Input path not found for {kind}: {resolved}. "
                f"Using fallback file: {replacement}",
                RuntimeWarning,
                stacklevel=2,
            )
            return replacement

    available = ", ".join(sorted(candidate.name for candidate in parent.iterdir() if candidate.is_file()))
    raise FileNotFoundError(
        f"Input path not found for {kind}: {resolved}. "
        f"Available files under {parent}: {available}"
    )


def _filter_time_tail(
    df: pd.DataFrame,
    timestamp_col: str,
    tail_timestamps: int | None,
) -> pd.DataFrame:
    if tail_timestamps is None:
        return df
    keep_times = sorted(df[timestamp_col].unique())[-tail_timestamps:]
    return df.loc[df[timestamp_col].isin(keep_times)].copy()


def _filter_turbines(
    df: pd.DataFrame,
    turbine_col: str,
    max_turbines: int | None,
) -> pd.DataFrame:
    if max_turbines is None:
        return df
    keep_turbines = sorted(df[turbine_col].unique())[:max_turbines]
    return df.loc[df[turbine_col].isin(keep_turbines)].copy()


def _filter_specific_turbines(
    df: pd.DataFrame,
    turbine_col: str,
    turbine_ids: Sequence[str] | None,
) -> pd.DataFrame:
    if not turbine_ids:
        return df
    keep_turbines = [str(turbine_id) for turbine_id in turbine_ids]
    return df.loc[df[turbine_col].isin(keep_turbines)].copy()


def load_scada_15min(
    path: str | Path,
    *,
    max_turbines: int | None = None,
    turbine_ids: Sequence[str] | None = None,
    tail_timestamps: int | None = None,
) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="scada_15min")
    df = pd.read_parquet(resolved_path)
    df = df.rename(columns=RAW_SCADA_15MIN_TO_CANONICAL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _filter_specific_turbines(df, "turbine_id", turbine_ids)
    df = _filter_turbines(df, "turbine_id", max_turbines)
    df = _filter_time_tail(df, "timestamp", tail_timestamps)
    df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    return df


def load_scada_1min(
    path: str | Path,
    *,
    max_turbines: int | None = None,
    turbine_ids: Sequence[str] | None = None,
    tail_timestamps: int | None = None,
) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="scada_1min")
    df = pd.read_parquet(resolved_path)
    df = df.rename(columns=RAW_SCADA_1MIN_TO_CANONICAL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _filter_specific_turbines(df, "turbine_id", turbine_ids)
    df = _filter_turbines(df, "turbine_id", max_turbines)
    df = _filter_time_tail(df, "timestamp", tail_timestamps)
    df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    return df


def build_scada_1min_aggregates(
    scada_1min: pd.DataFrame,
    *,
    origin_timestamps: Sequence[pd.Timestamp] | None = None,
    windows: Sequence[int] = (15, 30, 60),
) -> pd.DataFrame:
    """Aggregate 1-minute SCADA into per-turbine features aligned to 15-minute origins."""
    df = scada_1min.copy()
    if df.empty:
        return pd.DataFrame(columns=["turbine_id", "timestamp"])

    max_window = max(int(window) for window in windows)
    if origin_timestamps is not None and len(origin_timestamps) > 0:
        origin_index = pd.DatetimeIndex(pd.Index(origin_timestamps).unique()).sort_values()
        min_timestamp = origin_index.min() - pd.Timedelta(minutes=max_window - 1)
        max_timestamp = origin_index.max()
        df = df.loc[
            (df["timestamp"] >= min_timestamp) & (df["timestamp"] <= max_timestamp)
        ].copy()
    else:
        origin_index = None

    wd_radians = np.deg2rad(df["wd"] % 360.0)
    df["wd_sin"] = np.sin(wd_radians)
    df["wd_cos"] = np.cos(wd_radians)

    continuous_cols = ["ws", "power", "nacelle_angle"]
    directional_cols = ["wd_sin", "wd_cos"]

    feature_blocks: list[pd.DataFrame] = []
    for turbine_id, group in df.groupby("turbine_id", sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        feature_frame = pd.DataFrame(index=group.index)

        for window in windows:
            window = int(window)
            min_periods = max(3, window // 2)
            for col in continuous_cols:
                rolling = group[col].rolling(window=window, min_periods=min_periods)
                prefix = f"m1_{col}_{window}m"
                feature_frame[f"{prefix}_mean"] = rolling.mean()
                feature_frame[f"{prefix}_std"] = rolling.std()
                feature_frame[f"{prefix}_min"] = rolling.min()
                feature_frame[f"{prefix}_max"] = rolling.max()
                feature_frame[f"{prefix}_last"] = group[col]
                feature_frame[f"{prefix}_ramp"] = group[col] - group[col].shift(window - 1)
            for col in directional_cols:
                rolling = group[col].rolling(window=window, min_periods=min_periods)
                prefix = f"m1_{col}_{window}m"
                feature_frame[f"{prefix}_mean"] = rolling.mean()
                feature_frame[f"{prefix}_std"] = rolling.std()
                feature_frame[f"{prefix}_last"] = group[col]
            feature_frame[f"m1_count_{window}m"] = (
                group["ws"].rolling(window=window, min_periods=1).count()
            )

        feature_frame["turbine_id"] = turbine_id
        feature_frame["timestamp"] = feature_frame.index
        if origin_index is not None:
            feature_frame = feature_frame.loc[feature_frame.index.isin(origin_index)]
        feature_blocks.append(feature_frame.reset_index(drop=True))

    if not feature_blocks:
        return pd.DataFrame(columns=["turbine_id", "timestamp"])
    return pd.concat(feature_blocks, ignore_index=True)


def load_turbine_metadata(path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="turbine_meta")
    df = pd.read_csv(resolved_path, encoding="utf-8-sig")
    df["longitude_deg"] = df["longitude_deg"].astype(float)
    df["latitude_deg"] = df["latitude_deg"].astype(float)
    return df.sort_values("turbine_id").reset_index(drop=True)


def load_tower_met_long(path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="tower_met")
    df = pd.read_excel(resolved_path, sheet_name="data_preQC")
    df = df.rename(columns=RAW_TOWER_TO_CANONICAL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["height_m"] = df["height_m"].astype(int)
    return df.sort_values(["timestamp", "height_m"]).reset_index(drop=True)


def build_tower_met_wide(path: str | Path) -> pd.DataFrame:
    long_df = load_tower_met_long(path)
    blocks: list[pd.DataFrame] = []
    for variable in TOWER_WIDE_VARIABLES:
        pivot = long_df.pivot(index="timestamp", columns="height_m", values=variable)
        pivot = pivot.sort_index()
        pivot.columns = [f"tower_{variable}_{int(height)}m" for height in pivot.columns]
        blocks.append(pivot)
    wide = pd.concat(blocks, axis=1).sort_index()
    keep_cols = [col for col in wide.columns if not wide[col].isna().all()]
    wide = wide.loc[:, keep_cols].reset_index()
    return wide


def select_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    return df.loc[:, list(columns)].copy()
