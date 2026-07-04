"""Data loading and normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

from .schema import (
    RAW_SCADA_15MIN_DIRECTION_TO_CANONICAL,
    RAW_SCADA_15MIN_TO_CANONICAL,
    RAW_SCADA_1MIN_TO_CANONICAL,
    RAW_TOWER_TO_CANONICAL,
    SCADA_15MIN_BASE_COLUMNS,
    SCADA_15MIN_DIRECTION_COLUMNS,
    SCADA_1MIN_BASE_COLUMNS,
    TOWER_WIDE_VARIABLES,
)


_FALLBACK_PATTERNS: dict[str, tuple[str, ...]] = {
    "scada_15min": ("ALL_TURBINES_15min_*.parquet",),
    "scada_15min_direction": ("ALL_TURBINES_DIRECTION_15min_*.parquet",),
    "scada_1min": ("ALL_TURBINES_1min_*.parquet",),
    "tower_met": ("pre_QC*.xlsx", "*气象*观测*.xlsx"),
    "turbine_meta": (
        "*风机基本信息汇总*.csv",
        "*风机基本信息*.csv",
        "*椋庢満鍩烘湰淇℃伅*.csv",
    ),
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


def filter_time_window(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out[timestamp_col] >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out[timestamp_col] <= pd.Timestamp(end)]
    return out.copy()


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


def _ensure_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def _normalize_canonical_frame(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    out = _ensure_columns(df, required_columns)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["turbine_id"] = out["turbine_id"].astype(str).str.strip()
    numeric_columns = [
        column
        for column in out.columns
        if column not in {"turbine_id", "timestamp"}
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _load_scada_15min_direction(path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="scada_15min_direction")
    df = pd.read_parquet(resolved_path)
    df = df.rename(columns=RAW_SCADA_15MIN_DIRECTION_TO_CANONICAL)
    df = _normalize_canonical_frame(
        df,
        required_columns=["turbine_id", "timestamp", *SCADA_15MIN_DIRECTION_COLUMNS],
    )
    keep_columns = ["turbine_id", "timestamp", *SCADA_15MIN_DIRECTION_COLUMNS]
    return (
        df.loc[:, keep_columns]
        .drop_duplicates(subset=["turbine_id", "timestamp"], keep="last")
        .sort_values(["turbine_id", "timestamp"])
        .reset_index(drop=True)
    )


def load_scada_15min_direction(
    path: str | Path,
    *,
    max_turbines: int | None = None,
    turbine_ids: Sequence[str] | None = None,
    tail_timestamps: int | None = None,
) -> pd.DataFrame:
    df = _load_scada_15min_direction(path)
    df = _filter_specific_turbines(df, "turbine_id", turbine_ids)
    df = _filter_turbines(df, "turbine_id", max_turbines)
    df = _filter_time_tail(df, "timestamp", tail_timestamps)
    return df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


def _merge_scada_15min_direction(
    scada_df: pd.DataFrame,
    direction_df: pd.DataFrame,
) -> pd.DataFrame:
    renamed_direction = direction_df.rename(
        columns={
            column: f"{column}__direction"
            for column in SCADA_15MIN_DIRECTION_COLUMNS
        }
    )
    out = scada_df.merge(
        renamed_direction,
        on=["turbine_id", "timestamp"],
        how="left",
    )
    for column in SCADA_15MIN_DIRECTION_COLUMNS:
        direction_column = f"{column}__direction"
        if direction_column not in out.columns:
            continue
        if column in out.columns:
            out[column] = out[column].combine_first(out[direction_column])
        else:
            out[column] = out[direction_column]
        out = out.drop(columns=[direction_column])
    return out


def load_scada_15min(
    path: str | Path,
    *,
    direction_path: str | Path | None = None,
    max_turbines: int | None = None,
    turbine_ids: Sequence[str] | None = None,
    tail_timestamps: int | None = None,
) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="scada_15min")
    df = pd.read_parquet(resolved_path)
    df = df.rename(columns=RAW_SCADA_15MIN_TO_CANONICAL)
    df = _normalize_canonical_frame(df, required_columns=["turbine_id", "timestamp"])
    if direction_path is not None:
        direction_df = _load_scada_15min_direction(direction_path)
        df = _merge_scada_15min_direction(df, direction_df)
    df = _normalize_canonical_frame(df, required_columns=SCADA_15MIN_BASE_COLUMNS)
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
    df = _normalize_canonical_frame(df, required_columns=SCADA_1MIN_BASE_COLUMNS)
    df = _filter_specific_turbines(df, "turbine_id", turbine_ids)
    df = _filter_turbines(df, "turbine_id", max_turbines)
    df = _filter_time_tail(df, "timestamp", tail_timestamps)
    df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    return df


def _circular_mean_deg(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    radians = np.deg2rad(np.mod(arr, 360.0))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if np.isclose(sin_mean, 0.0) and np.isclose(cos_mean, 0.0):
        return float("nan")
    angle = float(np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0))
    if np.isclose(angle, 360.0):
        return 0.0
    return angle


def _circular_std_deg(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    radians = np.deg2rad(np.mod(arr, 360.0))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    resultant = np.hypot(sin_mean, cos_mean)
    if not np.isfinite(resultant) or resultant <= 0.0:
        return float("nan")
    clipped = float(np.clip(resultant, 1e-12, 1.0))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(clipped))))


def aggregate_scada_1min_to_15min(
    scada_1min: pd.DataFrame,
    *,
    freq: str = "15min",
) -> pd.DataFrame:
    """Aggregate canonical 1-minute SCADA into canonical 15-minute SCADA."""
    df = scada_1min.copy()
    if df.empty:
        return pd.DataFrame(columns=SCADA_15MIN_BASE_COLUMNS)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["bucket_timestamp"] = df["timestamp"].dt.floor(freq)
    blocks: list[pd.DataFrame] = []
    for turbine_id, group in df.groupby("turbine_id", sort=False):
        indexed = group.sort_values("timestamp").set_index("timestamp")
        out = pd.DataFrame(index=indexed.resample(freq).size().index)

        if "ws" in indexed.columns:
            out["ws_mean"] = indexed["ws"].resample(freq).mean()
            out["ws_max"] = indexed["ws"].resample(freq).max()
            out["ws_min"] = indexed["ws"].resample(freq).min()
            out["ws_std"] = indexed["ws"].resample(freq).std()
            out["cnt_raw"] = indexed["ws"].resample(freq).count()

        if "power" in indexed.columns:
            out["power_mean"] = indexed["power"].resample(freq).mean()
            out["power_max"] = indexed["power"].resample(freq).max()
            out["power_min"] = indexed["power"].resample(freq).min()
            out["power_std"] = indexed["power"].resample(freq).std()

        if "wd" in indexed.columns and not indexed["wd"].isna().all():
            wd_resample = indexed["wd"].resample(freq)
            out["wd_mean"] = wd_resample.apply(_circular_mean_deg)
            out["wd_max"] = wd_resample.max()
            out["wd_min"] = wd_resample.min()
            out["wd_std"] = wd_resample.apply(_circular_std_deg)

        if "nacelle_angle" in indexed.columns and not indexed["nacelle_angle"].isna().all():
            nacelle_resample = indexed["nacelle_angle"].resample(freq)
            out["nacelle_mean"] = nacelle_resample.apply(_circular_mean_deg)
            out["nacelle_max"] = nacelle_resample.max()
            out["nacelle_min"] = nacelle_resample.min()
            out["nacelle_std"] = nacelle_resample.apply(_circular_std_deg)

        out["turbine_id"] = str(turbine_id).strip()
        out["timestamp"] = out.index
        if "cnt_raw" in out.columns:
            out = out.loc[out["cnt_raw"] > 0].copy()
        blocks.append(out.reset_index(drop=True))

    if not blocks:
        return pd.DataFrame(columns=SCADA_15MIN_BASE_COLUMNS)
    result = pd.concat(blocks, ignore_index=True)
    result = _normalize_canonical_frame(result, required_columns=SCADA_15MIN_BASE_COLUMNS)
    return result.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


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

    directional_cols: list[str] = []
    if "wd" in df.columns and not df["wd"].isna().all():
        wd_radians = np.deg2rad(df["wd"] % 360.0)
        df["wd_sin"] = np.sin(wd_radians)
        df["wd_cos"] = np.cos(wd_radians)
        directional_cols = ["wd_sin", "wd_cos"]

    continuous_cols = [
        col
        for col in ("ws", "power", "nacelle_angle")
        if col in df.columns and not df[col].isna().all()
    ]

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


def load_turbine_metadata(
    path: str | Path,
    *,
    site: str | None = None,
) -> pd.DataFrame:
    resolved_path = _resolve_input_path(path, kind="turbine_meta")
    df = pd.read_csv(resolved_path, encoding="utf-8-sig")
    if site is not None and "site" in df.columns:
        normalized_site = str(site).strip().lower()
        site_values = df["site"].astype(str).str.strip().str.lower()
        df = df.loc[site_values == normalized_site].copy()
    df["turbine_id"] = df["turbine_id"].astype(str).str.strip()
    duplicate_ids = (
        df.loc[df["turbine_id"].duplicated(keep=False), "turbine_id"].drop_duplicates().tolist()
    )
    if duplicate_ids:
        raise ValueError(
            "Duplicate turbine_id values found in turbine metadata after site filtering: "
            f"{duplicate_ids[:10]}"
        )
    df["longitude_deg"] = pd.to_numeric(df["longitude_deg"], errors="coerce")
    df["latitude_deg"] = pd.to_numeric(df["latitude_deg"], errors="coerce")
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
