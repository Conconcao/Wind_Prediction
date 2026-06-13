"""Feature engineering for xinyang wind-speed experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .graph import build_bearing_matrix, build_distance_matrix
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


def add_relative_direction_features(
    df: pd.DataFrame,
    *,
    wind_direction_col: str,
    yaw_direction_col: str,
    prefix: str = "yaw_error",
) -> pd.DataFrame:
    out = df.copy()
    yaw_error = ((out[wind_direction_col] - out[yaw_direction_col] + 180.0) % 360.0) - 180.0
    out[f"{prefix}_deg"] = yaw_error
    radians = np.deg2rad(yaw_error)
    out[f"{prefix}_sin"] = np.sin(radians)
    out[f"{prefix}_cos"] = np.cos(radians)
    out[f"{prefix}_abs"] = np.abs(yaw_error)
    return out


def _circular_difference_deg(
    lhs: pd.Series | np.ndarray,
    rhs: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    return ((lhs - rhs + 180.0) % 360.0) - 180.0


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


def _safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    *,
    min_abs_denominator: float = 1e-6,
) -> pd.Series | np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    out = np.full_like(numerator_array, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator_array)
        & np.isfinite(denominator_array)
        & (np.abs(denominator_array) > float(min_abs_denominator))
    )
    out[valid] = numerator_array[valid] / denominator_array[valid]
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return out


def add_operational_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["derived_ti_15m"] = _safe_divide(out["ws_std"], out["ws_mean"])
    out["derived_gust_factor_15m"] = _safe_divide(out["ws_max"], out["ws_mean"])
    out["derived_gust_excess_15m"] = out["ws_max"] - out["ws_mean"]
    out["derived_ws_range_15m"] = out["ws_max"] - out["ws_min"]
    return out


def add_tower_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    derived_feature_cols: list[str] = []

    shear_pairs = ((10, 125), (70, 125))
    for lower_height, upper_height in shear_pairs:
        lower_col = f"tower_ws_{lower_height}m"
        upper_col = f"tower_ws_{upper_height}m"
        if lower_col not in out.columns or upper_col not in out.columns:
            continue
        feature_name = f"profile_shear_alpha_{lower_height}m_{upper_height}m"
        lower_ws = out[lower_col].to_numpy(dtype=np.float64, copy=False)
        upper_ws = out[upper_col].to_numpy(dtype=np.float64, copy=False)
        shear_alpha = np.full(len(out), np.nan, dtype=np.float64)
        valid = (
            np.isfinite(lower_ws)
            & np.isfinite(upper_ws)
            & (lower_ws > 1e-6)
            & (upper_ws > 1e-6)
        )
        shear_alpha[valid] = np.log(upper_ws[valid] / lower_ws[valid]) / np.log(
            float(upper_height) / float(lower_height)
        )
        out[feature_name] = shear_alpha
        derived_feature_cols.append(feature_name)

    veer_pairs = ((10, 125), (70, 125))
    for lower_height, upper_height in veer_pairs:
        lower_col = f"tower_wd_{lower_height}m"
        upper_col = f"tower_wd_{upper_height}m"
        if lower_col not in out.columns or upper_col not in out.columns:
            continue
        veer_deg = _circular_difference_deg(out[upper_col], out[lower_col])
        radians = np.deg2rad(veer_deg)
        base_name = f"profile_veer_{lower_height}m_{upper_height}m"
        out[f"{base_name}_sin"] = np.sin(radians)
        out[f"{base_name}_cos"] = np.cos(radians)
        out[f"{base_name}_abs"] = np.abs(veer_deg)
        derived_feature_cols.extend(
            [
                f"{base_name}_sin",
                f"{base_name}_cos",
                f"{base_name}_abs",
            ]
        )

    for variable_name in ("temperature", "pressure"):
        lower_col = f"tower_{variable_name}_10m"
        upper_col = f"tower_{variable_name}_125m"
        if lower_col not in out.columns or upper_col not in out.columns:
            continue
        feature_name = f"profile_{variable_name}_delta_125m_10m"
        out[feature_name] = out[upper_col] - out[lower_col]
        derived_feature_cols.append(feature_name)

    if "tower_ws_125m" in out.columns:
        out["hub_tower_ws_125m_delta"] = out["ws_mean"] - out["tower_ws_125m"]
        derived_feature_cols.append("hub_tower_ws_125m_delta")

    if "tower_wd_125m" in out.columns:
        hub_tower_wd_deg = _circular_difference_deg(out["wd_mean"], out["tower_wd_125m"])
        hub_tower_wd_radians = np.deg2rad(hub_tower_wd_deg)
        out["hub_tower_wd_125m_sin"] = np.sin(hub_tower_wd_radians)
        out["hub_tower_wd_125m_cos"] = np.cos(hub_tower_wd_radians)
        out["hub_tower_wd_125m_abs"] = np.abs(hub_tower_wd_deg)
        derived_feature_cols.extend(
            [
                "hub_tower_wd_125m_sin",
                "hub_tower_wd_125m_cos",
                "hub_tower_wd_125m_abs",
            ]
        )

    if derived_feature_cols:
        out = _add_missing_indicators(out, derived_feature_cols)
    return out


def add_direction_aware_spatial_context(
    df: pd.DataFrame,
    *,
    turbine_meta: pd.DataFrame,
    wind_direction_col: str = "wd_mean",
    direction_sigma_deg: float = 30.0,
    distance_scale_km: float = 1.0,
) -> pd.DataFrame:
    if direction_sigma_deg <= 0.0:
        raise ValueError("direction_sigma_deg must be positive.")
    if distance_scale_km <= 0.0:
        raise ValueError("distance_scale_km must be positive.")

    out = df.copy()
    turbine_order = sorted(out["turbine_id"].unique())
    timestamps = pd.Index(sorted(out["timestamp"].unique()))
    n_turbines = len(turbine_order)
    n_timestamps = len(timestamps)
    timestamp_index = pd.Index(timestamps)

    wd_matrix = (
        out.pivot(index="timestamp", columns="turbine_id", values=wind_direction_col)
        .reindex(index=timestamp_index, columns=turbine_order)
        .to_numpy(dtype=np.float64)
    )
    ws_matrix = (
        out.pivot(index="timestamp", columns="turbine_id", values="ws_mean")
        .reindex(index=timestamp_index, columns=turbine_order)
        .to_numpy(dtype=np.float64)
    )
    power_matrix = (
        out.pivot(index="timestamp", columns="turbine_id", values="power_mean")
        .reindex(index=timestamp_index, columns=turbine_order)
        .to_numpy(dtype=np.float64)
    )

    distance_matrix = build_distance_matrix(turbine_meta, turbine_order).astype(np.float64)
    bearing_target_to_source = build_bearing_matrix(turbine_meta, turbine_order).astype(
        np.float64
    )
    distance_decay = np.exp(-distance_matrix / float(distance_scale_km))
    np.fill_diagonal(distance_decay, 0.0)

    feature_names = [
        "ctx_upwind_weight_sum",
        "ctx_upwind_count",
        "ctx_upwind_nearest_dist_km",
        "ctx_crosswind_nearest_abs_km",
        "ctx_upwind_ws_mean",
        "ctx_upwind_power_mean",
        "ctx_upwind_ws_gap",
        "ctx_upwind_power_gap",
    ]
    feature_arrays = {
        feature_name: np.full((n_timestamps, n_turbines), np.nan, dtype=np.float64)
        for feature_name in feature_names
    }

    for timestamp_idx in range(n_timestamps):
        wd_row = wd_matrix[timestamp_idx]
        ws_row = ws_matrix[timestamp_idx]
        power_row = power_matrix[timestamp_idx]

        diff_deg = _circular_difference_deg(bearing_target_to_source, wd_row[:, None])
        abs_diff_deg = np.abs(diff_deg)
        along_wind_km = distance_matrix * np.cos(np.deg2rad(diff_deg))
        crosswind_abs_km = np.abs(distance_matrix * np.sin(np.deg2rad(diff_deg)))
        angle_weights = np.exp(-0.5 * np.square(abs_diff_deg / float(direction_sigma_deg)))

        valid_target_direction = np.isfinite(wd_row)[:, None]
        valid_upwind = (along_wind_km > 0.0) & valid_target_direction
        weights = angle_weights * distance_decay * valid_upwind.astype(np.float64)
        np.fill_diagonal(weights, 0.0)

        feature_arrays["ctx_upwind_weight_sum"][timestamp_idx] = weights.sum(axis=1)
        feature_arrays["ctx_upwind_count"][timestamp_idx] = (weights > 1e-6).sum(axis=1)

        nearest_dist = np.where(valid_upwind, distance_matrix, np.inf).min(axis=1)
        nearest_crosswind = np.where(valid_upwind, crosswind_abs_km, np.inf).min(axis=1)
        nearest_dist[~np.isfinite(nearest_dist)] = 0.0
        nearest_crosswind[~np.isfinite(nearest_crosswind)] = 0.0
        feature_arrays["ctx_upwind_nearest_dist_km"][timestamp_idx] = nearest_dist
        feature_arrays["ctx_crosswind_nearest_abs_km"][timestamp_idx] = nearest_crosswind

        for value_row, mean_feature, gap_feature in [
            (ws_row, "ctx_upwind_ws_mean", "ctx_upwind_ws_gap"),
            (power_row, "ctx_upwind_power_mean", "ctx_upwind_power_gap"),
        ]:
            valid_sources = np.isfinite(value_row)[None, :]
            value_weights = weights * valid_sources.astype(np.float64)
            value_weight_sum = value_weights.sum(axis=1)
            weighted_mean = np.divide(
                value_weights @ np.nan_to_num(value_row, nan=0.0),
                value_weight_sum,
                out=np.full(n_turbines, np.nan, dtype=np.float64),
                where=value_weight_sum > 1e-6,
            )
            fallback_mask = value_weight_sum <= 1e-6
            weighted_mean[fallback_mask] = value_row[fallback_mask]
            feature_arrays[mean_feature][timestamp_idx] = weighted_mean
            feature_arrays[gap_feature][timestamp_idx] = value_row - weighted_mean

    context_frame = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [timestamp_index, turbine_order],
            names=["timestamp", "turbine_id"],
        )
    )
    for feature_name, feature_array in feature_arrays.items():
        context_frame[feature_name] = feature_array.reshape(-1)
    context_frame = context_frame.reset_index()

    out = out.merge(context_frame, on=["timestamp", "turbine_id"], how="left")
    missing_context_features = [
        feature_name for feature_name in feature_names if out[feature_name].isna().any()
    ]
    if missing_context_features:
        out = _add_missing_indicators(out, missing_context_features)
    return out


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
    turbine_meta: pd.DataFrame | None = None,
    include_derived_core: bool = False,
    include_spatial_context: bool = False,
    spatial_direction_sigma_deg: float = 30.0,
    spatial_distance_scale_km: float = 1.0,
) -> pd.DataFrame:
    out = scada_df.copy()
    out = out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    out = add_calendar_features(out, "timestamp")
    out = add_directional_features(out, "wd_mean", "wd")
    out = add_directional_features(out, "nacelle_mean", "nacelle")
    out = add_relative_direction_features(
        out,
        wind_direction_col="wd_mean",
        yaw_direction_col="nacelle_mean",
    )
    if include_derived_core:
        out = add_operational_proxy_features(out)
    if tower_wide is not None:
        out = merge_tower_features(out, tower_wide, origin_timestamp_col="timestamp")
        if include_derived_core:
            out = add_tower_profile_features(out)
        out = _impute_prefixed_columns(out, prefix="tower_", group_col="turbine_id")
        if include_derived_core:
            out = _impute_prefixed_columns(out, prefix="profile_", group_col="turbine_id")
            out = _impute_prefixed_columns(out, prefix="hub_tower_", group_col="turbine_id")
    if one_min_agg is not None:
        out = merge_scada_1min_features(out, one_min_agg, origin_timestamp_col="timestamp")
        out = _impute_prefixed_columns(out, prefix="m1_", group_col="turbine_id")
    if include_spatial_context:
        if turbine_meta is None:
            raise ValueError("turbine_meta is required when include_spatial_context=True.")
        out = add_direction_aware_spatial_context(
            out,
            turbine_meta=turbine_meta,
            wind_direction_col="wd_mean",
            direction_sigma_deg=spatial_direction_sigma_deg,
            distance_scale_km=spatial_distance_scale_km,
        )
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
    turbine_meta: pd.DataFrame | None = None,
    include_derived_core: bool = False,
    include_spatial_context: bool = False,
    spatial_direction_sigma_deg: float = 30.0,
    spatial_distance_scale_km: float = 1.0,
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
    out = add_relative_direction_features(
        out,
        wind_direction_col="wd_mean",
        yaw_direction_col="nacelle_mean",
    )
    if include_derived_core:
        out = add_operational_proxy_features(out)

    if tower_wide is not None:
        out = merge_tower_features(
            out,
            tower_wide,
            origin_timestamp_col="origin_timestamp",
        )
        if include_derived_core:
            out = add_tower_profile_features(out)
        out = _impute_prefixed_columns(out, prefix="tower_", group_col="turbine_id")
        if include_derived_core:
            out = _impute_prefixed_columns(out, prefix="profile_", group_col="turbine_id")
            out = _impute_prefixed_columns(out, prefix="hub_tower_", group_col="turbine_id")
    if one_min_agg is not None:
        out = merge_scada_1min_features(
            out,
            one_min_agg,
            origin_timestamp_col="origin_timestamp",
        )
        out = _impute_prefixed_columns(out, prefix="m1_", group_col="turbine_id")
    if include_spatial_context:
        if turbine_meta is None:
            raise ValueError("turbine_meta is required when include_spatial_context=True.")
        out = add_direction_aware_spatial_context(
            out,
            turbine_meta=turbine_meta,
            wind_direction_col="wd_mean",
            direction_sigma_deg=spatial_direction_sigma_deg,
            distance_scale_km=spatial_distance_scale_km,
        )

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
