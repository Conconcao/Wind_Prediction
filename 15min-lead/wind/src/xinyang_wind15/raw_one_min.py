"""Helpers for raw 1-minute SCADA forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from .features import add_directional_features, add_relative_direction_features
from .schema import SCADA_15MIN_DIRECTION_COLUMNS
from .settings import SplitBounds


@dataclass(frozen=True)
class RawOneMinSettings:
    experiment_name: str
    site: str
    target_column: str
    horizon_steps: int
    base_resolution_minutes: int
    data_paths: dict[str, str]
    data_window_start: pd.Timestamp | None
    data_window_end: pd.Timestamp | None
    split_ratio: tuple[float, float, float]
    lookback_steps: int
    feature_columns: list[str]
    min_target_coverage: float
    target_mode: str
    raw: dict[str, Any]


def _to_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def default_raw_one_min_config_path(
    project_dir: str | Path,
    *,
    prefer_server: bool | None = None,
) -> Path:
    root = Path(project_dir)
    splits_dir = root / "configs" / "splits"
    local_config = splits_dir / "huaian_1min_raw_7_1_2.yaml"
    server_config = splits_dir / "huaian_1min_raw_7_1_2_server.yaml"
    use_server = (Path("/").exists() and Path.cwd().anchor == "/") if prefer_server is None else bool(prefer_server)
    if use_server and server_config.exists():
        return server_config
    return local_config


def load_raw_one_min_settings(path: str | Path) -> RawOneMinSettings:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = raw["experiment"]
    split = raw["split"]
    modeling = raw.get("modeling", {})
    ratio = split["ratio"]
    split_ratio = (
        float(ratio["train"]),
        float(ratio["val"]),
        float(ratio["test"]),
    )
    if not np.isclose(sum(split_ratio), 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {split_ratio}.")
    return RawOneMinSettings(
        experiment_name=str(experiment["name"]),
        site=str(experiment["site"]),
        target_column=str(experiment.get("target_column", "ws")),
        horizon_steps=int(experiment["horizon_steps"]),
        base_resolution_minutes=int(experiment["base_resolution_minutes"]),
        data_paths={str(k): str(v) for k, v in raw["data"].items()},
        data_window_start=_to_optional_timestamp(split.get("data_window_start")),
        data_window_end=_to_optional_timestamp(split.get("data_window_end")),
        split_ratio=split_ratio,
        lookback_steps=int(modeling.get("lookback_steps", 60)),
        feature_columns=[str(x) for x in modeling.get("feature_columns", ["ws"])],
        min_target_coverage=float(modeling.get("min_target_coverage", 1.0)),
        target_mode=str(experiment.get("target_mode", "point")),
        raw=raw,
    )


def build_ratio_split_bounds(
    timestamps: Sequence[pd.Timestamp] | pd.Series | pd.Index | np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> SplitBounds:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0.0):
        raise ValueError(f"Split ratios must be positive, got {ratios.tolist()}.")
    if not np.isclose(ratios.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios.tolist()}.")

    unique_times = pd.Index(pd.to_datetime(pd.Index(timestamps).unique())).sort_values()
    n_times = len(unique_times)
    if n_times < 3:
        raise ValueError(f"Need at least 3 distinct timestamps to split, got {n_times}.")

    train_count = max(1, int(np.floor(n_times * train_ratio)))
    val_count = max(1, int(np.floor(n_times * val_ratio)))
    test_count = n_times - train_count - val_count
    if test_count < 1:
        deficit = 1 - test_count
        if val_count > train_count:
            val_count -= deficit
        else:
            train_count -= deficit
        test_count = 1
    if train_count < 1 or val_count < 1 or test_count < 1:
        raise ValueError(
            "Unable to derive non-empty chronological splits from the provided timestamps. "
            f"Counts: train={train_count}, val={val_count}, test={test_count}."
        )

    train_end_idx = train_count - 1
    val_end_idx = train_end_idx + val_count
    return SplitBounds(
        train_start=pd.Timestamp(unique_times[0]),
        train_end=pd.Timestamp(unique_times[train_end_idx]),
        val_start=pd.Timestamp(unique_times[train_end_idx + 1]),
        val_end=pd.Timestamp(unique_times[val_end_idx]),
        test_start=pd.Timestamp(unique_times[val_end_idx + 1]),
        test_end=pd.Timestamp(unique_times[-1]),
    )


def add_time_cyclic_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hour_float = out["timestamp"].dt.hour + (out["timestamp"].dt.minute / 60.0)
    day_of_year = out["timestamp"].dt.dayofyear.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    return out


def merge_direction_15min_snapshots(
    scada_1min: pd.DataFrame,
    direction_15min: pd.DataFrame,
    *,
    tolerance_minutes: int = 15,
) -> pd.DataFrame:
    if direction_15min.empty:
        return scada_1min.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    direction_columns = [
        column
        for column in SCADA_15MIN_DIRECTION_COLUMNS
        if column in direction_15min.columns and not direction_15min[column].isna().all()
    ]
    if not direction_columns:
        return scada_1min.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    tolerance = pd.Timedelta(minutes=int(tolerance_minutes))
    right = direction_15min.loc[:, ["turbine_id", "timestamp", *direction_columns]].copy()
    merged_blocks: list[pd.DataFrame] = []
    for turbine_id, left_group in scada_1min.groupby("turbine_id", sort=False):
        left_group = left_group.sort_values("timestamp").reset_index(drop=True).copy()
        right_group = (
            right.loc[right["turbine_id"] == turbine_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
            .copy()
        )
        if right_group.empty:
            for column in direction_columns:
                left_group[column] = np.nan
            merged_blocks.append(left_group)
            continue
        merged_group = pd.merge_asof(
            left_group,
            right_group,
            on="timestamp",
            direction="backward",
            tolerance=tolerance,
            suffixes=("", "_direction"),
        )
        if "turbine_id_direction" in merged_group.columns:
            merged_group = merged_group.drop(columns=["turbine_id_direction"])
        merged_blocks.append(merged_group)
    return pd.concat(merged_blocks, ignore_index=True).sort_values(
        ["turbine_id", "timestamp"]
    ).reset_index(drop=True)


def build_raw_one_min_feature_frame(
    scada_1min: pd.DataFrame,
    *,
    include_time_features: bool = False,
    include_direction_if_available: bool = False,
) -> pd.DataFrame:
    out = scada_1min.loc[:, ["turbine_id", "timestamp", "ws"]].copy()
    out["ws"] = pd.to_numeric(out["ws"], errors="coerce")
    if include_direction_if_available:
        for column in ("wd_mean", "wd_std", "nacelle_mean", "nacelle_std"):
            if column in scada_1min.columns and scada_1min[column].notna().any():
                out[column] = pd.to_numeric(scada_1min[column], errors="coerce")
        if "wd_mean" in out.columns:
            out = add_directional_features(out, source_col="wd_mean", prefix="wd")
        elif "wd" in scada_1min.columns and scada_1min["wd"].notna().any():
            wd = pd.to_numeric(scada_1min["wd"], errors="coerce")
            wd_rad = np.deg2rad(np.remainder(wd, 360.0))
            out["wd_sin"] = np.sin(wd_rad)
            out["wd_cos"] = np.cos(wd_rad)
        if "nacelle_mean" in out.columns:
            out = add_directional_features(out, source_col="nacelle_mean", prefix="nacelle")
        if "wd_mean" in out.columns and "nacelle_mean" in out.columns:
            out = add_relative_direction_features(
                out,
                wind_direction_col="wd_mean",
                yaw_direction_col="nacelle_mean",
            )
    if include_time_features:
        out = add_time_cyclic_features(out)
    return out.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)


def resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    requested: Sequence[str],
) -> list[str]:
    selected = [str(name) for name in requested]
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"Requested raw 1-minute features are missing: {missing}")
    all_nan = [name for name in selected if frame[name].isna().all()]
    if all_nan:
        raise ValueError(f"Requested raw 1-minute features are entirely missing: {all_nan}")
    return selected
