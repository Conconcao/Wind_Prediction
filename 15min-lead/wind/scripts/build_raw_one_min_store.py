"""Build a disk-backed store from raw 1-minute SCADA only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xinyang_wind15.graph import (  # noqa: E402
    build_bearing_matrix,
    build_correlation_adjacency,
    build_distance_adjacency,
)
from xinyang_wind15.loading import (  # noqa: E402
    filter_time_window,
    load_scada_1min,
    load_scada_15min_direction,
    load_turbine_metadata,
)
from xinyang_wind15.raw_one_min import (  # noqa: E402
    build_ratio_split_bounds,
    build_raw_one_min_feature_frame,
    default_raw_one_min_config_path,
    load_raw_one_min_settings,
    merge_direction_15min_snapshots,
    resolve_feature_columns,
)
from xinyang_wind15.window_store import write_window_store  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(default_raw_one_min_config_path(PROJECT_DIR)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "huaian_1min_raw_store"),
    )
    parser.add_argument("--lookback-steps", type=int, default=None)
    parser.add_argument("--max-turbines", type=int, default=None)
    parser.add_argument("--tail-timestamps", type=int, default=None)
    parser.add_argument("--include-time-features", action="store_true")
    parser.add_argument("--include-direction-if-available", action="store_true")
    parser.add_argument("--feature-columns", nargs="+", default=None)
    parser.add_argument("--min-target-coverage", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_raw_one_min_settings(args.config)
    if settings.base_resolution_minutes != 1:
        raise ValueError(
            f"build_raw_one_min_store.py expects 1-minute data, got base_resolution_minutes="
            f"{settings.base_resolution_minutes}."
        )
    if settings.target_mode != "point":
        raise ValueError(
            f"Only target_mode=point is currently supported for raw 1-minute forecasting, "
            f"got {settings.target_mode}."
        )

    scada_1min = load_scada_1min(
        settings.data_paths["scada_1min"],
        max_turbines=args.max_turbines,
    )
    scada_1min = filter_time_window(
        scada_1min,
        timestamp_col="timestamp",
        start=settings.data_window_start,
        end=settings.data_window_end,
    )
    if args.tail_timestamps is not None:
        keep_times = sorted(scada_1min["timestamp"].unique())[-int(args.tail_timestamps) :]
        scada_1min = scada_1min.loc[scada_1min["timestamp"].isin(keep_times)].copy()

    direction_snapshot_path = settings.data_paths.get("scada_15min_direction")
    direction_feature_coverage = None
    if direction_snapshot_path:
        direction_15min = load_scada_15min_direction(
            direction_snapshot_path,
            max_turbines=args.max_turbines,
        )
        direction_15min = filter_time_window(
            direction_15min,
            timestamp_col="timestamp",
            start=settings.data_window_start,
            end=settings.data_window_end,
        )
        if args.tail_timestamps is not None:
            keep_times = sorted(scada_1min["timestamp"].unique())[-int(args.tail_timestamps) :]
            min_keep_time = min(keep_times)
            direction_15min = direction_15min.loc[
                direction_15min["timestamp"] <= max(keep_times)
            ].copy()
            direction_15min = direction_15min.loc[
                direction_15min["timestamp"] >= (min_keep_time - pd.Timedelta(minutes=15))
            ].copy()
        scada_1min = merge_direction_15min_snapshots(scada_1min, direction_15min)
        if "wd_mean" in scada_1min.columns:
            direction_feature_coverage = float(scada_1min["wd_mean"].notna().mean())

    feature_frame = build_raw_one_min_feature_frame(
        scada_1min,
        include_time_features=bool(args.include_time_features),
        include_direction_if_available=bool(args.include_direction_if_available or direction_snapshot_path),
    )
    requested_feature_columns = (
        list(args.feature_columns)
        if args.feature_columns
        else list(settings.feature_columns)
    )
    feature_columns = resolve_feature_columns(
        feature_frame,
        requested=requested_feature_columns,
    )
    split_bounds = build_ratio_split_bounds(
        feature_frame["timestamp"].unique(),
        train_ratio=settings.split_ratio[0],
        val_ratio=settings.split_ratio[1],
        test_ratio=settings.split_ratio[2],
    )
    min_target_coverage = (
        float(args.min_target_coverage)
        if args.min_target_coverage is not None
        else float(settings.min_target_coverage)
    )
    lookback_steps = int(args.lookback_steps or settings.lookback_steps)

    out_dir = Path(args.output_dir)
    metadata = write_window_store(
        feature_frame,
        feature_columns=feature_columns,
        target_column=settings.target_column,
        lookback_steps=lookback_steps,
        horizon_steps=settings.horizon_steps,
        split_bounds=split_bounds,
        output_dir=out_dir,
        min_target_coverage=min_target_coverage,
    )

    turbine_meta = load_turbine_metadata(
        settings.data_paths["turbine_meta"],
        site=settings.site,
    )
    distance_adjacency = build_distance_adjacency(turbine_meta, metadata["turbine_order"])
    bearing_matrix = build_bearing_matrix(turbine_meta, metadata["turbine_order"])
    train_mask = (
        (feature_frame["timestamp"] >= split_bounds.train_start)
        & (feature_frame["timestamp"] <= split_bounds.train_end)
    )
    correlation_adjacency = build_correlation_adjacency(
        feature_frame.loc[train_mask].copy(),
        metadata["turbine_order"],
        value_column=settings.target_column,
        min_periods=max(30, lookback_steps),
    )
    np.save(out_dir / "distance_adjacency.npy", distance_adjacency)
    np.save(out_dir / "bearing_matrix.npy", bearing_matrix)
    np.save(out_dir / "correlation_adjacency.npy", correlation_adjacency)

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "experiment_name": settings.experiment_name,
        "site": settings.site,
        "target_column": settings.target_column,
        "target_mode": settings.target_mode,
        "base_resolution_minutes": settings.base_resolution_minutes,
        "horizon_steps": settings.horizon_steps,
        "lookback_steps": lookback_steps,
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "min_target_coverage": min_target_coverage,
        "direction_snapshot_path": direction_snapshot_path,
        "direction_snapshot_coverage": direction_feature_coverage,
        "n_rows": int(len(feature_frame)),
        "n_timestamps": int(feature_frame["timestamp"].nunique()),
        "n_turbines": int(feature_frame["turbine_id"].nunique()),
        "time_range": {
            "min": str(feature_frame["timestamp"].min()),
            "max": str(feature_frame["timestamp"].max()),
        },
        "split_bounds": {
            "train_start": str(split_bounds.train_start),
            "train_end": str(split_bounds.train_end),
            "val_start": str(split_bounds.val_start),
            "val_end": str(split_bounds.val_end),
            "test_start": str(split_bounds.test_start),
            "test_end": str(split_bounds.test_end),
        },
        "metadata": metadata,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
