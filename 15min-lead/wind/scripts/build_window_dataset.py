"""Build numpy window tensors for future GRU / STGNN experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xinyang_wind15.features import build_timestep_feature_frame  # noqa: E402
from xinyang_wind15.feature_presets import append_feature_block_columns, validate_feature_columns  # noqa: E402
from xinyang_wind15.graph import build_distance_adjacency  # noqa: E402
from xinyang_wind15.loading import (  # noqa: E402
    build_scada_1min_aggregates,
    build_tower_met_wide,
    filter_time_window,
    load_scada_15min,
    load_scada_1min,
    load_turbine_metadata,
)
from xinyang_wind15.settings import default_split_config_path  # noqa: E402
from xinyang_wind15.settings import load_settings  # noqa: E402
from xinyang_wind15.windows import (  # noqa: E402
    build_spatiotemporal_windows,
    save_window_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-config",
        default=str(default_split_config_path(PROJECT_DIR)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "window_dataset"),
    )
    parser.add_argument("--lookback-steps", type=int, default=32)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=["ws_mean", "power_mean", "wd_mean", "nacelle_mean", "ws_std"],
    )
    parser.add_argument("--include-tower", action="store_true")
    parser.add_argument("--include-1min", action="store_true")
    parser.add_argument("--include-derived-core", action="store_true")
    parser.add_argument("--include-spatial-context", action="store_true")
    parser.add_argument("--spatial-direction-sigma-deg", type=float, default=30.0)
    parser.add_argument("--spatial-distance-scale-km", type=float, default=1.0)
    parser.add_argument("--max-turbines", type=int, default=None)
    parser.add_argument("--tail-timestamps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.split_config)
    scada = load_scada_15min(
        settings.data_paths["scada_15min"],
        direction_path=settings.data_paths.get("scada_15min_direction"),
        max_turbines=args.max_turbines,
        tail_timestamps=args.tail_timestamps,
    )
    scada = filter_time_window(
        scada,
        timestamp_col="timestamp",
        start=settings.data_window_start,
        end=settings.data_window_end,
    )
    tower_wide = None
    if args.include_tower:
        tower_path = settings.data_paths.get("met_tower")
        if not tower_path:
            raise ValueError("--include-tower requires data.met_tower in the split config.")
        tower_wide = build_tower_met_wide(tower_path)
        tower_wide = filter_time_window(
            tower_wide,
            timestamp_col="timestamp",
            start=settings.data_window_start,
            end=settings.data_window_end,
        )
    one_min_agg = None
    if args.include_1min:
        one_min_path = settings.data_paths.get("scada_1min")
        if not one_min_path:
            raise ValueError("--include-1min requires data.scada_1min in the split config.")
        scada_1min = load_scada_1min(
            one_min_path,
            max_turbines=args.max_turbines,
        )
        scada_1min = filter_time_window(
            scada_1min,
            timestamp_col="timestamp",
            start=settings.data_window_start,
            end=settings.data_window_end,
        )
        one_min_agg = build_scada_1min_aggregates(
            scada_1min,
            origin_timestamps=scada["timestamp"].unique(),
        )
    turbine_meta = None
    if args.include_spatial_context:
        turbine_meta = load_turbine_metadata(
            settings.data_paths["turbine_meta"],
            site=settings.site,
        )
    feature_frame = build_timestep_feature_frame(
        scada,
        tower_wide=tower_wide,
        one_min_agg=one_min_agg,
        turbine_meta=turbine_meta,
        include_derived_core=bool(args.include_derived_core),
        include_spatial_context=bool(args.include_spatial_context),
        spatial_direction_sigma_deg=float(args.spatial_direction_sigma_deg),
        spatial_distance_scale_km=float(args.spatial_distance_scale_km),
    )
    feature_columns = list(args.feature_columns)
    feature_blocks: list[str] = []
    if args.include_tower:
        feature_blocks.append("tower")
    if args.include_1min:
        feature_blocks.append("one_min")
    if args.include_derived_core:
        feature_blocks.append("derived_core")
    if args.include_spatial_context:
        feature_blocks.append("spatial_context")
    feature_columns = append_feature_block_columns(
        feature_columns,
        frame_columns=list(feature_frame.columns),
        block_names=feature_blocks,
    )
    feature_columns = validate_feature_columns(feature_columns, frame=feature_frame)
    bundle = build_spatiotemporal_windows(
        feature_frame,
        feature_columns=feature_columns,
        target_column="ws_mean",
        lookback_steps=args.lookback_steps,
        horizon_steps=settings.horizon_steps,
        split_bounds=settings.split_bounds,
    )
    out_dir = Path(args.output_dir)
    save_window_bundle(bundle, out_dir)

    if turbine_meta is None:
        turbine_meta = load_turbine_metadata(
            settings.data_paths["turbine_meta"],
            site=settings.site,
        )
    adjacency = build_distance_adjacency(turbine_meta, bundle["turbine_order"])
    adjacency_path = out_dir / "distance_adjacency.npy"
    adjacency_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(adjacency_path, adjacency)

    summary = {
        "x_shape": list(bundle["x"].shape),
        "y_shape": list(bundle["y"].shape),
        "n_train": int((bundle["split"] == "train").sum()),
        "n_val": int((bundle["split"] == "val").sum()),
        "n_test": int((bundle["split"] == "test").sum()),
        "feature_columns": list(bundle["feature_columns"]),
        "include_tower": bool(args.include_tower),
        "include_1min": bool(args.include_1min),
        "include_derived_core": bool(args.include_derived_core),
        "include_spatial_context": bool(args.include_spatial_context),
        "spatial_direction_sigma_deg": float(args.spatial_direction_sigma_deg),
        "spatial_distance_scale_km": float(args.spatial_distance_scale_km),
        "turbines": list(bundle["turbine_order"]),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
