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
from xinyang_wind15.graph import build_distance_adjacency  # noqa: E402
from xinyang_wind15.loading import (  # noqa: E402
    build_scada_1min_aggregates,
    build_tower_met_wide,
    load_scada_15min,
    load_scada_1min,
    load_turbine_metadata,
)
from xinyang_wind15.settings import load_settings  # noqa: E402
from xinyang_wind15.windows import (  # noqa: E402
    build_spatiotemporal_windows,
    save_window_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-config",
        default=str(PROJECT_DIR / "configs" / "splits" / "xinyang_7_2_1.yaml"),
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
    parser.add_argument("--max-turbines", type=int, default=None)
    parser.add_argument("--tail-timestamps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.split_config)
    scada = load_scada_15min(
        settings.data_paths["scada_15min"],
        max_turbines=args.max_turbines,
        tail_timestamps=args.tail_timestamps,
    )
    tower_wide = None
    if args.include_tower:
        tower_wide = build_tower_met_wide(settings.data_paths["met_tower"])
    one_min_agg = None
    if args.include_1min:
        scada_1min = load_scada_1min(
            settings.data_paths["scada_1min"],
            max_turbines=args.max_turbines,
        )
        one_min_agg = build_scada_1min_aggregates(
            scada_1min,
            origin_timestamps=scada["timestamp"].unique(),
        )
    feature_frame = build_timestep_feature_frame(
        scada,
        tower_wide=tower_wide,
        one_min_agg=one_min_agg,
    )
    feature_columns = list(args.feature_columns)
    if args.include_tower:
        feature_columns.extend(
            col for col in feature_frame.columns if col.startswith("tower_")
        )
    if args.include_1min:
        feature_columns.extend(
            col for col in feature_frame.columns if col.startswith("m1_")
        )
    feature_columns = sorted(set(feature_columns))
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

    turbine_meta = load_turbine_metadata(settings.data_paths["turbine_meta"])
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
        "turbines": list(bundle["turbine_order"]),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
