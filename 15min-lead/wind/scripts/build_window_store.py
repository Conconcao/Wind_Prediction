"""Build a disk-backed time-major window store for larger deep-learning runs."""

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
from xinyang_wind15.feature_presets import resolve_feature_columns  # noqa: E402
from xinyang_wind15.graph import (  # noqa: E402
    build_correlation_adjacency,
    build_distance_adjacency,
)
from xinyang_wind15.loading import (  # noqa: E402
    build_scada_1min_aggregates,
    build_tower_met_wide,
    load_scada_15min,
    load_scada_1min,
    load_turbine_metadata,
)
from xinyang_wind15.settings import load_settings  # noqa: E402
from xinyang_wind15.window_store import (  # noqa: E402
    estimate_window_store_bytes,
    write_window_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-config",
        default=str(PROJECT_DIR / "configs" / "splits" / "xinyang_7_2_1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "window_store"),
    )
    parser.add_argument("--lookback-steps", type=int, default=32)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--feature-preset",
        default="default_multivariate",
        choices=["default_multivariate", "hub_ws_only", "scada_core"],
    )
    parser.add_argument("--include-tower", action="store_true")
    parser.add_argument("--include-1min", action="store_true")
    parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=0.85,
        help="Minimum fraction of turbine targets that must be observed at the forecast step.",
    )
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
    feature_columns = resolve_feature_columns(
        feature_preset=args.feature_preset,
        feature_columns=args.feature_columns,
    )
    if args.include_tower:
        feature_columns.extend(
            col for col in feature_frame.columns if col.startswith("tower_")
        )
    if args.include_1min:
        feature_columns.extend(
            col for col in feature_frame.columns if col.startswith("m1_")
        )
    feature_columns = sorted(set(feature_columns))

    out_dir = Path(args.output_dir)
    metadata = write_window_store(
        feature_frame,
        feature_columns=feature_columns,
        target_column="ws_mean",
        lookback_steps=args.lookback_steps,
        horizon_steps=settings.horizon_steps,
        split_bounds=settings.split_bounds,
        output_dir=out_dir,
        min_target_coverage=args.min_target_coverage,
    )

    turbine_meta = load_turbine_metadata(settings.data_paths["turbine_meta"])
    adjacency = build_distance_adjacency(turbine_meta, metadata["turbine_order"])
    train_scada = scada.loc[
        (scada["timestamp"] >= settings.split_bounds.train_start)
        & (scada["timestamp"] <= settings.split_bounds.train_end)
    ].copy()
    correlation_adjacency = build_correlation_adjacency(
        train_scada,
        metadata["turbine_order"],
        value_column="ws_mean",
    )
    import numpy as np

    np.save(out_dir / "distance_adjacency.npy", adjacency)
    np.save(out_dir / "correlation_adjacency.npy", correlation_adjacency)
    store_bytes = estimate_window_store_bytes(
        n_timestamps=int(metadata["feature_tensor_shape"][0]),
        n_turbines=int(metadata["feature_tensor_shape"][1]),
        n_features=int(metadata["feature_tensor_shape"][2]),
    )
    summary = {
        **metadata,
        "estimated_store_gib": store_bytes / (1024**3),
        "distance_adjacency_path": str(out_dir / "distance_adjacency.npy"),
        "correlation_adjacency_path": str(out_dir / "correlation_adjacency.npy"),
        "include_tower": bool(args.include_tower),
        "include_1min": bool(args.include_1min),
        "feature_preset": str(args.feature_preset),
        "min_target_coverage": float(args.min_target_coverage),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
