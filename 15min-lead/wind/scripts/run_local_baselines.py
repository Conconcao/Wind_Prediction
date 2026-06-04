"""Run local baseline models for xinyang 15-minute wind-speed forecasting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xinyang_wind15.baselines import (  # noqa: E402
    attach_skill_scores,
    build_farm_mean_series,
    lightgbm_baseline,
    persistence_baseline,
    sarima_farm_mean_baseline,
    seasonal_persistence_baseline,
)
from xinyang_wind15.features import build_supervised_frame  # noqa: E402
from xinyang_wind15.loading import (  # noqa: E402
    build_scada_1min_aggregates,
    build_tower_met_wide,
    load_scada_15min,
    load_scada_1min,
)
from xinyang_wind15.settings import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-config",
        default=str(PROJECT_DIR / "configs" / "splits" / "xinyang_7_2_1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "baseline_run"),
    )
    parser.add_argument(
        "--max-turbines",
        type=int,
        default=None,
        help="Optional debug filter on the alphabetically first N turbines.",
    )
    parser.add_argument(
        "--tail-timestamps",
        type=int,
        default=None,
        help="Optional debug filter keeping only the most recent N timestamps.",
    )
    parser.add_argument(
        "--include-tower",
        action="store_true",
        help="Merge tower features into the tabular baseline frame.",
    )
    parser.add_argument(
        "--include-1min",
        action="store_true",
        help="Merge 1-minute rolling aggregate features into the tabular baseline frame.",
    )
    parser.add_argument(
        "--skip-sarima",
        action="store_true",
        help="Skip the slower farm-mean SARIMA baseline.",
    )
    parser.add_argument(
        "--sarima-train-tail-points",
        type=int,
        default=1500,
        help="Tail length used for local SARIMA fitting.",
    )
    parser.add_argument(
        "--sarima-max-eval-points",
        type=int,
        default=None,
        help="Optional cap on validation/test points for local ARIMA smoke runs.",
    )
    return parser.parse_args()


def _write_prediction_artifacts(output_dir: Path, results: list) -> None:
    metrics_rows = []
    for result in results:
        metrics_rows.append({"model": result.name, "split": result.split, **result.metrics})
        pred_path = output_dir / f"{result.name}_{result.split}_predictions.csv"
        result.predictions.to_csv(pred_path, index=False)
        if result.per_turbine is not None:
            per_turbine_path = output_dir / f"{result.name}_{result.split}_per_turbine.csv"
            result.per_turbine.to_csv(per_turbine_path, index=False)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    (output_dir / "baseline_metrics.json").write_text(
        metrics_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            tail_timestamps=None,
        )
        one_min_agg = build_scada_1min_aggregates(
            scada_1min,
            origin_timestamps=scada["timestamp"].unique(),
        )

    frame = build_supervised_frame(
        scada,
        split_bounds=settings.split_bounds,
        horizon_steps=settings.horizon_steps,
        tower_wide=tower_wide,
        one_min_agg=one_min_agg,
    )

    persistence = persistence_baseline(frame)
    seasonal = seasonal_persistence_baseline(
        frame,
        season_steps=96,
        horizon_steps=settings.horizon_steps,
    )
    lightgbm = lightgbm_baseline(frame)

    all_results = []
    all_results.extend(persistence)
    all_results.extend(attach_skill_scores(seasonal, persistence_results=persistence))
    all_results.extend(attach_skill_scores(lightgbm, persistence_results=persistence))

    if not args.skip_sarima:
        farm_mean = build_farm_mean_series(scada)
        sarima = sarima_farm_mean_baseline(
            farm_mean,
            train_end=settings.split_bounds.train_end,
            val_end=settings.split_bounds.val_end,
            train_tail_points=args.sarima_train_tail_points,
            max_eval_points=args.sarima_max_eval_points,
        )
        all_results.extend(sarima)

    _write_prediction_artifacts(output_dir, all_results)
    summary = {
        "experiment_name": settings.experiment_name,
        "n_rows_scada": int(len(scada)),
        "n_turbines": int(scada["turbine_id"].nunique()),
        "include_tower": bool(args.include_tower),
        "include_1min": bool(args.include_1min),
        "timestamp_min": str(scada["timestamp"].min()),
        "timestamp_max": str(scada["timestamp"].max()),
        "output_dir": str(output_dir),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
