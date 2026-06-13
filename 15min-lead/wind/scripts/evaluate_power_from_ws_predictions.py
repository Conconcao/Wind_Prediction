"""Map wind-speed forecasts to power forecasts with WTPC basic-feature MVPC."""

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

from xinyang_wind15.power_curve_bridge import (  # noqa: E402
    DEFAULT_PREPARED_SUBDIR,
    DEFAULT_RATED_POWER_KW,
    BASIC_CURVE_FEATURE_COLUMNS,
    build_basic_feature_frame,
    build_static_info_table,
    default_wtpc_root,
    ensure_wtpc_importable,
    evaluate_curve_sanity,
    fit_basic_curve_model,
    load_prepared_splits,
    load_ws_prediction_frame,
    merge_predictions_with_actuals,
    predict_power_norm,
    resolve_prepared_dir,
    summarize_power_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-predictions", required=True, help="CSV or parquet file with turbine_id, target_timestamp, and y_pred/ws_pred.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "power_from_ws_predictions"),
    )
    parser.add_argument(
        "--wtpc-root",
        default=str(default_wtpc_root()),
        help="Sibling WTPC project root used for prepared data and model helpers.",
    )
    parser.add_argument(
        "--prepared-dir",
        default="",
        help=(
            "Optional WTPC prepared split directory. "
            f"Defaults to <wtpc-root>/{DEFAULT_PREPARED_SUBDIR.as_posix()}."
        ),
    )
    parser.add_argument(
        "--curve-model",
        choices=["lgbm", "hgb", "ridge"],
        default="lgbm",
    )
    parser.add_argument(
        "--fit-split",
        choices=["train", "trainval"],
        default="train",
        help="Power-curve fitting split. Default keeps the 7:2:1 protocol strict by fitting only on train.",
    )
    parser.add_argument("--rated-power-kw", type=float, default=DEFAULT_RATED_POWER_KW)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wtpc_root = ensure_wtpc_importable(args.wtpc_root)
    prepared_dir = resolve_prepared_dir(
        wtpc_root=wtpc_root,
        prepared_dir=args.prepared_dir or None,
    )
    prepared = load_prepared_splits(prepared_dir)
    static_info = build_static_info_table([prepared["train"], prepared["val"], prepared["test"]])

    fit_frames = (
        [prepared["train"]]
        if args.fit_split == "train"
        else [prepared["train"], prepared["val"]]
    )
    fit_frame = pd.concat(fit_frames, ignore_index=True)
    model, feature_columns = fit_basic_curve_model(
        fit_frame,
        wtpc_root=wtpc_root,
        model_name=args.curve_model,
        random_state=args.random_state,
    )

    ws_predictions = load_ws_prediction_frame(args.ws_predictions)
    joined = merge_predictions_with_actuals(
        ws_predictions,
        {
            "train": prepared["train"],
            "val": prepared["val"],
            "test": prepared["test"],
        },
    )
    inference_base = joined.rename(columns={"target_timestamp": "time", "ws_pred": "ws"}).copy()
    inference_features = build_basic_feature_frame(
        inference_base,
        static_info=static_info,
        wtpc_root=wtpc_root,
        ws_col="ws",
        time_col="time",
    )
    power_pred_norm = predict_power_norm(
        model,
        inference_features,
        feature_columns=feature_columns,
    )

    result = joined.copy()
    result["power_pred_norm"] = power_pred_norm
    result["power_pred_kw"] = result["power_pred_norm"] * float(args.rated_power_kw)
    if "power_true_kw" not in result.columns:
        result["power_true_kw"] = result["power_true_norm"] * float(args.rated_power_kw)
    result["q_sample"] = (
        1.0 - (result["power_pred_kw"] - result["power_true_kw"]).abs() / float(args.rated_power_kw)
    ) * 100.0

    overall_df, per_turbine_df, per_split_df = summarize_power_predictions(
        result,
        rated_power_kw=float(args.rated_power_kw),
        wtpc_root=wtpc_root,
    )
    sanity_df = evaluate_curve_sanity(
        model,
        {"val": prepared["val"], "test": prepared["test"]},
        feature_columns=feature_columns,
        rated_power_kw=float(args.rated_power_kw),
        wtpc_root=wtpc_root,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_out = result.rename(columns={"time": "target_timestamp"}).copy()
    result_out.to_csv(output_dir / "power_predictions.csv", index=False)
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    per_turbine_df.to_csv(output_dir / "per_turbine_metrics.csv", index=False)
    per_split_df.to_csv(output_dir / "per_split_metrics.csv", index=False)
    sanity_df.to_csv(output_dir / "curve_sanity_metrics.csv", index=False)

    summary = {
        "ws_predictions": str(Path(args.ws_predictions).resolve()),
        "wtpc_root": str(wtpc_root),
        "prepared_dir": str(prepared_dir),
        "curve_model": args.curve_model,
        "fit_split": args.fit_split,
        "rated_power_kw": float(args.rated_power_kw),
        "random_state": int(args.random_state),
        "n_input_rows": int(len(ws_predictions)),
        "n_matched_rows": int(len(result)),
        "feature_columns": feature_columns,
        "feature_column_count": int(len(feature_columns)),
        "feature_preset": "basic_wtpc_mvpc",
        "feature_candidates": BASIC_CURVE_FEATURE_COLUMNS,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "summary": summary,
                "overall_metrics": overall_df.to_dict(orient="records"),
                "per_split_metrics": per_split_df.to_dict(orient="records"),
                "curve_sanity_metrics": sanity_df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
