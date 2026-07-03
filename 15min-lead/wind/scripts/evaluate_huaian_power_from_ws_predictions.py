"""Evaluate Huai'an wind-speed forecasts through WTPC multivariate power curves."""

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
    DEFAULT_RATED_POWER_KW,
    HUAIAN_G3_CURVE_CATEGORICAL_COLUMNS,
    HUAIAN_G3_CURVE_FEATURE_COLUMNS,
    HUAIAN_G5_CURVE_CATEGORICAL_COLUMNS,
    HUAIAN_G5_CURVE_FEATURE_COLUMNS,
    HUAIAN_G6_CURVE_CATEGORICAL_COLUMNS,
    HUAIAN_G6_CURVE_FEATURE_COLUMNS,
    HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_CATEGORICAL_COLUMNS,
    HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS,
    HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_CATEGORICAL_COLUMNS,
    HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_FEATURE_COLUMNS,
    default_wtpc_root,
    evaluate_curve_sanity,
    fit_curve_model,
    load_huaian_local_curve_splits,
    load_ws_prediction_frames,
    merge_predictions_with_reference_frame,
    predict_power_norm,
    summarize_power_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ws-predictions",
        nargs="+",
        required=True,
        help="One or more CSV/parquet files with turbine_id, target_timestamp, and y_pred/ws_pred.",
    )
    parser.add_argument(
        "--split-config",
        default=str(PROJECT_DIR / "configs" / "splits" / "huaian_7_2_1.yaml"),
        help="Huai'an local split config used by the wind-speed baselines.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "huaian_power_from_ws_predictions"),
    )
    parser.add_argument(
        "--wtpc-root",
        default=str(default_wtpc_root()),
        help="Sibling WTPC project root used for shared power-curve helpers.",
    )
    parser.add_argument(
        "--curve-model",
        choices=["lgbm", "hgb", "ridge"],
        default="hgb",
    )
    parser.add_argument(
        "--curve-feature-set",
        choices=[
            "huaian_g3_direct",
            "huaian_g5_minimal_directional",
            "huaian_g6_history",
            "huaian_latest_history_no_direction",
            "huaian_latest_history_past_direction_only",
        ],
        default="huaian_g6_history",
        help="Huai'an WTPC curve feature definition. Default uses the latest history-enhanced G6 set.",
    )
    parser.add_argument(
        "--fit-split",
        choices=["train", "trainval"],
        default="train",
        help="Curve fitting split. Default keeps the 7:2:1 protocol strict by fitting only on train.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=350000,
        help="Optional cap on rows used to fit the power curve. Set <=0 to disable sampling.",
    )
    parser.add_argument("--rated-power-kw", type=float, default=DEFAULT_RATED_POWER_KW)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-label",
        default="",
        help="Optional label stored in the summary for the upstream wind-speed model.",
    )
    return parser.parse_args()


def _maybe_sample_fit_frame(
    frame: pd.DataFrame,
    *,
    max_train_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_train_rows <= 0 or len(frame) <= max_train_rows:
        return frame
    return frame.sample(n=int(max_train_rows), random_state=int(random_state)).sort_values(
        ["turbine_id", "time"],
        kind="mergesort",
    )


def _resolve_curve_feature_spec(name: str) -> tuple[list[str], list[str], bool]:
    if name == "huaian_g3_direct":
        return HUAIAN_G3_CURVE_FEATURE_COLUMNS, HUAIAN_G3_CURVE_CATEGORICAL_COLUMNS, False
    if name == "huaian_g5_minimal_directional":
        return HUAIAN_G5_CURVE_FEATURE_COLUMNS, HUAIAN_G5_CURVE_CATEGORICAL_COLUMNS, False
    if name == "huaian_g6_history":
        return HUAIAN_G6_CURVE_FEATURE_COLUMNS, HUAIAN_G6_CURVE_CATEGORICAL_COLUMNS, True
    if name == "huaian_latest_history_no_direction":
        return (
            HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_FEATURE_COLUMNS,
            HUAIAN_LATEST_HISTORY_NO_DIRECTION_CURVE_CATEGORICAL_COLUMNS,
            True,
        )
    if name == "huaian_latest_history_past_direction_only":
        return (
            HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_FEATURE_COLUMNS,
            HUAIAN_LATEST_HISTORY_PAST_DIRECTION_ONLY_CURVE_CATEGORICAL_COLUMNS,
            True,
        )
    raise ValueError(f"Unsupported curve feature set: {name}")


def main() -> None:
    args = parse_args()
    feature_columns_all, categorical_columns, include_history = _resolve_curve_feature_spec(args.curve_feature_set)
    split_frames, split_meta = load_huaian_local_curve_splits(
        args.split_config,
        wtpc_root=args.wtpc_root,
        rated_power_kw=float(args.rated_power_kw),
        include_history_features=include_history,
    )

    fit_frames = (
        [split_frames["train"]]
        if args.fit_split == "train"
        else [split_frames["train"], split_frames["val"]]
    )
    fit_frame = pd.concat(fit_frames, ignore_index=True)
    fit_frame = _maybe_sample_fit_frame(
        fit_frame,
        max_train_rows=int(args.max_train_rows),
        random_state=int(args.random_state),
    ).reset_index(drop=True)

    model, feature_columns = fit_curve_model(
        fit_frame,
        wtpc_root=args.wtpc_root,
        feature_columns=feature_columns_all,
        categorical_columns=categorical_columns,
        model_name=args.curve_model,
        random_state=int(args.random_state),
    )

    ws_predictions = load_ws_prediction_frames(args.ws_predictions)
    joined = merge_predictions_with_reference_frame(
        ws_predictions,
        {"train": split_frames["train"], "val": split_frames["val"], "test": split_frames["test"]},
        extra_columns=[col for col in feature_columns_all if col != "ws"],
    )

    inference_frame = joined.copy()
    inference_frame["ws"] = inference_frame["ws_pred"]
    power_pred_norm = predict_power_norm(
        model,
        inference_frame,
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
        wtpc_root=args.wtpc_root,
    )
    sanity_df = evaluate_curve_sanity(
        model,
        {"val": split_frames["val"], "test": split_frames["test"]},
        feature_columns=feature_columns,
        rated_power_kw=float(args.rated_power_kw),
        wtpc_root=args.wtpc_root,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "power_predictions.csv", index=False)
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    per_turbine_df.to_csv(output_dir / "per_turbine_metrics.csv", index=False)
    per_split_df.to_csv(output_dir / "per_split_metrics.csv", index=False)
    sanity_df.to_csv(output_dir / "curve_sanity_metrics.csv", index=False)

    summary = {
        "model_label": args.model_label,
        "ws_prediction_files": [str(Path(path).resolve()) for path in args.ws_predictions],
        "split_config_path": str(Path(args.split_config).resolve()),
        "wtpc_root": str(Path(args.wtpc_root).resolve()),
        "curve_model": args.curve_model,
        "curve_feature_set": args.curve_feature_set,
        "fit_split": args.fit_split,
        "max_train_rows": int(args.max_train_rows),
        "rated_power_kw": float(args.rated_power_kw),
        "random_state": int(args.random_state),
        "n_input_rows": int(len(ws_predictions)),
        "n_matched_rows": int(len(result)),
        "fit_rows_after_sampling": int(len(fit_frame)),
        "feature_columns": feature_columns,
        "feature_column_count": int(len(feature_columns)),
        "include_history_features": bool(include_history),
        "split_metadata": split_meta,
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
