"""Evaluation metrics."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = math.sqrt(float(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def skill_score_vs_baseline(
    rmse_model: float,
    rmse_baseline: float,
) -> float:
    if rmse_baseline == 0.0:
        return 0.0
    return 1.0 - (rmse_model / rmse_baseline)


def summarize_predictions(
    prediction_frame: pd.DataFrame,
    *,
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    group_col: str = "turbine_id",
) -> tuple[pd.DataFrame, dict[str, float]]:
    grouped_rows = []
    for turbine_id, group in prediction_frame.groupby(group_col, sort=True):
        metrics = regression_summary(group[truth_col], group[pred_col])
        grouped_rows.append({"turbine_id": turbine_id, **metrics, "count": len(group)})
    grouped_df = pd.DataFrame(grouped_rows)
    overall_point = regression_summary(
        prediction_frame[truth_col],
        prediction_frame[pred_col],
    )
    overall = {
        **overall_point,
        "mae_macro": float(grouped_df["mae"].mean()),
        "rmse_macro": float(grouped_df["rmse"].mean()),
        "r2_macro": float(grouped_df["r2"].mean()),
        "count": int(len(prediction_frame)),
    }
    return grouped_df, overall
