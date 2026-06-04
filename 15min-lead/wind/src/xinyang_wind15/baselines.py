"""Baseline models for xinyang wind-speed forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA

from .features import feature_columns_for_lightgbm
from .metrics import regression_summary, skill_score_vs_baseline, summarize_predictions


@dataclass(frozen=True)
class BaselineResult:
    name: str
    split: str
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    per_turbine: pd.DataFrame | None = None


def _keep_scored_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["split"].isin(["val", "test"])
        & frame["y_true"].notna()
        & frame["y_pred"].notna()
    ].copy()


def persistence_baseline(frame: pd.DataFrame) -> list[BaselineResult]:
    baseline_frame = frame[["turbine_id", "target_timestamp", "split", "y_true", "ws_mean"]].copy()
    baseline_frame["y_pred"] = baseline_frame["ws_mean"]
    scored = _keep_scored_rows(baseline_frame)
    results: list[BaselineResult] = []
    for split_name, split_df in scored.groupby("split", sort=False):
        per_turbine, overall = summarize_predictions(split_df)
        results.append(
            BaselineResult(
                name="persistence",
                split=split_name,
                metrics=overall,
                predictions=split_df,
                per_turbine=per_turbine,
            )
        )
    return results


def seasonal_persistence_baseline(
    frame: pd.DataFrame,
    *,
    season_steps: int = 96,
    horizon_steps: int = 1,
) -> list[BaselineResult]:
    seasonal_lag = season_steps - horizon_steps
    baseline_frame = frame[["turbine_id", "target_timestamp", "split", "y_true", "ws_mean"]].copy()
    grouped = baseline_frame.groupby("turbine_id", sort=False)
    baseline_frame["y_pred"] = grouped["ws_mean"].shift(seasonal_lag)
    scored = _keep_scored_rows(baseline_frame)
    results: list[BaselineResult] = []
    for split_name, split_df in scored.groupby("split", sort=False):
        per_turbine, overall = summarize_predictions(split_df)
        results.append(
            BaselineResult(
                name="seasonal_persistence",
                split=split_name,
                metrics=overall,
                predictions=split_df,
                per_turbine=per_turbine,
            )
        )
    return results


def lightgbm_baseline(frame: pd.DataFrame) -> list[BaselineResult]:
    # LightGBM's sklearn API and early-stopping callback are documented here:
    # https://lightgbm.readthedocs.io/en/latest/Python-API.html
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.html
    feature_cols = feature_columns_for_lightgbm(frame)
    usable = frame.dropna(subset=["y_true"]).copy()
    train_df = usable.loc[usable["split"] == "train"].copy()
    val_df = usable.loc[usable["split"] == "val"].copy()
    test_df = usable.loc[usable["split"] == "test"].copy()

    split_counts = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    if min(split_counts.values()) == 0:
        raise ValueError(
            "LightGBM baseline requires non-empty train/val/test splits after "
            f"target construction. Current counts: {split_counts}."
        )

    na_free_cols = [
        col for col in feature_cols if not train_df[col].isna().all()
    ]
    feature_cols = na_free_cols
    if not feature_cols:
        raise ValueError("LightGBM baseline found no usable feature columns.")
    train_df = train_df.dropna(subset=feature_cols + ["y_true"]).copy()
    val_df = val_df.dropna(subset=feature_cols + ["y_true"]).copy()
    test_df = test_df.dropna(subset=feature_cols + ["y_true"]).copy()

    categorical_features = ["turbine_id"]
    for df in [train_df, val_df, test_df]:
        df["turbine_id"] = df["turbine_id"].astype("category")

    x_train = train_df[feature_cols]
    y_train = train_df["y_true"]
    x_val = val_df[feature_cols]
    y_val = val_df["y_true"]
    x_test = test_df[feature_cols]

    model = LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(50, first_metric_only=True, verbose=False),
        ],
    )

    val_pred = val_df[["turbine_id", "target_timestamp", "split", "y_true"]].copy()
    val_pred["y_pred"] = model.predict(x_val)
    test_pred = test_df[["turbine_id", "target_timestamp", "split", "y_true"]].copy()
    test_pred["y_pred"] = model.predict(x_test)

    results: list[BaselineResult] = []
    for split_name, split_df in [("val", val_pred), ("test", test_pred)]:
        per_turbine, overall = summarize_predictions(split_df)
        overall["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
        results.append(
            BaselineResult(
                name="lightgbm",
                split=split_name,
                metrics=overall,
                predictions=split_df,
                per_turbine=per_turbine,
            )
        )
    return results


def build_farm_mean_series(frame: pd.DataFrame) -> pd.Series:
    farm = (
        frame.groupby("timestamp", sort=True)["ws_mean"]
        .mean()
        .sort_index()
    )
    farm.index = pd.DatetimeIndex(farm.index)
    farm = farm.asfreq("15min")
    farm = farm.ffill().bfill()
    return farm


def _rolling_arima_forecast(
    fitted_result: Any,
    observations: pd.Series,
) -> pd.Series:
    preds = []
    result = fitted_result
    endog_name = getattr(result.model, "endog_names", "y")
    for timestamp, value in observations.items():
        forecast = result.forecast(steps=1)
        preds.append((timestamp, float(forecast.iloc[0])))
        observed = pd.DataFrame({endog_name: [value]}, index=[timestamp])
        result = result.append(observed, refit=False)
    return pd.Series(
        data=[value for _, value in preds],
        index=[timestamp for timestamp, _ in preds],
        name="y_pred",
    )


def sarima_farm_mean_baseline(
    farm_mean: pd.Series,
    *,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    train_tail_points: int = 1500,
    max_eval_points: int | None = None,
) -> list[BaselineResult]:
    # Statsmodels 0.14.6 documents ARIMA/SARIMA and append-based rolling updates:
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.append.html
    train = farm_mean.loc[:train_end].tail(train_tail_points)
    val = farm_mean.loc[(farm_mean.index > train_end) & (farm_mean.index <= val_end)]
    test = farm_mean.loc[farm_mean.index > val_end]
    if max_eval_points is not None:
        val = val.head(max_eval_points)
        test = test.head(max_eval_points)

    order_grid = [(1, 0, 0), (2, 0, 0)]
    seasonal_grid = [(0, 0, 0, 0)]

    best_cfg: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None
    best_val_rmse = float("inf")

    for order, seasonal_order in product(order_grid, seasonal_grid):
        try:
            model = ARIMA(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit()
            val_pred = _rolling_arima_forecast(result, val)
            val_metrics = regression_summary(val, val_pred)
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_cfg = (order, seasonal_order)
        except Exception:
            continue

    if best_cfg is None:
        raise RuntimeError("SARIMA search failed for all candidate configurations.")

    order, seasonal_order = best_cfg
    model_name = (
        "arima_farm_mean"
        if seasonal_order == (0, 0, 0, 0)
        else "sarima_farm_mean"
    )
    train_val = farm_mean.loc[:val_end].tail(train_tail_points)
    final_model = ARIMA(
        train_val,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_result = final_model.fit()
    val_fit_result = ARIMA(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    val_pred = _rolling_arima_forecast(val_fit_result, val)
    test_pred = _rolling_arima_forecast(final_result, test)

    val_frame = pd.DataFrame(
        {
            "series_id": "farm_mean",
            "target_timestamp": val.index,
            "split": "val",
            "y_true": val.values,
            "y_pred": val_pred.reindex(val.index).values,
        }
    )
    test_frame = pd.DataFrame(
        {
            "series_id": "farm_mean",
            "target_timestamp": test.index,
            "split": "test",
            "y_true": test.values,
            "y_pred": test_pred.reindex(test.index).values,
        }
    )

    results: list[BaselineResult] = []
    for split_name, split_df in [("val", val_frame), ("test", test_frame)]:
        metrics = regression_summary(split_df["y_true"], split_df["y_pred"])
        metrics["selected_order"] = list(order)
        metrics["selected_seasonal_order"] = list(seasonal_order)
        results.append(
            BaselineResult(
                name=model_name,
                split=split_name,
                metrics=metrics,
                predictions=split_df,
            )
        )
    return results


def attach_skill_scores(
    baseline_results: list[BaselineResult],
    *,
    persistence_results: list[BaselineResult],
) -> list[BaselineResult]:
    persistence_by_split = {
        result.split: result.metrics["rmse"] for result in persistence_results
    }
    enriched: list[BaselineResult] = []
    for result in baseline_results:
        metrics = dict(result.metrics)
        baseline_rmse = persistence_by_split.get(result.split)
        if baseline_rmse is not None and "rmse" in metrics:
            metrics["skill_score_vs_persistence"] = skill_score_vs_baseline(
                metrics["rmse"],
                baseline_rmse,
            )
        enriched.append(
            BaselineResult(
                name=result.name,
                split=result.split,
                metrics=metrics,
                predictions=result.predictions,
                per_turbine=result.per_turbine,
            )
        )
    return enriched
