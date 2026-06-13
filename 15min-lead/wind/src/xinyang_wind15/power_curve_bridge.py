"""Bridge wind-speed forecasts to power forecasts with WTPC basic-feature MVPC."""

from __future__ import annotations

import importlib
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


DEFAULT_RATED_POWER_KW = 2500.0
DEFAULT_PREPARED_SUBDIR = Path("results") / "exp110_xinyang_split721_direct" / "prepared"
DEFAULT_WTPC_RESULT_SUBDIR = Path("results") / "exp114_xinyang_model_specific_direct_mvpc_formal_20260603"

STATIC_INFO_COLUMNS = [
    "turbine_id",
    "manufacturer",
    "hub_height",
    "rotor_diameter",
    "rated_capacity_mw",
    "cut_in_ws_ms",
    "rated_ws_ms",
    "cut_out_ws_ms",
]

BASIC_CURVE_FEATURE_COLUMNS = [
    "ws",
    "turbine_id",
    "hub_height",
    "rotor_diameter",
    "rated_capacity_mw",
    "cut_in_ws_ms",
    "rated_ws_ms",
    "cut_out_ws_ms",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "dist_to_cutin",
    "dist_to_rated",
    "ws_region",
]

BASIC_CURVE_CATEGORICAL_COLUMNS = [
    "turbine_id",
    "ws_region",
]


def default_wtpc_root() -> Path:
    return Path(__file__).resolve().parents[5] / "WTPC"


def ensure_wtpc_importable(wtpc_root: str | Path | None = None) -> Path:
    root = Path(wtpc_root) if wtpc_root is not None else default_wtpc_root()
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_wtpc_symbols(wtpc_root: str | Path | None = None) -> dict[str, Any]:
    ensure_wtpc_importable(wtpc_root)
    evaluate_mod = importlib.import_module("wtpc_dynamic.evaluate")
    models_mod = importlib.import_module("wtpc_dynamic.models")
    xinyang_mod = importlib.import_module("wtpc_dynamic.xinyang_e12_utils")
    return {
        "qualification_rate": evaluate_mod.qualification_rate,
        "rmse": evaluate_mod.rmse,
        "ModelSpec": models_mod.ModelSpec,
        "build_model": models_mod.build_model,
        "add_time_cyclic_features": xinyang_mod.add_time_cyclic_features,
        "add_operating_regime_features": xinyang_mod.add_operating_regime_features,
    }


def resolve_prepared_dir(
    *,
    wtpc_root: str | Path | None = None,
    prepared_dir: str | Path | None = None,
) -> Path:
    if prepared_dir is not None:
        return Path(prepared_dir).resolve()
    return (ensure_wtpc_importable(wtpc_root) / DEFAULT_PREPARED_SUBDIR).resolve()


def load_prepared_splits(prepared_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(prepared_dir)
    splits: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_path = root / f"{split_name}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing prepared split file: {split_path}")
        splits[split_name] = pd.read_parquet(split_path)
    return splits


def build_static_info_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("frames must not be empty when building the static info table.")
    merged = pd.concat(
        [frame.loc[:, [col for col in STATIC_INFO_COLUMNS if col in frame.columns]] for frame in frames],
        ignore_index=True,
    )
    merged["turbine_id"] = merged["turbine_id"].astype(str).str.strip()
    return merged.drop_duplicates(subset=["turbine_id"]).sort_values("turbine_id").reset_index(drop=True)


def load_ws_prediction_frame(path: str | Path) -> pd.DataFrame:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Missing wind-speed prediction file: {src}")
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    elif src.suffix.lower() == ".parquet":
        df = pd.read_parquet(src)
    else:
        raise ValueError(f"Unsupported prediction file format: {src.suffix}")

    cols = set(df.columns)
    time_col = "target_timestamp" if "target_timestamp" in cols else "time" if "time" in cols else None
    pred_col = "y_pred" if "y_pred" in cols else "ws_pred" if "ws_pred" in cols else "pred" if "pred" in cols else None
    if time_col is None or pred_col is None or "turbine_id" not in cols:
        raise ValueError(
            "Prediction file must contain turbine_id plus a time column "
            "(target_timestamp or time) and a prediction column (y_pred, ws_pred, or pred)."
        )

    out = df.copy()
    out["turbine_id"] = out["turbine_id"].astype(str).str.strip()
    out["target_timestamp"] = pd.to_datetime(out[time_col], errors="coerce")
    out["ws_pred"] = pd.to_numeric(out[pred_col], errors="coerce")
    if "y_true" in out.columns:
        out["ws_model_truth"] = pd.to_numeric(out["y_true"], errors="coerce")
    if "origin_timestamp" in out.columns:
        out["origin_timestamp"] = pd.to_datetime(out["origin_timestamp"], errors="coerce")
    keep_cols = [
        col
        for col in [
            "turbine_id",
            "origin_timestamp",
            "target_timestamp",
            "split",
            "ws_model_truth",
            "ws_pred",
        ]
        if col in out.columns
    ]
    out = out.loc[:, keep_cols].copy()
    out = out.dropna(subset=["turbine_id", "target_timestamp", "ws_pred"]).reset_index(drop=True)
    return out


def merge_predictions_with_actuals(
    ws_predictions: pd.DataFrame,
    actual_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for split_name, frame in actual_frames.items():
        block = frame.copy()
        block["actual_split"] = split_name
        pieces.append(block)
    actual = pd.concat(pieces, ignore_index=True)
    actual["turbine_id"] = actual["turbine_id"].astype(str).str.strip()
    actual["time"] = pd.to_datetime(actual["time"], errors="coerce")

    keep_actual = [
        col
        for col in [
            "turbine_id",
            "time",
            "actual_split",
            "ws",
            "power",
            "power_norm",
        ]
        if col in actual.columns
    ]
    actual = actual.loc[:, keep_actual].copy()
    actual = actual.rename(
        columns={
            "ws": "ws_actual",
            "power": "power_true_kw",
            "power_norm": "power_true_norm",
        }
    )

    out = ws_predictions.merge(
        actual,
        left_on=["turbine_id", "target_timestamp"],
        right_on=["turbine_id", "time"],
        how="left",
        validate="one_to_one",
    )
    out = out.drop(columns=["time"])
    missing = out["power_true_norm"].isna().sum()
    if missing > 0:
        raise ValueError(
            "Some predicted rows could not be matched to WTPC prepared actuals. "
            f"Missing rows: {int(missing)} / {len(out)}."
        )
    return out


def build_basic_feature_frame(
    frame: pd.DataFrame,
    *,
    static_info: pd.DataFrame,
    wtpc_root: str | Path | None = None,
    ws_col: str = "ws",
    time_col: str = "time",
) -> pd.DataFrame:
    symbols = _load_wtpc_symbols(wtpc_root)
    out = frame.copy()
    out["turbine_id"] = out["turbine_id"].astype(str).str.strip()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[ws_col] = pd.to_numeric(out[ws_col], errors="coerce")
    out = out.merge(static_info, on="turbine_id", how="left", validate="many_to_one")
    out = symbols["add_time_cyclic_features"](out, time_col=time_col)
    out = symbols["add_operating_regime_features"](out, ws_col=ws_col)
    return out


def basic_curve_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in BASIC_CURVE_FEATURE_COLUMNS if col in frame.columns]


def basic_curve_categorical_columns(feature_columns: list[str]) -> list[str]:
    return [col for col in BASIC_CURVE_CATEGORICAL_COLUMNS if col in feature_columns]


def default_curve_model_params(model_name: str) -> dict[str, Any]:
    name = model_name.lower()
    if name == "lgbm":
        return {
            "n_estimators": 320,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "num_leaves": 31,
            "min_child_samples": 20,
            "n_jobs": -1,
        }
    if name == "hgb":
        return {
            "learning_rate": 0.05,
            "max_iter": 320,
            "max_depth": 8,
            "min_samples_leaf": 20,
        }
    if name == "ridge":
        return {
            "alpha": 2.0,
        }
    raise ValueError(f"Unsupported basic curve model: {model_name}")


def fit_basic_curve_model(
    train_frame: pd.DataFrame,
    *,
    wtpc_root: str | Path | None = None,
    model_name: str = "lgbm",
    random_state: int = 42,
) -> tuple[Any, list[str]]:
    symbols = _load_wtpc_symbols(wtpc_root)
    feature_columns = basic_curve_feature_columns(train_frame)
    categorical_columns = basic_curve_categorical_columns(feature_columns)
    spec = symbols["ModelSpec"](model_name.lower(), default_curve_model_params(model_name))
    model = symbols["build_model"](
        spec,
        feature_cols=feature_columns,
        categorical_cols=categorical_columns,
        random_state=int(random_state),
    )
    model.fit(
        train_frame[feature_columns],
        train_frame["power_norm"].to_numpy(dtype=float),
    )
    return model, feature_columns


def predict_power_norm(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> np.ndarray:
    pred = np.asarray(model.predict(frame[feature_columns]), dtype=float)
    return np.clip(pred, 0.0, 1.2)


def compute_power_metrics(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    *,
    rated_power_kw: float,
    qualification_rate_fn: Any | None = None,
    rmse_fn: Any | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true_kw, dtype=float)
    y_pred = np.asarray(y_pred_kw, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {
            "n_points": 0,
            "rmse_kw": math.nan,
            "nrmse": math.nan,
            "mae_kw": math.nan,
            "nmae": math.nan,
            "bias_kw": math.nan,
            "q_mean": math.nan,
            "qualification_rate": math.nan,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if rmse_fn is None:
        rmse_kw = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    else:
        rmse_kw = float(rmse_fn(y_true, y_pred))
    mae_kw = float(np.mean(np.abs(y_true - y_pred)))
    bias_kw = float(np.mean(y_pred - y_true))
    q_mean = (
        float(np.mean((1.0 - np.abs(y_true - y_pred) / float(rated_power_kw)) * 100.0))
        if qualification_rate_fn is None
        else float(qualification_rate_fn(y_true, y_pred, rated_power=float(rated_power_kw)))
    )
    return {
        "n_points": int(mask.sum()),
        "rmse_kw": rmse_kw,
        "nrmse": rmse_kw / float(rated_power_kw),
        "mae_kw": mae_kw,
        "nmae": mae_kw / float(rated_power_kw),
        "bias_kw": bias_kw,
        "q_mean": q_mean,
        "qualification_rate": q_mean,
    }


def summarize_power_predictions(
    frame: pd.DataFrame,
    *,
    rated_power_kw: float,
    wtpc_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbols = _load_wtpc_symbols(wtpc_root)
    overall_rows: list[dict[str, Any]] = []
    per_turbine_rows: list[dict[str, Any]] = []
    per_split_rows: list[dict[str, Any]] = []

    overall_rows.append(
        {
            "scope": "overall",
            **compute_power_metrics(
                frame["power_true_kw"].to_numpy(dtype=float),
                frame["power_pred_kw"].to_numpy(dtype=float),
                rated_power_kw=rated_power_kw,
                qualification_rate_fn=symbols["qualification_rate"],
                rmse_fn=symbols["rmse"],
            ),
        }
    )

    for split_name, group in frame.groupby("actual_split", sort=True):
        per_split_rows.append(
            {
                "split": str(split_name),
                **compute_power_metrics(
                    group["power_true_kw"].to_numpy(dtype=float),
                    group["power_pred_kw"].to_numpy(dtype=float),
                    rated_power_kw=rated_power_kw,
                    qualification_rate_fn=symbols["qualification_rate"],
                    rmse_fn=symbols["rmse"],
                ),
            }
        )

    for turbine_id, group in frame.groupby("turbine_id", sort=True):
        per_turbine_rows.append(
            {
                "turbine_id": str(turbine_id),
                **compute_power_metrics(
                    group["power_true_kw"].to_numpy(dtype=float),
                    group["power_pred_kw"].to_numpy(dtype=float),
                    rated_power_kw=rated_power_kw,
                    qualification_rate_fn=symbols["qualification_rate"],
                    rmse_fn=symbols["rmse"],
                ),
            }
        )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(per_turbine_rows).sort_values("turbine_id").reset_index(drop=True),
        pd.DataFrame(per_split_rows).sort_values("split").reset_index(drop=True),
    )


def evaluate_curve_sanity(
    model: Any,
    split_frames: dict[str, pd.DataFrame],
    *,
    feature_columns: list[str],
    rated_power_kw: float,
    wtpc_root: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in split_frames.items():
        pred_norm = predict_power_norm(model, frame, feature_columns=feature_columns)
        pred_kw = pred_norm * float(rated_power_kw)
        true_kw = frame["power"].to_numpy(dtype=float)
        metrics = compute_power_metrics(
            true_kw,
            pred_kw,
            rated_power_kw=rated_power_kw,
            qualification_rate_fn=_load_wtpc_symbols(wtpc_root)["qualification_rate"],
            rmse_fn=_load_wtpc_symbols(wtpc_root)["rmse"],
        )
        rows.append({"split": split_name, **metrics})
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
