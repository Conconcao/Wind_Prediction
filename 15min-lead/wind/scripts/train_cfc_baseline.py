"""Train a local CfC baseline on xinyang 15-minute wind-speed windows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xinyang_wind15.cfc import (  # noqa: E402
    MultiTurbineCfC,
    compute_standardization_stats,
    make_loader,
    save_checkpoint,
    standardize_arrays,
)
from xinyang_wind15.features import build_timestep_feature_frame  # noqa: E402
from xinyang_wind15.loading import (  # noqa: E402
    build_scada_1min_aggregates,
    build_tower_met_wide,
    load_scada_15min,
    load_scada_1min,
)
from xinyang_wind15.sequence import evaluate_window_model  # noqa: E402
from xinyang_wind15.settings import default_split_config_path  # noqa: E402
from xinyang_wind15.settings import load_settings  # noqa: E402
from xinyang_wind15.windows import (  # noqa: E402
    build_spatiotemporal_windows,
    estimate_dense_window_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-config",
        default=str(default_split_config_path(PROJECT_DIR)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "cfc_smoke"),
    )
    parser.add_argument("--lookback-steps", type=int, default=32)
    parser.add_argument(
        "--max-dense-window-gib",
        type=float,
        default=4.0,
        help="Abort if estimated dense window tensors would exceed this size in GiB.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=["ws_mean", "power_mean", "wd_mean", "nacelle_mean", "ws_std"],
    )
    parser.add_argument("--include-tower", action="store_true")
    parser.add_argument("--include-1min", action="store_true")
    parser.add_argument(
        "--turbine-id",
        default=None,
        help="Optional single-turbine experiment id, for example S29.",
    )
    parser.add_argument("--max-turbines", type=int, default=4)
    parser.add_argument("--tail-timestamps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--mode",
        choices=("default", "pure", "no_gate"),
        default="default",
        help="CfC transition mode from the official ncps implementation.",
    )
    parser.add_argument(
        "--mixed-memory",
        action="store_true",
        help="Enable the ncps mixed-memory augmentation on top of CfC.",
    )
    parser.add_argument("--backbone-units", type=int, default=128)
    parser.add_argument("--backbone-layers", type=int, default=1)
    parser.add_argument("--backbone-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = load_settings(args.split_config)
    scada = load_scada_15min(
        settings.data_paths["scada_15min"],
        max_turbines=args.max_turbines,
        turbine_ids=[args.turbine_id] if args.turbine_id else None,
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
            turbine_ids=[args.turbine_id] if args.turbine_id else None,
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
    estimated_dense_bytes = estimate_dense_window_bytes(
        n_timestamps=int(feature_frame["timestamp"].nunique()),
        n_turbines=int(feature_frame["turbine_id"].nunique()),
        n_features=len(feature_columns),
        lookback_steps=args.lookback_steps,
        horizon_steps=settings.horizon_steps,
    )
    estimated_dense_gib = estimated_dense_bytes / (1024**3)
    if estimated_dense_gib > args.max_dense_window_gib:
        raise ValueError(
            "Estimated dense window materialization is too large for the current "
            f"local script: about {estimated_dense_gib:.2f} GiB, above the "
            f"configured limit {args.max_dense_window_gib:.2f} GiB. "
            "Use a smaller debug subset or switch to a future streaming/full-farm path."
        )
    bundle = build_spatiotemporal_windows(
        feature_frame,
        feature_columns=feature_columns,
        target_column="ws_mean",
        lookback_steps=args.lookback_steps,
        horizon_steps=settings.horizon_steps,
        split_bounds=settings.split_bounds,
    )

    x = bundle["x"]
    y = bundle["y"]
    split = bundle["split"]
    turbine_order = list(bundle["turbine_order"])

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "CfC training requires non-empty train/val/test windows. "
            f"Counts: train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
            f"test={int(test_mask.sum())}."
        )

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    stats = compute_standardization_stats(x_train, y_train)
    x_train_s, y_train_s = standardize_arrays(x_train, y_train, stats)
    x_val_s, y_val_s = standardize_arrays(x_val, y_val, stats)
    x_test_s, y_test_s = standardize_arrays(x_test, y_test, stats)

    train_loader = make_loader(x_train_s, y_train_s, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(x_val_s, y_val_s, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(x_test_s, y_test_s, batch_size=args.batch_size, shuffle=False)

    model = MultiTurbineCfC(
        n_turbines=len(turbine_order),
        n_features=len(feature_columns),
        hidden_size=args.hidden_size,
        mixed_memory=args.mixed_memory,
        mode=args.mode,
        backbone_units=args.backbone_units,
        backbone_layers=args.backbone_layers,
        backbone_dropout=args.backbone_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.HuberLoss()

    best_val_rmse = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics, _ = evaluate_window_model(
            model,
            val_loader,
            device=device,
            stats=stats,
            turbine_order=turbine_order,
            split_name="val",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_rmse": float(val_metrics["rmse"]),
            }
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("CfC training finished without a valid checkpoint.")
    model.load_state_dict(best_state)

    val_metrics, val_per_turbine, val_predictions = evaluate_window_model(
        model,
        val_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="val",
        target_timestamps=bundle["target_times"][val_mask],
        origin_timestamps=bundle["origin_times"][val_mask],
        return_predictions=True,
    )
    test_metrics, test_per_turbine, test_predictions = evaluate_window_model(
        model,
        test_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="test",
        target_timestamps=bundle["target_times"][test_mask],
        origin_timestamps=bundle["origin_times"][test_mask],
        return_predictions=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output_dir / "cfc_baseline.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    val_per_turbine.to_csv(output_dir / "val_per_turbine.csv", index=False)
    test_per_turbine.to_csv(output_dir / "test_per_turbine.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps([val_metrics, test_metrics], indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_name": settings.experiment_name,
        "x_train_shape": list(x_train.shape),
        "x_val_shape": list(x_val.shape),
        "x_test_shape": list(x_test.shape),
        "estimated_dense_window_gib": estimated_dense_gib,
        "feature_columns": feature_columns,
        "turbine_order": turbine_order,
        "hidden_size": int(args.hidden_size),
        "mode": args.mode,
        "mixed_memory": bool(args.mixed_memory),
        "backbone_units": int(args.backbone_units),
        "backbone_layers": int(args.backbone_layers),
        "backbone_dropout": float(args.backbone_dropout),
        "include_tower": bool(args.include_tower),
        "include_1min": bool(args.include_1min),
        "turbine_id": args.turbine_id,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
