"""Train a seq2one LSTM / GRU / TFT-style model from a disk-backed store."""

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

from xinyang_wind15.gru import MultiTurbineGRU  # noqa: E402
from xinyang_wind15.lstm import MultiTurbineLSTM  # noqa: E402
from xinyang_wind15.sequence import (  # noqa: E402
    WindowStoreDataset,
    compute_standardization_stats_from_store,
    evaluate_window_model,
    make_loader,
    save_checkpoint,
)
from xinyang_wind15.tft import TemporalFusionTransformerLite  # noqa: E402
from xinyang_wind15.window_store import load_window_store  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument(
        "--arch",
        choices=["lstm", "gru", "tft"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "seq_model_from_store"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--stats-chunk-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    *,
    arch: str,
    n_turbines: int,
    n_features: int,
    lookback_steps: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_attention_heads: int,
) -> nn.Module:
    if arch == "gru":
        return MultiTurbineGRU(
            n_turbines=n_turbines,
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if arch == "lstm":
        return MultiTurbineLSTM(
            n_turbines=n_turbines,
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    return TemporalFusionTransformerLite(
        n_turbines=n_turbines,
        n_features=n_features,
        lookback_steps=lookback_steps,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_encoder_layers=num_layers,
        dropout=dropout,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store_dir = Path(args.store_dir)
    store = load_window_store(store_dir, mmap_mode="r")
    metadata = store["metadata"]
    feature_tensor = store["feature_tensor"]
    target_matrix = store["target_matrix"]
    target_mask = store["target_mask"]
    origin_indices = np.asarray(store["origin_indices"], dtype=np.int64)
    split_labels = np.asarray(store["split_labels"])
    timestamps = pd.to_datetime(np.asarray(store["timestamps"]))
    lookback_steps = int(metadata["lookback_steps"])
    horizon_steps = int(metadata["horizon_steps"])
    feature_columns = list(metadata["feature_columns"])
    turbine_order = list(metadata["turbine_order"])

    train_origins = origin_indices[split_labels == "train"]
    val_origins = origin_indices[split_labels == "val"]
    test_origins = origin_indices[split_labels == "test"]
    if len(train_origins) == 0 or len(val_origins) == 0 or len(test_origins) == 0:
        raise ValueError(
            f"{args.arch.upper()} store training requires non-empty train/val/test windows. "
            f"Counts: train={len(train_origins)}, val={len(val_origins)}, test={len(test_origins)}."
        )

    stats = compute_standardization_stats_from_store(
        feature_tensor,
        target_matrix,
        train_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        target_mask=target_mask,
        chunk_size=args.stats_chunk_size,
    )
    train_dataset = WindowStoreDataset(
        feature_tensor,
        target_matrix,
        train_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        target_mask=target_mask,
        stats=stats,
    )
    val_dataset = WindowStoreDataset(
        feature_tensor,
        target_matrix,
        val_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        target_mask=target_mask,
        stats=stats,
    )
    test_dataset = WindowStoreDataset(
        feature_tensor,
        target_matrix,
        test_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        target_mask=target_mask,
        stats=stats,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        arch=args.arch,
        n_turbines=len(turbine_order),
        n_features=len(feature_columns),
        lookback_steps=lookback_steps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_attention_heads=args.num_attention_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.HuberLoss()

    best_val_rmse = float("inf")
    best_state = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch, y_mask_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_mask_batch = y_mask_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            observed = y_mask_batch
            if not observed.any():
                continue
            loss = loss_fn(pred[observed], y_batch[observed])
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
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_rmse": float(val_metrics["rmse"]),
            }
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"{args.arch.upper()} store training finished without a valid checkpoint.")
    model.load_state_dict(best_state)

    val_target_times = timestamps[val_origins + horizon_steps]
    test_target_times = timestamps[test_origins + horizon_steps]
    val_origin_times = timestamps[val_origins]
    test_origin_times = timestamps[test_origins]
    val_metrics, val_per_turbine, val_predictions = evaluate_window_model(
        model,
        val_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="val",
        target_timestamps=val_target_times,
        origin_timestamps=val_origin_times,
        return_predictions=True,
    )
    test_metrics, test_per_turbine, test_predictions = evaluate_window_model(
        model,
        test_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="test",
        target_timestamps=test_target_times,
        origin_timestamps=test_origin_times,
        return_predictions=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output_dir / f"{args.arch}_from_store.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    val_per_turbine.to_csv(output_dir / "val_per_turbine.csv", index=False)
    test_per_turbine.to_csv(output_dir / "test_per_turbine.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps([val_metrics, test_metrics], indent=2),
        encoding="utf-8",
    )
    summary = {
        "arch": args.arch,
        "store_dir": str(store_dir),
        "lookback_steps": lookback_steps,
        "horizon_steps": horizon_steps,
        "feature_columns": feature_columns,
        "turbine_order": turbine_order,
        "train_windows": int(len(train_origins)),
        "val_windows": int(len(val_origins)),
        "test_windows": int(len(test_origins)),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_attention_heads": args.num_attention_heads,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
