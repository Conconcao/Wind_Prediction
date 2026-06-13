"""Train a ModernTCN-style model from a disk-backed hub-ws store."""

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

from xinyang_wind15.modern_tcn import ModernTCNForecaster  # noqa: E402
from xinyang_wind15.sequence import (  # noqa: E402
    WindowStoreDataset,
    compute_standardization_stats_from_store,
    evaluate_window_model,
    make_loader,
    masked_mean_loss,
    save_checkpoint,
)
from xinyang_wind15.window_store import load_window_store  # noqa: E402


class ModernTCNStoreAdapter(nn.Module):
    """Adapt [batch, steps, nodes, 1] store windows to ModernTCNForecaster."""

    def __init__(self, base_model: ModernTCNForecaster) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 1:
            raise ValueError(
                "ModernTCNStoreAdapter currently expects hub_ws_only windows with a single feature."
            )
        return self.base_model(x[..., 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "moderntcn_store_train"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ffn-ratio", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--patch-stride", type=int, default=2)
    parser.add_argument("--large-kernel", type=int, default=13)
    parser.add_argument("--small-kernel", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = load_window_store(args.store_dir, mmap_mode="r")
    metadata = store["metadata"]
    feature_tensor = store["feature_tensor"]
    target_matrix = store["target_matrix"]
    target_mask = store["target_mask"]
    origin_indices = np.asarray(store["origin_indices"], dtype=np.int64)
    target_indices = np.asarray(store["target_indices"], dtype=np.int64)
    split_labels = np.asarray(store["split_labels"])
    timestamps = pd.to_datetime(np.asarray(store["timestamps"]))
    lookback_steps = int(metadata["lookback_steps"])
    horizon_steps = int(metadata["horizon_steps"])
    feature_columns = list(metadata["feature_columns"])
    turbine_order = list(metadata["turbine_order"])
    if len(feature_columns) != 1:
        raise ValueError(
            "ModernTCN store training currently expects a hub_ws_only store with one feature. "
            f"Current features: {feature_columns}"
        )

    train_origins = origin_indices[split_labels == "train"]
    val_origins = origin_indices[split_labels == "val"]
    test_origins = origin_indices[split_labels == "test"]
    if len(train_origins) == 0 or len(val_origins) == 0 or len(test_origins) == 0:
        raise ValueError(
            "ModernTCN store training requires non-empty train/val/test windows. "
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
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ModernTCNStoreAdapter(
        ModernTCNForecaster(
            n_vars=len(turbine_order),
            seq_len=lookback_steps,
            hidden_dim=args.hidden_dim,
            ffn_ratio=args.ffn_ratio,
            num_blocks=args.num_blocks,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            large_kernel=args.large_kernel,
            small_kernel=args.small_kernel if args.small_kernel > 0 else None,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.HuberLoss(reduction="none")

    best_val_rmse = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch, y_mask_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_mask_batch = y_mask_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = masked_mean_loss(loss_fn(pred, y_batch), y_mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("ModernTCN store training finished without a valid checkpoint.")
    model.load_state_dict(best_state)

    val_metrics, val_per_turbine, val_predictions = evaluate_window_model(
        model,
        val_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="val",
        target_timestamps=timestamps[target_indices[split_labels == "val"]],
        origin_timestamps=timestamps[val_origins],
        return_predictions=True,
    )
    test_metrics, test_per_turbine, test_predictions = evaluate_window_model(
        model,
        test_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="test",
        target_timestamps=timestamps[target_indices[split_labels == "test"]],
        origin_timestamps=timestamps[test_origins],
        return_predictions=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output_dir / "moderntcn_baseline.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    val_per_turbine.to_csv(output_dir / "val_per_turbine.csv", index=False)
    test_per_turbine.to_csv(output_dir / "test_per_turbine.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps([val_metrics, test_metrics], indent=2), encoding="utf-8")
    summary = {
        "store_dir": str(Path(args.store_dir)),
        "x_train_windows": int(len(train_dataset)),
        "x_val_windows": int(len(val_dataset)),
        "x_test_windows": int(len(test_dataset)),
        "lookback_steps": lookback_steps,
        "horizon_steps": horizon_steps,
        "feature_columns": feature_columns,
        "turbine_order": turbine_order,
        "min_target_coverage": float(metadata.get("min_target_coverage", 1.0)),
        "min_target_count": int(metadata.get("min_target_count", len(turbine_order))),
        "hidden_dim": int(args.hidden_dim),
        "ffn_ratio": int(args.ffn_ratio),
        "num_blocks": int(args.num_blocks),
        "patch_size": int(args.patch_size),
        "patch_stride": int(args.patch_stride),
        "large_kernel": int(args.large_kernel),
        "small_kernel": int(args.small_kernel),
        "num_workers": int(args.num_workers),
        "stats_chunk_size": int(args.stats_chunk_size),
        "device": str(device),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
