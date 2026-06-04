"""Train a Graph WaveNet style model from a disk-backed window store."""

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

from xinyang_wind15.graph import build_graph_wavenet_supports, supports_to_torch  # noqa: E402
from xinyang_wind15.gwnet import GraphWaveNetLite  # noqa: E402
from xinyang_wind15.sequence import (  # noqa: E402
    WindowStoreDataset,
    compute_standardization_stats_from_store,
    evaluate_window_model,
    make_loader,
    save_checkpoint,
)
from xinyang_wind15.window_store import load_window_store  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "artifacts" / "local_debug" / "gwnet_store_train"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--residual-channels", type=int, default=32)
    parser.add_argument("--dilation-channels", type=int, default=32)
    parser.add_argument("--skip-channels", type=int, default=128)
    parser.add_argument("--end-channels", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--graph-order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--stats-chunk-size",
        type=int,
        default=32,
        help="Number of origin indices processed per stats chunk.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers. Keep 0 on Windows unless explicitly needed.",
    )
    parser.add_argument(
        "--support-mode",
        choices=["auto", "distance", "distance_correlation"],
        default="auto",
        help="Which fixed graph supports to load from the store directory.",
    )
    parser.add_argument(
        "--disable-adaptive-adj",
        action="store_true",
        help="Use only the fixed supports saved with the store.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    store_dir = Path(args.store_dir)
    store = load_window_store(store_dir, mmap_mode="r")
    metadata = store["metadata"]
    feature_tensor = store["feature_tensor"]
    target_matrix = store["target_matrix"]
    origin_indices = np.asarray(store["origin_indices"], dtype=np.int64)
    split_labels = np.asarray(store["split_labels"])
    lookback_steps = int(metadata["lookback_steps"])
    horizon_steps = int(metadata["horizon_steps"])
    feature_columns = list(metadata["feature_columns"])
    turbine_order = list(metadata["turbine_order"])

    distance_adjacency_path = store_dir / "distance_adjacency.npy"
    if not distance_adjacency_path.exists():
        raise FileNotFoundError(
            f"Expected adjacency file at {distance_adjacency_path}. Build the store first."
        )
    correlation_adjacency_path = store_dir / "correlation_adjacency.npy"
    adjacency_list = [np.load(distance_adjacency_path)]
    if args.support_mode == "distance_correlation":
        if not correlation_adjacency_path.exists():
            raise FileNotFoundError(
                f"Expected correlation adjacency at {correlation_adjacency_path}."
            )
        adjacency_list.append(np.load(correlation_adjacency_path))
    elif args.support_mode == "auto" and correlation_adjacency_path.exists():
        adjacency_list.append(np.load(correlation_adjacency_path))
    support_arrays = build_graph_wavenet_supports(adjacency_list)
    supports = supports_to_torch(support_arrays, device=device)

    train_origins = origin_indices[split_labels == "train"]
    val_origins = origin_indices[split_labels == "val"]
    test_origins = origin_indices[split_labels == "test"]
    if len(train_origins) == 0 or len(val_origins) == 0 or len(test_origins) == 0:
        raise ValueError(
            "GraphWaveNet store training requires non-empty train/val/test windows. "
            f"Counts: train={len(train_origins)}, val={len(val_origins)}, test={len(test_origins)}."
        )

    stats = compute_standardization_stats_from_store(
        feature_tensor,
        target_matrix,
        train_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        chunk_size=args.stats_chunk_size,
    )
    train_dataset = WindowStoreDataset(
        feature_tensor,
        target_matrix,
        train_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        stats=stats,
    )
    val_dataset = WindowStoreDataset(
        feature_tensor,
        target_matrix,
        val_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        stats=stats,
    )
    test_dataset = WindowStoreDataset(
        feature_tensor,
        target_matrix,
        test_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
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

    model = GraphWaveNetLite(
        num_nodes=len(turbine_order),
        in_dim=len(feature_columns),
        supports=supports,
        dropout=args.dropout,
        gcn_bool=True,
        addaptadj=not args.disable_adaptive_adj,
        residual_channels=args.residual_channels,
        dilation_channels=args.dilation_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        kernel_size=args.kernel_size,
        blocks=args.blocks,
        layers=args.layers,
        graph_order=args.graph_order,
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("GraphWaveNet store training finished without a valid checkpoint.")
    model.load_state_dict(best_state)

    val_metrics, val_per_turbine = evaluate_window_model(
        model,
        val_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="val",
    )
    test_metrics, test_per_turbine = evaluate_window_model(
        model,
        test_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="test",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output_dir / "gwnet_baseline.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    val_per_turbine.to_csv(output_dir / "val_per_turbine.csv", index=False)
    test_per_turbine.to_csv(output_dir / "test_per_turbine.csv", index=False)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps([val_metrics, test_metrics], indent=2),
        encoding="utf-8",
    )
    summary = {
        "store_dir": str(store_dir),
        "x_train_windows": int(len(train_dataset)),
        "x_val_windows": int(len(val_dataset)),
        "x_test_windows": int(len(test_dataset)),
        "lookback_steps": lookback_steps,
        "horizon_steps": horizon_steps,
        "feature_columns": feature_columns,
        "turbine_order": turbine_order,
        "supports": len(supports),
        "support_mode": args.support_mode,
        "adaptive_adj": not args.disable_adaptive_adj,
        "residual_channels": int(args.residual_channels),
        "dilation_channels": int(args.dilation_channels),
        "skip_channels": int(args.skip_channels),
        "end_channels": int(args.end_channels),
        "kernel_size": int(args.kernel_size),
        "blocks": int(args.blocks),
        "layers": int(args.layers),
        "graph_order": int(args.graph_order),
        "num_workers": int(args.num_workers),
        "stats_chunk_size": int(args.stats_chunk_size),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
