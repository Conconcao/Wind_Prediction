"""Export val/test predictions from an existing Graph WaveNet run directory."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xinyang_wind15.graph import (  # noqa: E402
    build_graph_wavenet_supports,
    supports_to_torch,
)
from xinyang_wind15.gwnet import GraphWaveNetLite  # noqa: E402
from xinyang_wind15.sequence import (  # noqa: E402
    WindowStoreDataset,
    compute_standardization_stats_from_store,
    evaluate_window_model,
    make_loader,
)
from xinyang_wind15.window_store import load_window_store  # noqa: E402
from train_gwnet_from_store import _make_directional_support_builder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--store-dir", default="")
    parser.add_argument("--checkpoint-name", default="gwnet_baseline.pt")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dir = Path(args.train_dir)
    summary_path = train_dir / args.summary_name
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    store_dir = Path(args.store_dir) if args.store_dir else Path(summary["store_dir"])
    checkpoint_path = train_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    store = load_window_store(store_dir, mmap_mode="r")
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

    train_origins = origin_indices[split_labels == "train"]
    val_origins = origin_indices[split_labels == "val"]
    test_origins = origin_indices[split_labels == "test"]
    if len(train_origins) == 0 or len(val_origins) == 0 or len(test_origins) == 0:
        raise ValueError(
            "Prediction export requires non-empty train/val/test windows. "
            f"Counts: train={len(train_origins)}, val={len(val_origins)}, test={len(test_origins)}."
        )

    distance_adjacency_path = store_dir / "distance_adjacency.npy"
    if not distance_adjacency_path.exists():
        raise FileNotFoundError(f"Missing adjacency file: {distance_adjacency_path}")
    correlation_adjacency_path = store_dir / "correlation_adjacency.npy"
    distance_adjacency = np.load(distance_adjacency_path)
    adjacency_list = [distance_adjacency]
    support_mode = str(summary.get("support_mode", "auto"))
    if support_mode == "distance_correlation":
        if not correlation_adjacency_path.exists():
            raise FileNotFoundError(f"Expected correlation adjacency at {correlation_adjacency_path}")
        adjacency_list.append(np.load(correlation_adjacency_path))
    elif support_mode == "auto" and correlation_adjacency_path.exists():
        adjacency_list.append(np.load(correlation_adjacency_path))
    support_arrays = build_graph_wavenet_supports(adjacency_list)
    supports = supports_to_torch(support_arrays, device=device)

    stats_chunk_size = int(args.stats_chunk_size or summary.get("stats_chunk_size", 32))
    stats = compute_standardization_stats_from_store(
        feature_tensor,
        target_matrix,
        train_origins,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        target_mask=target_mask,
        chunk_size=stats_chunk_size,
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
    num_workers = int(args.num_workers if args.num_workers is not None else summary.get("num_workers", 0))
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = GraphWaveNetLite(
        num_nodes=len(turbine_order),
        in_dim=len(feature_columns),
        supports=supports,
        dropout=0.0,
        gcn_bool=True,
        addaptadj=bool(summary.get("adaptive_adj", True)),
        residual_channels=int(summary["residual_channels"]),
        dilation_channels=int(summary["dilation_channels"]),
        skip_channels=int(summary["skip_channels"]),
        end_channels=int(summary["end_channels"]),
        kernel_size=int(summary["kernel_size"]),
        blocks=int(summary["blocks"]),
        layers=int(summary["layers"]),
        graph_order=int(summary["graph_order"]),
        extra_support_len=2 if bool(summary.get("dynamic_directional_support", False)) else 0,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    support_builder = None
    if bool(summary.get("dynamic_directional_support", False)):
        support_builder = _make_directional_support_builder(
            feature_columns=feature_columns,
            stats=stats,
            store_dir=store_dir,
            distance_adjacency=distance_adjacency,
            device=device,
            direction_support_source=str(summary.get("direction_support_source", "auto")),
            sigma_deg=float(summary.get("direction_support_sigma_deg", 35.0)),
        )

    val_metrics, val_per_turbine, val_predictions = evaluate_window_model(
        model,
        val_loader,
        device=device,
        stats=stats,
        turbine_order=turbine_order,
        split_name="val",
        supports_builder=support_builder,
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
        supports_builder=support_builder,
        target_timestamps=timestamps[target_indices[split_labels == "test"]],
        origin_timestamps=timestamps[test_origins],
        return_predictions=True,
    )

    val_per_turbine.to_csv(train_dir / "val_per_turbine.csv", index=False)
    test_per_turbine.to_csv(train_dir / "test_per_turbine.csv", index=False)
    val_predictions.to_csv(train_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(train_dir / "test_predictions.csv", index=False)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(train_dir / "metrics.csv", index=False)
    (train_dir / "metrics.json").write_text(
        json.dumps([val_metrics, test_metrics], indent=2),
        encoding="utf-8",
    )

    export_summary = {
        "train_dir": str(train_dir),
        "store_dir": str(store_dir),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(num_workers),
        "stats_chunk_size": int(stats_chunk_size),
        "val_rows": int(len(val_predictions)),
        "test_rows": int(len(test_predictions)),
    }
    (train_dir / "prediction_export_summary.json").write_text(
        json.dumps(export_summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "export_summary": export_summary,
                "val": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
