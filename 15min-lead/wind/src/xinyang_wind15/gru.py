"""GRU baseline for multi-turbine 15-minute wind-speed forecasting."""

from __future__ import annotations

from torch import nn
import torch

from .sequence import (
    StandardizationStats,
    WindowDataset,
    compute_standardization_stats,
    evaluate_window_model,
    inverse_targets,
    make_loader,
    save_checkpoint,
    standardize_arrays,
)


class MultiTurbineGRU(nn.Module):
    """Flatten-per-step GRU baseline over multi-turbine windows."""

    def __init__(
        self,
        *,
        n_turbines: int,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # PyTorch documents GRU with batch_first=True as [batch, seq, feature].
        # Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.gru = nn.GRU(
            input_size=n_turbines * n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_turbines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_turbines, n_features = x.shape
        x = x.reshape(batch_size, steps, n_turbines * n_features)
        hidden_seq, _ = self.gru(x)
        return self.head(hidden_seq[:, -1, :])


def evaluate_gru_model(*args, **kwargs):
    return evaluate_window_model(*args, **kwargs)
