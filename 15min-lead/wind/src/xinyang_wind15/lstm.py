"""LSTM baseline for multi-turbine raw-SCADA forecasting."""

from __future__ import annotations

import torch
from torch import nn


class MultiTurbineLSTM(nn.Module):
    """Flatten-per-step LSTM baseline over multi-turbine windows."""

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
        self.lstm = nn.LSTM(
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
        hidden_seq, _ = self.lstm(x)
        return self.head(hidden_seq[:, -1, :])
