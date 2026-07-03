"""CNN+LSTM baseline for multi-turbine raw-SCADA forecasting."""

from __future__ import annotations

import torch
from torch import nn


class MultiTurbineCnnLstm(nn.Module):
    """Temporal Conv1D encoder followed by an LSTM decoder head."""

    def __init__(
        self,
        *,
        n_turbines: int,
        n_features: int,
        conv_channels: int = 64,
        kernel_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_channels = n_turbines * n_features
        padding = max(0, kernel_size // 2)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.BatchNorm1d(conv_channels),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_turbines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_turbines, n_features = x.shape
        x = x.reshape(batch_size, steps, n_turbines * n_features).transpose(1, 2)
        conv_out = self.temporal_conv(x).transpose(1, 2)
        encoded, _ = self.lstm(conv_out)
        return self.head(encoded[:, -1, :])
