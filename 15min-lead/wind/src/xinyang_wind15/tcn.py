"""TCN baseline for multi-turbine 15-minute wind-speed forecasting."""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Trim right padding so the convolution stays causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual causal convolution block inspired by the locuslab TCN repo."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        # PyTorch Conv1d uses [batch, channels, length].
        # Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.out_relu(self.net(x) + residual)


class TemporalConvNet(nn.Module):
    """Stacked causal residual convolution blocks."""

    def __init__(
        self,
        in_channels: int,
        channel_sizes: list[int],
        *,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        for level, out_channels in enumerate(channel_sizes):
            dilation = 2**level
            block_in = in_channels if level == 0 else channel_sizes[level - 1]
            layers.append(
                TemporalBlock(
                    block_in,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiTurbineTCN(nn.Module):
    """Flattened multi-turbine TCN seq2one baseline."""

    def __init__(
        self,
        *,
        n_turbines: int,
        n_features: int,
        channel_sizes: list[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not channel_sizes:
            raise ValueError("channel_sizes must contain at least one layer width.")
        in_channels = n_turbines * n_features
        self.tcn = TemporalConvNet(
            in_channels,
            channel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(channel_sizes[-1], n_turbines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_turbines, n_features = x.shape
        x = x.reshape(batch_size, steps, n_turbines * n_features)
        x = x.transpose(1, 2)
        encoded = self.tcn(x)
        last_hidden = encoded[:, :, -1]
        return self.head(last_hidden)
