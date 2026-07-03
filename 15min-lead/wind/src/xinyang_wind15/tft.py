"""Compact TFT-style forecaster for multi-turbine raw-SCADA experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


def _build_sinusoidal_positional_encoding(steps: int, dim: int) -> torch.Tensor:
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(steps, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalFusionTransformerLite(nn.Module):
    """Compact TFT-style seq2one forecaster with input gating and self-attention."""

    def __init__(
        self,
        *,
        n_turbines: int,
        n_features: int,
        lookback_steps: int,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_size = n_turbines * n_features
        self.lookback_steps = int(lookback_steps)
        self.hidden_size = int(hidden_size)

        self.variable_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )
        pe = _build_sinusoidal_positional_encoding(self.lookback_steps, hidden_size)
        self.register_buffer("positional_encoding", pe.unsqueeze(0), persistent=False)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_turbines),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_turbines, n_features = x.shape
        if steps > self.lookback_steps:
            raise ValueError(
                f"TemporalFusionTransformerLite expects at most {self.lookback_steps} steps, "
                f"got {steps}."
            )
        x = x.reshape(batch_size, steps, n_turbines * n_features)
        gated = x * self.variable_gate(x)
        hidden = self.input_projection(gated)
        hidden = hidden * self.context_gate(hidden)
        hidden = hidden + self.positional_encoding[:, :steps, :]
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])
