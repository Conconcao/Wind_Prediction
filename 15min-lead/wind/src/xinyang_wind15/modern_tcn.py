"""ModernTCN-style temporal convolution model for joint wind forecasting."""

from __future__ import annotations

import torch
from torch import nn


class FlattenHead(nn.Module):
    """Forecast head adapted from the official ModernTCN repo."""

    def __init__(
        self,
        *,
        n_vars: int,
        hidden_dim: int,
        patch_count: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden_dim * patch_count, 1)
        self.dropout = nn.Dropout(head_dropout)
        self.n_vars = int(n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_vars, hidden_dim, patch_count = x.shape
        x = self.flatten(x.reshape(batch_size * n_vars, hidden_dim, patch_count))
        x = self.dropout(self.linear(x))
        return x.reshape(batch_size, n_vars)


class ReparamLargeKernelConv1d(nn.Module):
    """Large-kernel depthwise temporal conv with optional small branch."""

    def __init__(
        self,
        channels: int,
        *,
        large_kernel: int,
        small_kernel: int | None,
    ) -> None:
        super().__init__()
        padding = large_kernel // 2
        self.large_branch = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=large_kernel,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        if small_kernel is not None:
            self.small_branch = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=small_kernel,
                    padding=small_kernel // 2,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
            )
        else:
            self.small_branch = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.large_branch(x)
        if self.small_branch is not None:
            out = out + self.small_branch(x)
        return out


class ModernTCNBlock(nn.Module):
    """Compact ModernTCN block with grouped channel mixing."""

    def __init__(
        self,
        *,
        n_vars: int,
        hidden_dim: int,
        ffn_dim: int,
        large_kernel: int,
        small_kernel: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = n_vars * hidden_dim
        self.n_vars = int(n_vars)
        self.hidden_dim = int(hidden_dim)
        self.depthwise = ReparamLargeKernelConv1d(
            channels,
            large_kernel=large_kernel,
            small_kernel=small_kernel,
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.ffn1 = nn.Conv1d(
            channels,
            n_vars * ffn_dim,
            kernel_size=1,
            groups=n_vars,
        )
        self.act = nn.GELU()
        self.ffn2 = nn.Conv1d(
            n_vars * ffn_dim,
            channels,
            kernel_size=1,
            groups=n_vars,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, n_vars, hidden_dim, patch_count = x.shape
        x = x.reshape(batch_size, n_vars * hidden_dim, patch_count)
        x = self.depthwise(x)
        x = x.reshape(batch_size * n_vars, hidden_dim, patch_count)
        x = self.norm(x)
        x = x.reshape(batch_size, n_vars * hidden_dim, patch_count)
        x = self.dropout(self.ffn1(x))
        x = self.act(x)
        x = self.dropout(self.ffn2(x))
        x = x.reshape(batch_size, n_vars, hidden_dim, patch_count)
        return residual + x


class ModernTCNForecaster(nn.Module):
    """Seq2one ModernTCN-style forecaster for multi-series inputs."""

    def __init__(
        self,
        *,
        n_vars: int,
        seq_len: int,
        hidden_dim: int = 32,
        ffn_ratio: int = 2,
        num_blocks: int = 3,
        patch_size: int = 4,
        patch_stride: int = 2,
        large_kernel: int = 13,
        small_kernel: int | None = 5,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_stride < 1:
            raise ValueError("patch_stride must be positive.")
        self.n_vars = int(n_vars)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.hidden_dim = int(hidden_dim)
        effective_seq_len = self.seq_len + max(0, self.patch_size - self.patch_stride)

        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(hidden_dim),
        )
        patch_count = max(1, ((effective_seq_len - self.patch_size) // self.patch_stride) + 1)
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    n_vars=self.n_vars,
                    hidden_dim=hidden_dim,
                    ffn_dim=hidden_dim * int(ffn_ratio),
                    large_kernel=int(large_kernel),
                    small_kernel=small_kernel,
                    dropout=dropout,
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.head = FlattenHead(
            n_vars=self.n_vars,
            hidden_dim=hidden_dim,
            patch_count=patch_count,
            head_dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("ModernTCNForecaster expects [batch, steps, n_vars].")
        batch_size, steps, n_vars = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"Expected n_vars={self.n_vars}, got {n_vars}.")
        x = x.transpose(1, 2).reshape(batch_size * n_vars, 1, steps)
        if self.patch_size != self.patch_stride:
            pad_len = self.patch_size - self.patch_stride
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
        x = self.patch_embed(x)
        _, hidden_dim, patch_count = x.shape
        x = x.reshape(batch_size, n_vars, hidden_dim, patch_count)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
