"""Graph WaveNet style model for multi-turbine 15-minute forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NeighborhoodConv(nn.Module):
    """Apply a graph support to node features."""

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Graph WaveNet uses einsum over [batch, channels, nodes, steps].
        # Reference repo: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
        # torch.einsum docs: https://docs.pytorch.org/docs/stable/generated/torch.einsum.html
        return torch.einsum("bcnt,nm->bcmt", x, adjacency).contiguous()


class Linear1x1(nn.Module):
    """Pointwise 1x1 Conv2d projection on [batch, channels, nodes, steps]."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Conv2d expects [N, C, H, W].
        # Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GraphConv(nn.Module):
    """Graph WaveNet style multi-support graph convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        support_len: int,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.neighborhood_conv = NeighborhoodConv()
        self.order = int(order)
        self.dropout = float(dropout)
        expanded_in = (self.order * support_len + 1) * in_channels
        self.mlp = Linear1x1(expanded_in, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        supports: list[torch.Tensor],
    ) -> torch.Tensor:
        out = [x]
        for adjacency in supports:
            x_k = self.neighborhood_conv(x, adjacency)
            out.append(x_k)
            for _ in range(2, self.order + 1):
                x_k = self.neighborhood_conv(x_k, adjacency)
                out.append(x_k)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNetLite(nn.Module):
    """A compact Graph WaveNet style seq2one forecaster."""

    def __init__(
        self,
        *,
        num_nodes: int,
        in_dim: int,
        supports: list[torch.Tensor] | None = None,
        dropout: float = 0.3,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 128,
        end_channels: int = 256,
        kernel_size: int = 2,
        blocks: int = 2,
        layers: int = 2,
        graph_order: int = 2,
        adaptive_embedding_dim: int = 10,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.blocks = int(blocks)
        self.layers = int(layers)
        self.gcn_bool = bool(gcn_bool)
        self.addaptadj = bool(addaptadj)
        self.kernel_size = int(kernel_size)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.graph_convs = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        self.supports = supports if supports is not None else []
        self.supports_len = len(self.supports)
        if self.gcn_bool and self.addaptadj:
            self.nodevec1 = nn.Parameter(
                torch.randn(num_nodes, adaptive_embedding_dim),
                requires_grad=True,
            )
            self.nodevec2 = nn.Parameter(
                torch.randn(adaptive_embedding_dim, num_nodes),
                requires_grad=True,
            )
            self.supports_len += 1
        else:
            self.register_parameter("nodevec1", None)
            self.register_parameter("nodevec2", None)

        receptive_field = 1
        for _ in range(self.blocks):
            additional_scope = self.kernel_size - 1
            dilation = 1
            for _ in range(self.layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        dilation_channels,
                        residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        dilation_channels,
                        skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.batch_norms.append(nn.BatchNorm2d(residual_channels))
                if self.gcn_bool:
                    self.graph_convs.append(
                        GraphConv(
                            dilation_channels,
                            residual_channels,
                            dropout=self.dropout,
                            support_len=self.supports_len,
                            order=graph_order,
                        )
                    )
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )

    def _current_supports(self) -> list[torch.Tensor]:
        supports = list(self.supports)
        if self.gcn_bool and self.addaptadj and self.nodevec1 is not None and self.nodevec2 is not None:
            adaptive = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
            supports.append(adaptive)
        return supports

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input comes from the store dataset as [batch, steps, nodes, features].
        x = x.permute(0, 3, 2, 1).contiguous()
        in_steps = x.size(3)
        if in_steps < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_steps, 0, 0, 0))

        x = self.start_conv(x)
        skip = None
        supports = self._current_supports()

        for layer_idx in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[layer_idx](residual))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](residual))
            x = filter_out * gate_out

            skip_part = self.skip_convs[layer_idx](x)
            if skip is None:
                skip = skip_part
            else:
                skip = skip[..., -skip_part.size(3) :] + skip_part

            if self.gcn_bool and supports:
                x = self.graph_convs[layer_idx](x, supports)
            else:
                x = self.residual_convs[layer_idx](x)

            x = x + residual[..., -x.size(3) :]
            x = self.batch_norms[layer_idx](x)

        if skip is None:
            raise RuntimeError("GraphWaveNetLite forward pass produced no skip activations.")
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # Use the most recent causal output slice for one-step forecasting.
        return x[:, 0, :, -1]
