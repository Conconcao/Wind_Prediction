"""MTGNN-style graph temporal model for multi-turbine forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NodeConv(nn.Module):
    """Apply a node-to-node adjacency on [batch, channels, nodes, steps]."""

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, adjacency).contiguous()


class Linear1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MixProp(nn.Module):
    """Official MTGNN-style mix-hop propagation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        gcn_depth: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.nconv = NodeConv()
        self.mlp = Linear1x1((gcn_depth + 1) * in_channels, out_channels)
        self.gcn_depth = int(gcn_depth)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency + torch.eye(adjacency.size(0), device=x.device)
        degree = adjacency.sum(dim=1)
        norm_adj = adjacency / degree.clamp_min(1e-6).view(-1, 1)
        h = x
        out = [h]
        for _ in range(self.gcn_depth):
            h = self.alpha * x + (1.0 - self.alpha) * self.nconv(h, norm_adj)
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))


class DilatedInception(nn.Module):
    """Multi-kernel dilated temporal convolution from official MTGNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
    ) -> None:
        super().__init__()
        kernel_set = [2, 3, 6, 7]
        branch_out = int(out_channels / len(kernel_set))
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    branch_out,
                    kernel_size=(1, kernel),
                    dilation=(1, dilation),
                )
                for kernel in kernel_set
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        min_steps = min(output.size(3) for output in outputs)
        outputs = [output[..., -min_steps:] for output in outputs]
        return torch.cat(outputs, dim=1)


class GraphConstructor(nn.Module):
    """Learn a sparse directed adjacency like the official MTGNN graph constructor."""

    def __init__(
        self,
        num_nodes: int,
        *,
        k: int,
        embed_dim: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.k = int(k)
        self.alpha = float(alpha)
        self.emb1 = nn.Embedding(self.num_nodes, embed_dim)
        self.emb2 = nn.Embedding(self.num_nodes, embed_dim)
        self.lin1 = nn.Linear(embed_dim, embed_dim)
        self.lin2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        nodevec1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        nodevec2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        logits = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2,
            nodevec1.transpose(1, 0),
        )
        adjacency = F.relu(torch.tanh(self.alpha * logits))
        mask = torch.zeros_like(adjacency)
        _, top_idx = (adjacency + torch.rand_like(adjacency) * 0.01).topk(self.k, dim=1)
        mask.scatter_(1, top_idx, 1.0)
        return adjacency * mask


class MTGNNLite(nn.Module):
    """Compact MTGNN-style seq2one forecaster."""

    def __init__(
        self,
        *,
        num_nodes: int,
        in_dim: int,
        seq_length: int,
        gcn_depth: int = 2,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        dilation_exponential: int = 2,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: float = 3.0,
        gcn_true: bool = True,
        build_graph: bool = True,
    ) -> None:
        super().__init__()
        self.gcn_true = bool(gcn_true)
        self.build_graph = bool(build_graph)
        self.num_nodes = int(num_nodes)
        self.seq_length = int(seq_length)
        self.dropout = float(dropout)
        self.layers = int(layers)
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**self.layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.graph_constructor = GraphConstructor(
            self.num_nodes,
            k=min(int(subgraph_size), self.num_nodes),
            embed_dim=int(node_dim),
            alpha=float(tanhalpha),
        )
        self.register_buffer("node_idx", torch.arange(self.num_nodes), persistent=False)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norms = nn.ModuleList()

        new_dilation = 1
        for layer in range(self.layers):
            rf_size = (
                int(
                    1
                    + (kernel_size - 1)
                    * (dilation_exponential ** (layer + 1) - 1)
                    / (dilation_exponential - 1)
                )
                if dilation_exponential > 1
                else 1 + (layer + 1) * (kernel_size - 1)
            )
            self.filter_convs.append(
                DilatedInception(
                    residual_channels,
                    conv_channels,
                    dilation=new_dilation,
                )
            )
            self.gate_convs.append(
                DilatedInception(
                    residual_channels,
                    conv_channels,
                    dilation=new_dilation,
                )
            )
            self.residual_convs.append(
                nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
            )
            skip_kernel = (
                self.seq_length - rf_size + 1
                if self.seq_length > self.receptive_field
                else self.receptive_field - rf_size + 1
            )
            self.skip_convs.append(
                nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, skip_kernel))
            )
            self.gconv1.append(
                MixProp(
                    conv_channels,
                    residual_channels,
                    gcn_depth=gcn_depth,
                    alpha=propalpha,
                )
            )
            self.gconv2.append(
                MixProp(
                    conv_channels,
                    residual_channels,
                    gcn_depth=gcn_depth,
                    alpha=propalpha,
                )
            )
            self.norms.append(nn.BatchNorm2d(residual_channels))
            new_dilation *= dilation_exponential

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.seq_length))
            self.skipE = nn.Conv2d(
                residual_channels,
                skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
            )
        else:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.receptive_field))
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1).contiguous()
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        raw_input = x

        adjacency = None
        if self.gcn_true:
            adjacency = self.graph_constructor(self.node_idx.to(x.device)) if self.build_graph else None

        x = self.start_conv(x)
        skip = self.skip0(F.dropout(raw_input, self.dropout, training=self.training))
        for layer_idx in range(self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[layer_idx](x))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](x))
            x = F.dropout(filter_out * gate_out, self.dropout, training=self.training)

            skip = skip + self.skip_convs[layer_idx](x)
            if self.gcn_true and adjacency is not None:
                x = self.gconv1[layer_idx](x, adjacency) + self.gconv2[layer_idx](
                    x,
                    adjacency.transpose(0, 1),
                )
            else:
                x = self.residual_convs[layer_idx](x)

            x = x + residual[..., -x.size(3) :]
            x = self.norms[layer_idx](x)

        x = self.skipE(x) + skip
        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[:, 0, :, -1]
