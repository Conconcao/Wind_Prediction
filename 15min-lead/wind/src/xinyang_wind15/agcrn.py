"""AGCRN-style adaptive graph recurrent model for multi-turbine forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveVertexWiseGraphConv(nn.Module):
    """Chebyshev graph convolution with node-adaptive parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cheb_k: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.cheb_k = int(cheb_k)
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, self.cheb_k, in_channels, out_channels),
            requires_grad=True,
        )
        self.bias_pool = nn.Parameter(
            torch.empty(embed_dim, out_channels),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        node_num = int(node_embeddings.shape[0])
        supports = F.softmax(F.relu(node_embeddings @ node_embeddings.transpose(0, 1)), dim=1)
        support_set = [torch.eye(node_num, device=x.device), supports]
        for _ in range(2, self.cheb_k):
            support_set.append((2 * supports @ support_set[-1]) - support_set[-2])
        support_stack = torch.stack(support_set[: self.cheb_k], dim=0)
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        x_g = torch.einsum("knm,bmc->bknc", support_stack, x)
        x_g = x_g.permute(0, 2, 1, 3).contiguous()
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    """Official AGCRN-style recurrent cell."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        *,
        cheb_k: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.node_num = int(node_num)
        self.hidden_dim = int(hidden_dim)
        self.gate = AdaptiveVertexWiseGraphConv(
            input_dim + hidden_dim,
            2 * hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
        )
        self.update = AdaptiveVertexWiseGraphConv(
            input_dim + hidden_dim,
            hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        state = state.to(x.device)
        combined = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate(combined, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, z * state], dim=-1)
        h_tilde = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1.0 - r) * h_tilde

    def init_hidden_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)


class AGCRNEncoder(nn.Module):
    """Stacked AGCRN recurrent encoder."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        *,
        cheb_k: int,
        embed_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("AGCRNEncoder requires at least one recurrent layer.")
        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        cells = [
            AGCRNCell(
                node_num,
                input_dim,
                hidden_dim,
                cheb_k=cheb_k,
                embed_dim=embed_dim,
            )
        ]
        for _ in range(1, self.num_layers):
            cells.append(
                AGCRNCell(
                    node_num,
                    hidden_dim,
                    hidden_dim,
                    cheb_k=cheb_k,
                    embed_dim=embed_dim,
                )
            )
        self.cells = nn.ModuleList(cells)

    def init_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.stack(
            [cell.init_hidden_state(batch_size, device=device) for cell in self.cells],
            dim=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(x.shape[0])
        init_state = self.init_hidden(batch_size, device=x.device)
        current_inputs = x
        output_hidden = []
        for layer_idx, cell in enumerate(self.cells):
            state = init_state[layer_idx]
            inner_states = []
            for step in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, step, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, torch.stack(output_hidden, dim=0)


class AGCRNSeq2One(nn.Module):
    """Compact seq2one AGCRN forecaster for one-step turbine-level prediction."""

    def __init__(
        self,
        *,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        embed_dim: int = 10,
        cheb_k: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, embed_dim),
            requires_grad=True,
        )
        self.encoder = AGCRNEncoder(
            self.num_nodes,
            input_dim,
            hidden_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.output_dim,
            kernel_size=(1, hidden_dim),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x, self.node_embeddings)
        last_hidden = encoded[:, -1:, :, :]
        out = self.end_conv(last_hidden)
        out = out.squeeze(-1).permute(0, 2, 1).contiguous()
        return out[:, :, 0]
