from __future__ import annotations

import torch

from xinyang_wind15.graph import build_graph_wavenet_supports
from xinyang_wind15.gru import MultiTurbineGRU
from xinyang_wind15.gwnet import GraphWaveNetLite
from xinyang_wind15.tcn import MultiTurbineTCN


def test_multi_turbine_gru_forward_shape() -> None:
    model = MultiTurbineGRU(
        n_turbines=4,
        n_features=6,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
    )
    x = torch.randn(8, 32, 4, 6)
    y = model(x)
    assert tuple(y.shape) == (8, 4)


def test_multi_turbine_tcn_forward_shape() -> None:
    model = MultiTurbineTCN(
        n_turbines=4,
        n_features=6,
        channel_sizes=[16, 16, 16],
        kernel_size=3,
        dropout=0.1,
    )
    x = torch.randn(8, 32, 4, 6)
    y = model(x)
    assert tuple(y.shape) == (8, 4)


def test_graph_wavenet_lite_forward_shape() -> None:
    adjacency = torch.eye(4, dtype=torch.float32).numpy()
    supports = [
        torch.tensor(support, dtype=torch.float32)
        for support in build_graph_wavenet_supports(adjacency)
    ]
    model = GraphWaveNetLite(
        num_nodes=4,
        in_dim=6,
        supports=supports,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=32,
        end_channels=64,
        blocks=2,
        layers=2,
    )
    x = torch.randn(8, 32, 4, 6)
    y = model(x)
    assert tuple(y.shape) == (8, 4)
