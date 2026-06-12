from __future__ import annotations

import torch

from xinyang_wind15.agcrn import AGCRNSeq2One
from xinyang_wind15.graph import build_graph_wavenet_supports
from xinyang_wind15.gru import MultiTurbineGRU
from xinyang_wind15.modern_tcn import ModernTCNForecaster
from xinyang_wind15.gwnet import GraphWaveNetLite
from xinyang_wind15.mtgnn import MTGNNLite
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


def test_graph_wavenet_lite_forward_shape_with_dynamic_supports() -> None:
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
        extra_support_len=2,
    )
    x = torch.randn(8, 32, 4, 6)
    dynamic_supports = [
        torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(8, 1, 1),
        torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(8, 1, 1),
    ]
    y = model(x, extra_supports=dynamic_supports)
    assert tuple(y.shape) == (8, 4)


def test_agcrn_forward_shape() -> None:
    model = AGCRNSeq2One(
        num_nodes=4,
        input_dim=2,
        hidden_dim=16,
        embed_dim=8,
        cheb_k=3,
        num_layers=2,
    )
    x = torch.randn(8, 32, 4, 2)
    y = model(x)
    assert tuple(y.shape) == (8, 4)


def test_mtgnn_lite_forward_shape() -> None:
    model = MTGNNLite(
        num_nodes=4,
        in_dim=2,
        seq_length=32,
        gcn_depth=2,
        subgraph_size=4,
        node_dim=8,
        conv_channels=16,
        residual_channels=16,
        skip_channels=32,
        end_channels=64,
        layers=2,
    )
    x = torch.randn(8, 32, 4, 2)
    y = model(x)
    assert tuple(y.shape) == (8, 4)


def test_modern_tcn_forward_shape() -> None:
    model = ModernTCNForecaster(
        n_vars=4,
        seq_len=32,
        hidden_dim=16,
        num_blocks=2,
        patch_size=4,
        patch_stride=2,
        large_kernel=9,
        small_kernel=3,
    )
    x = torch.randn(8, 32, 4)
    y = model(x)
    assert tuple(y.shape) == (8, 4)
