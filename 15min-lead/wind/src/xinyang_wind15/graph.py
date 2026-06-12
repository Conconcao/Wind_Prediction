"""Graph utilities for future spatio-temporal turbine models."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch


def _haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    radius_km = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(a))
    return radius_km * c


def _bearing_deg(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def build_distance_adjacency(
    turbine_meta: pd.DataFrame,
    turbine_order: Sequence[str],
    *,
    bandwidth_km: float = 0.5,
) -> np.ndarray:
    lookup = turbine_meta.set_index("turbine_id")
    coords = [
        (
            float(lookup.loc[turbine_id, "longitude_deg"]),
            float(lookup.loc[turbine_id, "latitude_deg"]),
        )
        for turbine_id in turbine_order
    ]
    n_turbines = len(turbine_order)
    adjacency = np.zeros((n_turbines, n_turbines), dtype=np.float32)
    for i, (lon_i, lat_i) in enumerate(coords):
        for j, (lon_j, lat_j) in enumerate(coords):
            if i == j:
                adjacency[i, j] = 1.0
                continue
            dist_km = _haversine_km(lon_i, lat_i, lon_j, lat_j)
            adjacency[i, j] = math.exp(-(dist_km**2) / (2.0 * bandwidth_km**2))
    return adjacency


def build_bearing_matrix(
    turbine_meta: pd.DataFrame,
    turbine_order: Sequence[str],
) -> np.ndarray:
    lookup = turbine_meta.set_index("turbine_id")
    coords = [
        (
            float(lookup.loc[turbine_id, "longitude_deg"]),
            float(lookup.loc[turbine_id, "latitude_deg"]),
        )
        for turbine_id in turbine_order
    ]
    n_turbines = len(turbine_order)
    bearing_matrix = np.zeros((n_turbines, n_turbines), dtype=np.float32)
    for source_idx, (lon_i, lat_i) in enumerate(coords):
        for target_idx, (lon_j, lat_j) in enumerate(coords):
            if source_idx == target_idx:
                continue
            bearing_matrix[source_idx, target_idx] = _bearing_deg(lon_i, lat_i, lon_j, lat_j)
    return bearing_matrix


def build_correlation_adjacency(
    scada_df: pd.DataFrame,
    turbine_order: Sequence[str],
    *,
    value_column: str = "ws_mean",
    min_periods: int = 96,
    positive_only: bool = True,
) -> np.ndarray:
    """Build a turbine-to-turbine correlation adjacency from training data only."""
    pivot = (
        scada_df.pivot(index="timestamp", columns="turbine_id", values=value_column)
        .reindex(columns=list(turbine_order))
    )
    corr = pivot.corr(method="pearson", min_periods=min_periods)
    adjacency = corr.to_numpy(dtype=np.float32)
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    if positive_only:
        adjacency = np.clip(adjacency, 0.0, None)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def row_normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Row-normalize an adjacency matrix with safe zero handling."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim == 2:
        row_sums = adjacency.sum(axis=1, keepdims=True)
    elif adjacency.ndim == 3:
        row_sums = adjacency.sum(axis=2, keepdims=True)
    else:
        raise ValueError(f"Expected 2D or 3D adjacency, got shape {adjacency.shape}.")
    row_sums = np.where(row_sums <= 1e-8, 1.0, row_sums)
    return adjacency / row_sums


def row_normalize_adjacency_torch(adjacency: torch.Tensor) -> torch.Tensor:
    row_sums = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return adjacency / row_sums


def build_directional_supports_torch(
    source_wind_direction_deg: torch.Tensor,
    *,
    bearing_matrix_deg: torch.Tensor,
    base_adjacency: torch.Tensor,
    sigma_deg: float = 30.0,
    include_transpose: bool = True,
) -> list[torch.Tensor]:
    if sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be positive for directional support construction.")
    if source_wind_direction_deg.ndim == 1:
        source_wind_direction_deg = source_wind_direction_deg.unsqueeze(0)
    if source_wind_direction_deg.ndim != 2:
        raise ValueError(
            "source_wind_direction_deg must have shape [nodes] or [batch, nodes]. "
            f"Got {tuple(source_wind_direction_deg.shape)}."
        )

    flow_direction = torch.remainder(source_wind_direction_deg + 180.0, 360.0).unsqueeze(-1)
    bearing = bearing_matrix_deg.unsqueeze(0)
    diff = torch.remainder(bearing - flow_direction + 180.0, 360.0) - 180.0
    diff = diff.abs()
    weights = torch.exp(-0.5 * torch.square(diff / float(sigma_deg)))
    weights = weights * base_adjacency.unsqueeze(0)
    eye = torch.eye(weights.size(-1), device=weights.device, dtype=weights.dtype).unsqueeze(0)
    weights = weights * (1.0 - eye) + eye

    supports = [row_normalize_adjacency_torch(weights)]
    if include_transpose:
        supports.append(row_normalize_adjacency_torch(weights.transpose(1, 2)))
    return supports


def build_graph_wavenet_supports(
    adjacency: np.ndarray | Sequence[np.ndarray],
    *,
    include_transpose: bool = True,
) -> list[np.ndarray]:
    """Prepare normalized supports for Graph WaveNet style graph convolutions."""
    adjacency_list = [adjacency] if isinstance(adjacency, np.ndarray) else list(adjacency)
    supports: list[np.ndarray] = []
    for adjacency_item in adjacency_list:
        normalized = row_normalize_adjacency(adjacency_item)
        supports.append(normalized)
        if include_transpose:
            supports.append(row_normalize_adjacency(adjacency_item.T))
    deduplicated: list[np.ndarray] = []
    for support in supports:
        if any(np.allclose(support, existing) for existing in deduplicated):
            continue
        deduplicated.append(support)
    return deduplicated


def supports_to_torch(
    supports: Sequence[np.ndarray],
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    """Move normalized adjacency supports to torch tensors."""
    return [torch.tensor(support, dtype=torch.float32, device=device) for support in supports]
