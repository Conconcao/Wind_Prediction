"""CfC baseline for xinyang 15-minute wind-speed forecasting."""

from __future__ import annotations

import torch
from torch import nn

from .sequence import (
    compute_standardization_stats,
    evaluate_window_model,
    make_loader,
    save_checkpoint,
    standardize_arrays,
)

try:
    from ncps.torch import CfC as _NcpsCfC
except ImportError:  # pragma: no cover - exercised through runtime error path
    _NcpsCfC = None


class MultiTurbineCfC(nn.Module):
    """Flatten-per-step CfC baseline over multi-turbine windows."""

    def __init__(
        self,
        *,
        n_turbines: int,
        n_features: int,
        hidden_size: int = 128,
        mixed_memory: bool = False,
        mode: str = "default",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if _NcpsCfC is None:
            raise ImportError(
                "MultiTurbineCfC requires the optional dependency 'ncps'. "
                "Install it with `python -m pip install ncps` or install the project "
                "requirements file before running CfC experiments."
            )
        self.n_turbines = int(n_turbines)
        self.n_features = int(n_features)
        self.core = _NcpsCfC(
            input_size=self.n_turbines * self.n_features,
            units=int(hidden_size),
            proj_size=self.n_turbines,
            return_sequences=False,
            batch_first=True,
            mixed_memory=bool(mixed_memory),
            mode=str(mode),
            backbone_units=int(backbone_units),
            backbone_layers=int(backbone_layers),
            backbone_dropout=float(backbone_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_turbines, n_features = x.shape
        if n_turbines != self.n_turbines or n_features != self.n_features:
            raise ValueError(
                "MultiTurbineCfC received an unexpected input shape. "
                f"Expected turbines/features=({self.n_turbines}, {self.n_features}), "
                f"got ({n_turbines}, {n_features})."
            )
        x = x.reshape(batch_size, steps, n_turbines * n_features)
        output, _ = self.core(x)
        return output


def evaluate_cfc_model(*args, **kwargs):
    return evaluate_window_model(*args, **kwargs)
