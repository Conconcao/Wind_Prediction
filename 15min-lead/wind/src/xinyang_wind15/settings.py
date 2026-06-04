"""Configuration loading for xinyang 15-minute wind experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class SplitBounds:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ExperimentSettings:
    experiment_name: str
    site: str
    target_column: str
    horizon_steps: int
    base_resolution_minutes: int
    data_paths: dict[str, str]
    split_bounds: SplitBounds
    raw: dict[str, Any]


def _to_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def load_settings(path: str | Path) -> ExperimentSettings:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = raw["split"]
    bounds = SplitBounds(
        train_start=_to_timestamp(split["train_start"]),
        train_end=_to_timestamp(split["train_end"]),
        val_start=_to_timestamp(split["val_start"]),
        val_end=_to_timestamp(split["val_end"]),
        test_start=_to_timestamp(split["test_start"]),
        test_end=_to_timestamp(split["test_end"]),
    )
    experiment = raw["experiment"]
    return ExperimentSettings(
        experiment_name=experiment["name"],
        site=experiment["site"],
        target_column=experiment["target_column"],
        horizon_steps=int(experiment["horizon_steps"]),
        base_resolution_minutes=int(experiment["base_resolution_minutes"]),
        data_paths=raw["data"],
        split_bounds=bounds,
        raw=raw,
    )

