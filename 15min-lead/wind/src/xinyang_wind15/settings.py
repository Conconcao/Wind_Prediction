"""Configuration loading for xinyang 15-minute wind experiments."""

from __future__ import annotations

import os
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
    data_window_start: pd.Timestamp | None
    data_window_end: pd.Timestamp | None
    raw: dict[str, Any]


def _to_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def _to_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return _to_timestamp(value)


def default_split_config_path(
    project_dir: str | Path,
    *,
    prefer_server: bool | None = None,
) -> Path:
    root = Path(project_dir)
    splits_dir = root / "configs" / "splits"
    local_config = splits_dir / "xinyang_7_2_1.yaml"
    server_config = splits_dir / "xinyang_7_2_1_server.yaml"
    use_server = (os.name != "nt") if prefer_server is None else bool(prefer_server)
    if use_server and server_config.exists():
        return server_config
    return local_config


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
        data_window_start=_to_optional_timestamp(split.get("data_window_start")),
        data_window_end=_to_optional_timestamp(split.get("data_window_end")),
        raw=raw,
    )
