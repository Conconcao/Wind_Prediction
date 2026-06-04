"""Chronological split helpers."""

from __future__ import annotations

import pandas as pd

from .settings import SplitBounds


def assign_split_labels(
    target_timestamps: pd.Series,
    bounds: SplitBounds,
) -> pd.Series:
    split = pd.Series("discard", index=target_timestamps.index, dtype="object")
    split.loc[
        (target_timestamps >= bounds.train_start)
        & (target_timestamps <= bounds.train_end)
    ] = "train"
    split.loc[
        (target_timestamps >= bounds.val_start)
        & (target_timestamps <= bounds.val_end)
    ] = "val"
    split.loc[
        (target_timestamps >= bounds.test_start)
        & (target_timestamps <= bounds.test_end)
    ] = "test"
    return split

