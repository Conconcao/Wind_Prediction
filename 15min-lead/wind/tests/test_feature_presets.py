from __future__ import annotations

from xinyang_wind15.feature_presets import resolve_feature_columns


def test_resolve_feature_columns_hub_ws_only() -> None:
    columns = resolve_feature_columns(feature_preset="hub_ws_only")
    assert columns == ["ws_mean"]


def test_explicit_feature_columns_override_preset() -> None:
    columns = resolve_feature_columns(
        feature_preset="hub_ws_only",
        feature_columns=["ws_mean", "ws_std"],
    )
    assert columns == ["ws_mean", "ws_std"]
