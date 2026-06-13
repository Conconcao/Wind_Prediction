from __future__ import annotations

from xinyang_wind15.feature_presets import append_feature_block_columns, resolve_feature_columns


def test_resolve_feature_columns_hub_ws_only() -> None:
    columns = resolve_feature_columns(feature_preset="hub_ws_only")
    assert columns == ["ws_mean"]


def test_explicit_feature_columns_override_preset() -> None:
    columns = resolve_feature_columns(
        feature_preset="hub_ws_only",
        feature_columns=["ws_mean", "ws_std"],
    )
    assert columns == ["ws_mean", "ws_std"]


def test_direction_feature_presets() -> None:
    assert resolve_feature_columns(feature_preset="direction_wd_only") == [
        "ws_mean",
        "wd_sin",
        "wd_cos",
    ]
    assert resolve_feature_columns(feature_preset="direction_wd_yaw") == [
        "ws_mean",
        "wd_sin",
        "wd_cos",
        "nacelle_sin",
        "nacelle_cos",
    ]
    assert resolve_feature_columns(feature_preset="direction_wd_yaw_error") == [
        "ws_mean",
        "wd_sin",
        "wd_cos",
        "nacelle_sin",
        "nacelle_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "yaw_error_abs",
    ]


def test_append_feature_block_columns() -> None:
    columns = append_feature_block_columns(
        ["ws_mean"],
        frame_columns=[
            "ws_mean",
            "tower_ws_125m",
            "derived_ti_15m",
            "ctx_upwind_ws_mean",
            "ctx_upwind_ws_mean_missing",
        ],
        block_names=["tower", "derived_core", "spatial_context"],
    )
    assert columns == [
        "ctx_upwind_ws_mean",
        "ctx_upwind_ws_mean_missing",
        "derived_ti_15m",
        "tower_ws_125m",
        "ws_mean",
    ]
