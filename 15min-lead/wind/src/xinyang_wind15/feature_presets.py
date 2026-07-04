"""Named feature presets for xinyang 15-minute experiments."""

from __future__ import annotations


FEATURE_PRESETS: dict[str, list[str]] = {
    "default_multivariate": [
        "ws_mean",
        "power_mean",
        "wd_mean",
        "nacelle_mean",
        "ws_std",
    ],
    "hub_ws_only": [
        "ws_mean",
    ],
    "direction_wd_only": [
        "ws_mean",
        "wd_sin",
        "wd_cos",
    ],
    "direction_wd_yaw": [
        "ws_mean",
        "wd_sin",
        "wd_cos",
        "nacelle_sin",
        "nacelle_cos",
    ],
    "direction_wd_yaw_error": [
        "ws_mean",
        "wd_sin",
        "wd_cos",
        "nacelle_sin",
        "nacelle_cos",
        "yaw_error_sin",
        "yaw_error_cos",
        "yaw_error_abs",
    ],
    "multivariate_directional": [
        "ws_mean",
        "power_mean",
        "ws_std",
        "wd_sin",
        "wd_cos",
        "wd_std",
        "nacelle_sin",
        "nacelle_cos",
        "nacelle_std",
        "yaw_error_sin",
        "yaw_error_cos",
        "yaw_error_abs",
    ],
    "huaian_directional_core": [
        "ws_mean",
        "power_mean",
        "wd_sin",
        "wd_cos",
        "wd_std",
        "nacelle_sin",
        "nacelle_cos",
        "nacelle_std",
        "yaw_error_sin",
        "yaw_error_cos",
        "yaw_error_abs",
    ],
    "scada_core": [
        "ws_mean",
        "ws_std",
        "wd_mean",
        "power_mean",
        "nacelle_mean",
        "cnt_raw",
    ],
}

FEATURE_BLOCK_PREFIXES: dict[str, tuple[str, ...]] = {
    "tower": ("tower_",),
    "one_min": ("m1_",),
    "derived_core": ("derived_", "profile_", "hub_tower_"),
    "spatial_context": ("ctx_",),
}


def resolve_feature_columns(
    *,
    feature_preset: str,
    feature_columns: list[str] | None = None,
) -> list[str]:
    if feature_columns:
        return list(feature_columns)
    if feature_preset not in FEATURE_PRESETS:
        raise KeyError(
            f"Unknown feature preset: {feature_preset}. "
            f"Available presets: {sorted(FEATURE_PRESETS)}"
        )
    return list(FEATURE_PRESETS[feature_preset])


def append_feature_block_columns(
    feature_columns: list[str],
    *,
    frame_columns: list[str],
    block_names: list[str],
) -> list[str]:
    selected = list(feature_columns)
    for block_name in block_names:
        if block_name not in FEATURE_BLOCK_PREFIXES:
            raise KeyError(
                f"Unknown feature block: {block_name}. "
                f"Available blocks: {sorted(FEATURE_BLOCK_PREFIXES)}"
            )
        prefixes = FEATURE_BLOCK_PREFIXES[block_name]
        selected.extend(
            column
            for column in frame_columns
            if column.startswith(prefixes)
        )
    return sorted(set(selected))


def validate_feature_columns(
    feature_columns: list[str],
    *,
    frame,
) -> list[str]:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Selected features are missing from the feature frame: "
            f"{missing}"
        )
    all_nan = [column for column in feature_columns if frame[column].isna().all()]
    if all_nan:
        raise ValueError(
            "Selected features are entirely missing for the current dataset/time window: "
            f"{all_nan}"
        )
    return list(feature_columns)
