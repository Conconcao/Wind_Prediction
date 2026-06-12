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
    "scada_core": [
        "ws_mean",
        "ws_std",
        "wd_mean",
        "power_mean",
        "nacelle_mean",
        "cnt_raw",
    ],
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
