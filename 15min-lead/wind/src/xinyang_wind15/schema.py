"""Schema definitions for raw 15-minute wind-farm data files."""

from __future__ import annotations


RAW_SCADA_15MIN_TO_CANONICAL = {
    "\u98ce\u673a": "turbine_id",
    "\u65f6\u95f4": "timestamp",
    "\u5e73\u5747\u98ce\u901f": "ws_mean",
    "\u98ce\u901f": "ws_mean",
    "\u6700\u5927\u98ce\u901f": "ws_max",
    "\u6700\u5c0f\u98ce\u901f": "ws_min",
    "\u98ce\u901f\u6807\u51c6\u5dee": "ws_std",
    "\u5e73\u5747\u6709\u529f\u529f\u7387": "power_mean",
    "\u6709\u529f\u529f\u7387": "power_mean",
    "\u6700\u5927\u6709\u529f\u529f\u7387": "power_max",
    "\u6700\u5c0f\u6709\u529f\u529f\u7387": "power_min",
    "\u6709\u529f\u529f\u7387\u6807\u51c6\u5dee": "power_std",
    "\u5e73\u5747\u673a\u8231\u65b9\u4f4d\u89d2": "nacelle_mean",
    "\u6700\u5927\u673a\u8231\u65b9\u4f4d\u89d2": "nacelle_max",
    "\u6700\u5c0f\u673a\u8231\u65b9\u4f4d\u89d2": "nacelle_min",
    "\u673a\u8231\u65b9\u4f4d\u89d2\u6807\u51c6\u5dee": "nacelle_std",
    "\u5e73\u5747\u98ce\u5411": "wd_mean",
    "\u6700\u5927\u98ce\u5411": "wd_max",
    "\u6700\u5c0f\u98ce\u5411": "wd_min",
    "\u98ce\u5411\u6807\u51c6\u5dee": "wd_std",
    "cnt_raw": "cnt_raw",
}

RAW_SCADA_15MIN_DIRECTION_TO_CANONICAL = {
    "\u540d\u79f0": "turbine_id",
    "\u65f6\u95f4": "timestamp",
    "\u98ce\u5411\u5e73\u5747": "wd_mean",
    "\u98ce\u5411\u6700\u5927": "wd_max",
    "\u98ce\u5411\u6700\u5c0f": "wd_min",
    "\u98ce\u5411\u6807\u51c6\u5dee": "wd_std",
    "\u673a\u8231\u4f4d\u7f6e\u5e73\u5747": "nacelle_mean",
    "\u673a\u8231\u4f4d\u7f6e\u6700\u5927": "nacelle_max",
    "\u673a\u8231\u4f4d\u7f6e\u6700\u5c0f": "nacelle_min",
    "\u673a\u8231\u4f4d\u7f6e\u6807\u51c6\u5dee": "nacelle_std",
}

RAW_SCADA_1MIN_TO_CANONICAL = {
    "\u98ce\u673a": "turbine_id",
    "\u65f6\u95f4": "timestamp",
    "\u98ce\u901f": "ws",
    "\u6709\u529f\u529f\u7387": "power",
    "\u673a\u8231\u65b9\u4f4d\u89d2": "nacelle_angle",
    "\u98ce\u5411": "wd",
    "\u7ecf\u5ea6": "longitude_deg",
    "\u7eac\u5ea6": "latitude_deg",
}

RAW_TOWER_TO_CANONICAL = {
    "\u65f6\u95f4": "timestamp",
    "\u5c42\u9ad8": "height_m",
    "\u6e29\u5ea6": "temperature",
    "\u6e7f\u5ea6": "humidity",
    "\u98ce\u901f": "ws",
    "\u98ce\u5411": "wd",
    "\u6c14\u538b": "pressure",
}

SCADA_15MIN_BASE_COLUMNS = [
    "turbine_id",
    "timestamp",
    "ws_mean",
    "ws_max",
    "ws_min",
    "ws_std",
    "power_mean",
    "power_max",
    "power_min",
    "power_std",
    "nacelle_mean",
    "nacelle_max",
    "nacelle_min",
    "nacelle_std",
    "wd_mean",
    "wd_max",
    "wd_min",
    "wd_std",
    "cnt_raw",
]

SCADA_15MIN_DIRECTION_COLUMNS = [
    "wd_mean",
    "wd_max",
    "wd_min",
    "wd_std",
    "nacelle_mean",
    "nacelle_max",
    "nacelle_min",
    "nacelle_std",
]

SCADA_1MIN_BASE_COLUMNS = [
    "turbine_id",
    "timestamp",
    "ws",
    "power",
    "nacelle_angle",
    "wd",
    "longitude_deg",
    "latitude_deg",
]

TOWER_WIDE_VARIABLES = ["temperature", "humidity", "ws", "wd", "pressure"]
