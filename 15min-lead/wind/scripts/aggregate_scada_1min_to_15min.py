"""Aggregate canonical 1-minute SCADA into 15-minute SCADA parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xinyang_wind15.loading import (  # noqa: E402
    aggregate_scada_1min_to_15min,
    filter_time_window,
    load_scada_1min,
)
from xinyang_wind15.settings import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-config", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-turbines", type=int, default=None)
    parser.add_argument("--tail-timestamps", type=int, default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.split_config) if args.split_config else None

    input_path = args.input_path
    if input_path is None:
        if settings is None:
            raise ValueError("Either --input-path or --split-config must be provided.")
        input_path = settings.data_paths["scada_1min"]

    start = args.start
    if start is None and settings is not None and settings.data_window_start is not None:
        start = str(settings.data_window_start)
    end = args.end
    if end is None and settings is not None and settings.data_window_end is not None:
        end = str(settings.data_window_end)

    scada_1min = load_scada_1min(
        input_path,
        max_turbines=args.max_turbines,
        tail_timestamps=args.tail_timestamps,
    )
    scada_1min = filter_time_window(
        scada_1min,
        timestamp_col="timestamp",
        start=start,
        end=end,
    )
    aggregated = aggregate_scada_1min_to_15min(scada_1min)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(output_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "start": start,
        "end": end,
        "n_rows_input": int(len(scada_1min)),
        "n_rows_output": int(len(aggregated)),
        "n_turbines": int(aggregated["turbine_id"].nunique()) if not aggregated.empty else 0,
        "timestamp_min": (
            str(aggregated["timestamp"].min()) if not aggregated.empty else None
        ),
        "timestamp_max": (
            str(aggregated["timestamp"].max()) if not aggregated.empty else None
        ),
    }
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
