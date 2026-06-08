"""Package server-side xinyang Graph WaveNet outputs into repo-safe summaries."""

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

from xinyang_wind15.remote_results import (  # noqa: E402
    RemoteRunPackageSpec,
    package_remote_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="LSF job id, for example 5934.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Repo result folder name. Defaults to xinyang_gwnet_<job_id>.",
    )
    parser.add_argument(
        "--train-dir",
        default=str(PROJECT_DIR / "artifacts" / "server_runs" / "gwnet_full_run"),
        help="Directory containing training outputs such as metrics.json.",
    )
    parser.add_argument(
        "--store-dir",
        default=str(PROJECT_DIR / "artifacts" / "server_runs" / "xinyang_store_full"),
        help="Directory containing the store summary.json.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing LSF stdout/stderr logs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "results" / "remote_runs"),
        help="Root directory inside the repo for versionable remote run summaries.",
    )
    parser.add_argument(
        "--max-log-lines",
        type=int,
        default=200,
        help="Number of tail lines to keep from stdout/stderr logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"xinyang_gwnet_{args.job_id}"
    manifest = package_remote_run(
        RemoteRunPackageSpec(
            run_name=run_name,
            job_id=str(args.job_id),
            train_dir=Path(args.train_dir),
            store_dir=Path(args.store_dir),
            log_dir=Path(args.log_dir),
            output_root=Path(args.output_root),
            max_log_lines=int(args.max_log_lines),
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
