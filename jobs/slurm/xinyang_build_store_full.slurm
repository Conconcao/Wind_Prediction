#!/bin/bash
#SBATCH --job-name=xinyang_store_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/s502024280003/gpfs/Wind_Prediction/logs/%x_%j.out
#SBATCH --error=/s502024280003/gpfs/Wind_Prediction/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/s502024280003/gpfs/Wind_Prediction"
STORE_DIR="$PROJECT_ROOT/15min-lead/wind/artifacts/server_runs/xinyang_store_full"
CONFIG_PATH="$PROJECT_ROOT/15min-lead/wind/configs/splits/xinyang_7_2_1_server.yaml"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# Adjust this environment activation block to match the server runtime.
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python 15min-lead/wind/scripts/build_window_store.py \
  --split-config "$CONFIG_PATH" \
  --lookback-steps 32 \
  --include-tower \
  --include-1min \
  --output-dir "$STORE_DIR"
