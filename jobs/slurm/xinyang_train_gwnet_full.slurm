#!/bin/bash
#SBATCH --job-name=xinyang_gwnet_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home3/s502024280003/Wind_Prediction/logs/%x_%j.out
#SBATCH --error=/home3/s502024280003/Wind_Prediction/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/home3/s502024280003/Wind_Prediction"
STORE_DIR="$PROJECT_ROOT/15min-lead/wind/artifacts/server_runs/xinyang_store_full"
RUN_DIR="$PROJECT_ROOT/15min-lead/wind/artifacts/server_runs/gwnet_full_run"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# Adjust this environment activation block to match the server runtime.
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python 15min-lead/wind/scripts/train_gwnet_from_store.py \
  --store-dir "$STORE_DIR" \
  --support-mode distance_correlation \
  --epochs 30 \
  --batch-size 64 \
  --residual-channels 32 \
  --dilation-channels 32 \
  --skip-channels 128 \
  --end-channels 256 \
  --kernel-size 2 \
  --blocks 2 \
  --layers 2 \
  --graph-order 2 \
  --output-dir "$RUN_DIR"
