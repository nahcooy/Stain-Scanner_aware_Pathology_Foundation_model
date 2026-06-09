#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/eval.sh <config.yaml> <checkpoint.pth> <output_dir>"
  exit 1
fi

CONFIG_PATH="$1"
CHECKPOINT_PATH="$2"
OUTPUT_DIR="$3"

python ssl/run_eval.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR"

