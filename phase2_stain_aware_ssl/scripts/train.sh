#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/train.sh <config.yaml>"
  exit 1
fi

CONFIG_PATH="$1"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" ssl/train.py --config "$CONFIG_PATH"

