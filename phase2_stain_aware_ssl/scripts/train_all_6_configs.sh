#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

CONFIGS=(
  "$ROOT_DIR/configs/tcga_scratch_adaln.yaml"
  "$ROOT_DIR/configs/tcga_scratch_prompt.yaml"
  "$ROOT_DIR/configs/tcga_scratch_cross_attention.yaml"
  "$ROOT_DIR/configs/tcga_uni_frozen_adaln.yaml"
  "$ROOT_DIR/configs/tcga_uni_frozen_prompt.yaml"
  "$ROOT_DIR/configs/tcga_uni_frozen_cross_attention.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "[RUN] $cfg (nproc_per_node=$NPROC_PER_NODE)"
  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" "$ROOT_DIR/ssl/train.py" --config "$cfg"
done

