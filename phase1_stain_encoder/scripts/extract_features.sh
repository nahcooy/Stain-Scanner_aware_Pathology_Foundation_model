#!/bin/bash
# =============================================================================
# Extract patch-level features for both SM and WSI domains after Step 1.
#
# Usage:
#   bash scripts/extract_features.sh <sm_checkpoint> <wsi_checkpoint>
#
# Arguments:
#   sm_checkpoint   Path to Step 1 SM checkpoint (.pth)
#   wsi_checkpoint  Path to Step 1 WSI checkpoint (.pth)
#
# Output layout:
#   features/
#     sm/
#       {split}/
#         {pos_key}.npy   [N, 1024]
#     wsi/
#       {split}/
#         {pos_key}.npy   [N, 1024]
# =============================================================================

set -euo pipefail

SM_CKPT="${1:?Usage: $0 <sm_checkpoint> <wsi_checkpoint>}"
WSI_CKPT="${2:?Usage: $0 <sm_checkpoint> <wsi_checkpoint>}"

# Move to project root (one level above scripts/)
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Feature extraction — SM domain"
echo " Checkpoint : ${SM_CKPT}"
echo "============================================================"
python step1/extract_features.py \
    --config     configs/step1_sm.yaml \
    --domain     sm \
    --checkpoint "${SM_CKPT}"

echo ""
echo "============================================================"
echo " Feature extraction — WSI domain"
echo " Checkpoint : ${WSI_CKPT}"
echo "============================================================"
python step1/extract_features.py \
    --config     configs/step1_wsi.yaml \
    --domain     wsi \
    --checkpoint "${WSI_CKPT}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Feature extraction complete."
