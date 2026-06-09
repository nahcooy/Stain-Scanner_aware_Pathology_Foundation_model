#!/bin/bash
# =============================================================================
# Train Step 1 patch encoder — SM (Scanning Microscope) domain
#
# Trains a ViT-L encoder with supervised contrastive learning on PLISM patches.
# Positive pairs are defined by pos_key = device||stain.
# Output: checkpoints saved to ${RUN_ROOT}/step1_sm/
# =============================================================================

set -euo pipefail

# Move to project root (one level above scripts/)
cd "$(dirname "$0")/.."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Step 1 SM training ..."
python step1/train.py --config configs/step1_sm.yaml --domain sm "$@"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1 SM training finished."
