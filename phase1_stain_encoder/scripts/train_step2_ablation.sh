#!/bin/bash
# =============================================================================
# Step 2 aggregator ablation study — runs all 3 aggregator variants in parallel
#
# Aggregator variants:
#   abmil            → Attention-Based MIL  (proposed)       → cuda:3 (SM) / cuda:2 (WSI)
#   cross_attention  → Cross-Attention pooling (ablation)    → cuda:2 (SM) / cuda:1 (WSI)
#   mean_pool        → Simple mean pooling   (ablation)      → cuda:1 (SM) / cuda:0 (WSI)
#
# Logs are written to:
#   logs/step2_{domain}_{agg_type}.log
#
# Usage:
#   bash scripts/train_step2_ablation.sh [sm|wsi|both]
#
# Default: both domains
# =============================================================================

set -euo pipefail

DOMAIN="${1:-both}"

# Move to project root (one level above scripts/)
cd "$(dirname "$0")/.."
mkdir -p logs

# ---------------------------------------------------------------------------
# Helper: launch one training run in the background with nohup
# ---------------------------------------------------------------------------
launch() {
    local domain="$1"
    local agg_type="$2"
    local gpu_override="$3"
    local cfg="configs/step2_${domain}.yaml"
    local log="logs/step2_${domain}_${agg_type}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: domain=${domain}  agg=${agg_type}  gpu=${gpu_override}"
    nohup python step2/train.py \
        --config    "${cfg}" \
        --domain    "${domain}" \
        --agg_type  "${agg_type}" \
        --device    "${gpu_override}" \
        > "${log}" 2>&1 &
    echo "  PID=$!  log=${log}"
}

# ---------------------------------------------------------------------------
# SM ablations
# ---------------------------------------------------------------------------
if [[ "${DOMAIN}" == "sm" || "${DOMAIN}" == "both" ]]; then
    echo ""
    echo "========== SM ablations =========="
    launch sm abmil           cuda:3
    launch sm cross_attention cuda:2
    launch sm mean_pool       cuda:1
fi

# ---------------------------------------------------------------------------
# WSI ablations
# ---------------------------------------------------------------------------
if [[ "${DOMAIN}" == "wsi" || "${DOMAIN}" == "both" ]]; then
    echo ""
    echo "========== WSI ablations =========="
    launch wsi abmil           cuda:2
    launch wsi cross_attention cuda:1
    launch wsi mean_pool       cuda:0
fi

echo ""
echo "All jobs launched in background.  Monitor with:"
echo "  tail -f logs/step2_*.log"
echo ""
echo "Wait for all jobs:"
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Step 2 ablation runs complete."
