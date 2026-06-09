#!/bin/bash
# =============================================================================
# Full evaluation pipeline for Phase 1 stain vectors.
#
# Usage:
#   bash scripts/eval.sh <stainvec_dir> [output_dir] [--skip_viz]
#
# Arguments:
#   stainvec_dir   Directory containing per-split stainvec outputs
#                  (see eval/run_eval.py for expected layout)
#   output_dir     Where to save eval_results.json and plots
#                  (default: <stainvec_dir>/eval_results)
#   --skip_viz     Pass to skip UMAP/t-SNE plot generation
#
# Example:
#   bash scripts/eval.sh \
#       runs/step2_sm_abmil/stainvecs \
#       runs/step2_sm_abmil/eval
# =============================================================================

set -euo pipefail

STAINVEC_DIR="${1:?Usage: $0 <stainvec_dir> [output_dir] [--skip_viz]}"
OUTPUT_DIR="${2:-${STAINVEC_DIR}/eval_results}"
SKIP_VIZ="${3:-}"

# Move to project root (one level above scripts/)
cd "$(dirname "$0")/.."

echo "============================================================"
echo " Phase 1 Stain Vector Evaluation"
echo " stainvec_dir : ${STAINVEC_DIR}"
echo " output_dir   : ${OUTPUT_DIR}"
echo "============================================================"

SKIP_FLAG=""
if [[ "${SKIP_VIZ}" == "--skip_viz" ]]; then
    SKIP_FLAG="--skip_viz"
    echo " Visualisation : SKIPPED"
else
    echo " Visualisation : enabled (UMAP + t-SNE)"
fi

echo ""

python eval/run_eval.py \
    --stainvec_dir  "${STAINVEC_DIR}" \
    --output_dir    "${OUTPUT_DIR}" \
    --splits        val internal_test unseen_test \
    --eval_pairs    val:internal_test val:unseen_test internal_test:unseen_test \
    ${SKIP_FLAG}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation complete."
echo "Results saved to: ${OUTPUT_DIR}/eval_results.json"
if [[ -z "${SKIP_FLAG}" ]]; then
    echo "Plots saved to  : ${OUTPUT_DIR}/plots/"
fi
