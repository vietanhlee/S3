#!/usr/bin/env bash
# ==============================================================================
# 04_evaluate_all_metrics.sh
# ==============================================================================
# Iterates through trained checkpoints and evaluates:
#   1. GGSL (Acc & Macro-F1 gap)
#   2. Specimen Recoverability Index (SRI) via 50/50 probe
#   3. Expected Calibration Error (ECE) & Reliability Diagrams
#   4. Dual t-SNE projections (Species vs. Specimen)
# ==============================================================================

set -e

BACKBONE="convnext_tiny"
FOLD=0
METADATA="out/metadata/metadata.csv"
IMG_ROOT="out"
BASE_DIR="specimen_invariance_outputs"

METHODS=(
    "focal"
    "mixup"
    "dann_unconditional"
    "club"
    "conditional_grl"
)

echo "=================================================================="
echo " Starting Full Module 3 Evaluation Across Trained Models          "
echo "=================================================================="

for METHOD in "${METHODS[@]}"; do
    RUN_DIR="${BASE_DIR}/${METHOD}_${BACKBONE}_fold${FOLD}_seed42"
    CKPT="${RUN_DIR}/best_model.pth"
    
    if [ -f "${CKPT}" ]; then
        echo ""
        echo ">>> Evaluating Method: ${METHOD} (${CKPT}) <<<"
        python evaluate.py \
            --checkpoint "${CKPT}" \
            --backbone "${BACKBONE}" \
            --fold "${FOLD}" \
            --metadata_csv "${METADATA}" \
            --image_root "${IMG_ROOT}" \
            --output_dir "${RUN_DIR}/eval_results"
    else
        echo "[!] Checkpoint not found, skipping: ${CKPT}"
    fi
done

echo ""
echo "=================================================================="
echo " Evaluation completed for all available checkpoints!              "
echo "=================================================================="
