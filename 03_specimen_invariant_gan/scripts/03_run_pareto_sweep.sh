#!/usr/bin/env bash
# ==============================================================================
# 03_run_pareto_sweep.sh
# ==============================================================================
# Sweeps the adversarial weighting parameter lambda_adv in [0.0, 2.0] to map:
#   1. Pareto frontier between strict accuracy and Specimen Recoverability Index (SRI)
#   2. Linear correlation between SRI and Generalization Gap (GGSL)
# ==============================================================================

set -e

BACKBONE="convnext_tiny"
FOLD=0
EPOCHS=17
BATCH_SIZE=64
METADATA="out/metadata/metadata.csv"
IMG_ROOT="out"
OUTPUT_DIR="specimen_invariance_outputs/pareto_sweep"

echo "=================================================================="
echo " Starting lambda_adv Pareto Sweep and Correlation Analysis        "
echo " Backbone: ${BACKBONE} | Epochs: ${EPOCHS}                        "
echo "=================================================================="

python run_pareto_sweep.py \
    --backbone "${BACKBONE}" \
    --fold "${FOLD}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lambdas 0.0 0.1 0.25 0.5 1.0 2.0 \
    --metadata_csv "${METADATA}" \
    --image_root "${IMG_ROOT}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo " Pareto sweep complete! Figures saved in: ${OUTPUT_DIR}            "
echo "=================================================================="
