#!/usr/bin/env bash
# ==============================================================================
# 01_run_loso_baselines.sh
# ==============================================================================
# Automatically detects all available GPUs on the system and trains all 13
# specimen-invariance baselines in parallel across all available GPU slots.
#
# Baselines:
#   1. Proposed: conditional_grl
#   2. No intervention: focal, arcface, supcon, semihard_triplet
#   3. General regularization: strong_reg, mixup
#   4. Adversarial: dann_unconditional
#   5. Mutual Information: club
#   6. Strong-simple: frozen_linear
# ==============================================================================

set -e

BACKBONE="convnext_tiny"
FOLD=0
EPOCHS=17
BATCH_SIZE=64
METADATA="out/metadata/metadata.csv"
IMG_ROOT="out"
OUTPUT_DIR="specimen_invariance_outputs"

echo "=================================================================="
echo " Auto-Detecting GPUs & Running Baselines in Parallel              "
echo " Backbone: ${BACKBONE} | Fold: ${FOLD} | Epochs: ${EPOCHS}       "
echo " Output Dir: ${OUTPUT_DIR}                                        "
echo "=================================================================="

# Invoke the multi-GPU parallel dispatcher
python run_parallel_dispatcher.py \
    --mode baselines \
    --backbone "${BACKBONE}" \
    --fold "${FOLD}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --metadata_csv "${METADATA}" \
    --image_root "${IMG_ROOT}" \
    --output_base_dir "${OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo " All baseline runs completed! Check dispatcher_logs for details.  "
echo "=================================================================="
