#!/usr/bin/env bash
# ==============================================================================
# 01_run_loso_baselines.sh
# ==============================================================================
# Automatically detects all available GPUs on the system and trains all 13
# specimen-invariance baselines in parallel across all available GPU slots.
#
# Baselines (Core Minimal Benchmark - 5 Key Paradigms):
#   1. Proposed: conditional_grl (Species-Conditioned Masked Softmax GRL)
#   2. Baseline: focal (Standard ERM with Focal Loss)
#   3. Regularization: mixup (Data Augmentation)
#   4. Adversarial: dann_unconditional (Domain Adaptation without class conditioning)
#   5. Mutual Information: club (Variational MI Bottleneck)
# ==============================================================================

set -e

BACKBONE="convnext_tiny"
FOLD=0
EPOCHS=20
BATCH_SIZE=64
METADATA="out/metadata/metadata.csv"
IMG_ROOT="out"
OUTPUT_DIR="specimen_invariance_outputs"

echo "=================================================================="
echo " Auto-Detecting GPUs & Running Baselines in Parallel              "
echo " Backbone: ${BACKBONE} | Fold: ${FOLD} | Epochs: ${EPOCHS}       "
echo " Output Dir: ${OUTPUT_DIR}                                        "
echo "=================================================================="

# Invoke the multi-GPU parallel dispatcher with DataParallel strategy
python run_parallel_dispatcher.py \
    --strategy data_parallel \
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
