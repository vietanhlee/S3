#!/usr/bin/env bash
# ==============================================================================
# 02_run_ablation_backbones.sh
# ==============================================================================
# Automatically detects all available GPUs and trains the proposed conditional
# GRL model across 4 visual backbones concurrently:
#   1. ConvNeXt-Tiny (Primary modern CNN)
#   2. ResNet-50 (Classical residual CNN)
#   3. EfficientNetV2-S (Compound scaling CNN)
#   4. Swin-T (Hierarchical Vision Transformer)
# ==============================================================================

set -e

FOLD=0
EPOCHS=20
BATCH_SIZE=64
METADATA="out/metadata/metadata.csv"
IMG_ROOT="out"
OUTPUT_DIR="specimen_invariance_outputs"

echo "=================================================================="
echo " Auto-Detecting GPUs & Running Backbone Ablations in Parallel     "
echo " Fold: ${FOLD} | Epochs: ${EPOCHS}                                "
echo " Output Dir: ${OUTPUT_DIR}                                        "
echo "=================================================================="

# Invoke the multi-GPU parallel dispatcher for backbones
python run_parallel_dispatcher.py \
    --mode backbones \
    --fold "${FOLD}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --metadata_csv "${METADATA}" \
    --image_root "${IMG_ROOT}" \
    --output_base_dir "${OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo " All backbone ablations completed!                                "
echo "=================================================================="
