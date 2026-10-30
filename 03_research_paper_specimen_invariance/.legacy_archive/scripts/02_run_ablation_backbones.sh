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

METHOD="conditional_grl_club"
FOLD=0
EPOCHS=13
BATCH_SIZE=64

# Tự động nhận diện đường dẫn metadata và ảnh trên Kaggle hoặc Local
if [ -f "/kaggle/working/S3/01_data_paper_forensic_cites/out/metadata/metadata.csv" ]; then
    METADATA="/kaggle/working/S3/01_data_paper_forensic_cites/out/metadata/metadata.csv"
elif [ -f "01_data_paper_forensic_cites/out/metadata/metadata.csv" ]; then
    METADATA="01_data_paper_forensic_cites/out/metadata/metadata.csv"
else
    METADATA="out/metadata/metadata.csv"
fi

if [ -d "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" ]; then
    IMG_ROOT="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
else
    IMG_ROOT="out"
fi

OUTPUT_DIR="specimen_invariance_outputs"

echo "=================================================================="
echo " Auto-Detecting GPUs & Running Backbone Ablations in Parallel     "
echo " Method: ${METHOD} | Fold: ${FOLD} | Epochs: ${EPOCHS}            "
echo " Output Dir: ${OUTPUT_DIR}                                        "
echo "=================================================================="

# Invoke the multi-GPU parallel dispatcher for backbones with DataParallel
python run_parallel_dispatcher.py \
    --strategy data_parallel \
    --mode backbones \
    --method "${METHOD}" \
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
