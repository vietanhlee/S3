#!/usr/bin/env bash
# ==============================================================================
# 05_run_ablation_study.sh
# ==============================================================================
# Executes all 7 Ablation configurations for Paper Table 3:
#   1. Baseline (Focal Loss only)
#   2. Full Framework w/o Masked Softmax (Global DANN)
#   3. Full Framework w/o Annealing (Constant lambda=1.0)
#   4. Full Framework w/o CLUB Bottleneck (GRL only)
#   5. Full Framework w/o GRL (CLUB only)
#   6. Full Framework w/o Specimen-Balanced Sampler
#   7. Proposed Full Framework (Conditional GRL + CLUB Bottleneck)
# ==============================================================================

set -e

BACKBONE="convnext_tiny"
FOLD=0
EPOCHS=13
BATCH_SIZE=64

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

OUTPUT_DIR="specimen_invariance_outputs/ablation_study"

echo "=================================================================="
echo " Starting Comprehensive Ablation Study (Table 3)                  "
echo " Backbone: ${BACKBONE} | Fold: ${FOLD} | Epochs: ${EPOCHS}        "
echo " Output Dir: ${OUTPUT_DIR}                                        "
echo "=================================================================="

python run_ablation_study.py \
    --backbone "${BACKBONE}" \
    --fold "${FOLD}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --metadata_csv "${METADATA}" \
    --image_root "${IMG_ROOT}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo " Ablation study complete! Artifacts saved in: ${OUTPUT_DIR}        "
echo " Check ${OUTPUT_DIR}/ablation_table.tex for Paper Table 3         "
echo "=================================================================="
