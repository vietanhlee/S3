#!/usr/bin/env bash
# ==============================================================================
# 03_run_pareto_sweep.sh
# ==============================================================================
# Sweeps the adversarial weighting parameter lambda_adv in [0.0, 2.0] to map:
#   1. Pareto frontier between strict accuracy and Specimen Recoverability Index (SRI)
#   2. Linear correlation between SRI and Generalization Gap (GGSL)
# ==============================================================================

set -e

METHOD="conditional_grl_club"
BACKBONE="convnext_tiny"
FOLD=0
EPOCHS=13
BATCH_SIZE=64
METADATA="/kaggle/working/S3/01_data_paper_forensic_cites/out/metadata/metadata.csv"
IMG_ROOT="/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
OUTPUT_DIR="specimen_invariance_outputs/pareto_sweep"

echo "=================================================================="
echo " Starting Pareto & Sensitivity Sweep (Table 4)                    "
echo " Method: ${METHOD} | Backbone: ${BACKBONE} | Epochs: ${EPOCHS}    "
echo " Output Dir: ${OUTPUT_DIR}                                        "
echo "=================================================================="

# Chạy sweep 7 điểm cấu hình tương ứng Table 4 trong bài báo (beta trong [0.2, 2.0], mu trong [0.05, 0.20])
python run_pareto_sweep.py \
    --method "${METHOD}" \
    --backbone "${BACKBONE}" \
    --fold "${FOLD}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --sweep_table4 \
    --metadata_csv "${METADATA}" \
    --image_root "${IMG_ROOT}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo " Pareto sweep complete! Figures and LaTeX saved in: ${OUTPUT_DIR} "
echo " Check ${OUTPUT_DIR}/pareto_sensitivity_table.tex for paper Table 4"
echo "=================================================================="
