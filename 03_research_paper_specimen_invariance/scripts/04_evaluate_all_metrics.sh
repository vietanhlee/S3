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

BASE_DIR="specimen_invariance_outputs"

METHODS=(
    "focal"
    "mixup"
    "dann_unconditional"
    "club"
    "conditional_grl"
    "conditional_grl_club"
)

echo "=================================================================="
echo " Starting Full Module 3 Evaluation Across Trained Models          "
echo " Base Dir: ${BASE_DIR} | Fold: ${FOLD}                            "
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

# Evaluate Additional Backbone Ablations (resnet50, tf_efficientnetv2_s, swin_t)
EXTRA_BACKBONES=(
    "resnet50"
    "tf_efficientnetv2_s"
    "swin_t"
)

for BB in "${EXTRA_BACKBONES[@]}"; do
    # Check conditional_grl_club first, fallback to conditional_grl
    RUN_DIR="${BASE_DIR}/conditional_grl_club_${BB}_fold${FOLD}_seed42"
    if [ ! -d "${RUN_DIR}" ]; then
        RUN_DIR="${BASE_DIR}/conditional_grl_${BB}_fold${FOLD}_seed42"
    fi
    CKPT="${RUN_DIR}/best_model.pth"
    
    if [ -f "${CKPT}" ]; then
        echo ""
        echo ">>> Evaluating Backbone Ablation: ${BB} (${CKPT}) <<<"
        python evaluate.py \
            --checkpoint "${CKPT}" \
            --backbone "${BB}" \
            --fold "${FOLD}" \
            --metadata_csv "${METADATA}" \
            --image_root "${IMG_ROOT}" \
            --output_dir "${RUN_DIR}/eval_results"
    fi
done

echo ""
echo "=================================================================="
echo " Aggregating All Results into Paper Tables (LaTeX)                "
echo "=================================================================="
python collect_all_benchmark_results.py --output_base_dir "${BASE_DIR}"

echo ""
echo "=================================================================="
echo " Evaluation and Master Table Aggregation Complete!               "
echo " Tables saved in: ${BASE_DIR}/summary_tables                      "
echo "=================================================================="
