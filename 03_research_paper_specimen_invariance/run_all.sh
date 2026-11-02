#!/usr/bin/env bash
# ==============================================================================
# run_all.sh
# Master 1-Click Multi-Seed Execution Script for Specimen-Invariant Timber Benchmark
# Directly scans: /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
# Evaluates 10 Seeds (Default) with Dynamic Per-Seed Holdout Partitioning
# Evaluates: Naive Random Leakage Audit, Val (11), Seen Test (44), and Unseen Test (55)
# Backbone: ResNet-18 (Default) | Device: CUDA (Kaggle T4 / P100)
# ==============================================================================

set -e
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "  VOUCHER-AWARE TIMBER FORENSICS BENCHMARK & SPECIMEN LEAKAGE AUDIT"
echo "  Protocol: Stress-Test 50% Specimen Holdout (10 Random Splits)"
echo "  Splits  : Train (44 vouchers) | Val (11 vouchers) | Test (55 vouchers = 5/sp)"
echo "  Taxa    : 11 Clean Fabaceae Species (110 physical vouchers)"
echo "  Backbone: ResNet-18 (Default) | Device: CUDA (Kaggle T4/P100)"
echo "  Audit   : Naive Image-Level Random Split vs Out-of-Specimen Voucher Split"
echo "  Direct  : /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
echo "  Configs : adv_weight=0.05, club_weight=0.001, label_smoothing=0.05"
echo "================================================================================"

python run_benchmark.py \
    --image_root "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" \
    --num_seeds 10 \
    --test_vouchers_per_species 5 \
    --val_vouchers_per_species 1 \
    --epochs 17 \
    --batch_size 64 \
    --lr 3e-4 \
    --adv_weight 0.05 \
    --club_weight 0.001 \
    --label_smoothing 0.05 \
    --backbone "resnet18" \
    "$@"

echo ""
echo "================================================================================"
echo "[SUCCESS] 10-Seed Voucher Benchmark & Leakage Audit completed successfully!"
echo "Check publication report at: $SCRIPT_DIR/EXPERIMENT_RESULTS_REPORT.md"
echo "================================================================================"
