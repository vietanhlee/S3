#!/usr/bin/env bash
# ==============================================================================
# run_all.sh
# Master 1-Click Multi-Seed Execution Script for Specimen-Invariant GAN Timber Benchmark
# Directly scans: /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
# Evaluates 4 Seeds (42, 123, 456, 789) with Dynamic Per-Seed Holdout Partitioning
# Evaluates across 3 sets: Val, Seen Test, and Out-of-Specimen Unseen Test
# ==============================================================================

set -e
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "  SPECIMEN-INVARIANT TIMBER FORENSICS: 4-SEED BENCHMARK"
echo "  Protocol: Stress-Test 50% Specimen Holdout"
echo "  Splits  : Train (44 vouchers) | Val (11 vouchers) | Test (55 vouchers = 5/sp)"
echo "  Taxa    : 11 Clean Fabaceae Species (110 physical vouchers)"
echo "  Backbone: ResNet-18 (Default) | Device: CUDA (Kaggle T4/P100)"
echo "  Direct  : /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3"
echo "  Seeds   : 42, 123, 456, 789 | Epochs: 20 | Batch Size: 64"
echo "================================================================================"

python run_benchmark.py \
    --image_root "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" \
    --seeds 42 123 456 789 \
    --test_vouchers_per_species 5 \
    --val_vouchers_per_species 1 \
    --epochs 20 \
    --batch_size 64 \
    --lr 3e-4 \
    --adv_weight 0.05 \
    --club_weight 0.001 \
    --label_smoothing 0.05 \
    --backbone "resnet18"

echo ""
echo "================================================================================"
echo "[SUCCESS] Multi-Seed Benchmark completed successfully!"
echo "Check publication report at: $SCRIPT_DIR/EXPERIMENT_RESULTS_REPORT.md"
echo "================================================================================"
