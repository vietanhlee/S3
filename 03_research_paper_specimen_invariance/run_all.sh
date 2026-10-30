#!/usr/bin/env bash
# ==============================================================================
# run_all.sh
# Master 1-Click Execution Script for Specimen-Invariant GAN Timber Benchmark
# Runs: Data Preparation -> Train Baseline -> Train GAN -> Dual Evaluation -> Report
# ==============================================================================

set -e
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "  SPECIMEN-INVARIANT TIMBER FORENSICS: 1-CLICK DUAL-PROTOCOL BENCHMARK"
echo "  Clean 11-Species Dataset (110 physical vouchers, 3,702 images) | Seed: 42"
echo "================================================================================"

# Step 1: Prepare clean 11-species metadata and dual-split partitions
echo ""
echo "[STEP 1/4] Preparing clean 11-species dataset & dual-split partitions..."
python prepare_dataset.py --seed 42

# Step 2: Train Standard Baseline Model (ERM)
echo ""
echo "[STEP 2/4] Training Standard Baseline Model (ConvNeXt-Tiny - ERM)..."
python train_baseline.py --epochs 20 --batch_size 64 --seed 42

# Step 3: Train Proposed Adversarial Invariant 'GAN' Model
echo ""
echo "[STEP 3/4] Training Proposed Adversarial Invariant 'GAN' Model (GRL + CLUB)..."
python train_gan.py --epochs 20 --batch_size 64 --adv_weight 0.30 --club_weight 0.05 --seed 42

# Step 4: Run Dual-Protocol Evaluation & Generate Markdown Report
echo ""
echo "[STEP 4/4] Running Dual-Protocol Audit (Seen vs. Unseen, SRI Probes, ECE)..."
python evaluate_dual.py --seed 42

echo ""
echo "[*] Generating final Markdown Report..."
python generate_report.py

echo ""
echo "================================================================================"
echo "[SUCCESS] Benchmark completed successfully!"
echo "Check detailed report at: 03_research_paper_specimen_invariance/EXPERIMENT_RESULTS_REPORT.md"
echo "================================================================================"
