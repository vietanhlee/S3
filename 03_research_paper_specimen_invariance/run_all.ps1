# ==============================================================================
# run_all.ps1
# Master 1-Click PowerShell Script for Specimen-Invariant GAN Timber Benchmark
# Runs: Data Preparation -> Train Baseline -> Train GAN -> Dual Evaluation -> Report
# ==============================================================================

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SPECIMEN-INVARIANT TIMBER FORENSICS: 1-CLICK DUAL-PROTOCOL BENCHMARK" -ForegroundColor Cyan
Write-Host "  Clean 11-Species Dataset (110 physical vouchers, 3,702 images) | Seed: 42" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# Step 1: Prepare clean 11-species metadata and dual-split partitions
Write-Host "`n[STEP 1/4] Preparing clean 11-species dataset & dual-split partitions..." -ForegroundColor Yellow
python prepare_dataset.py --seed 42

# Step 2: Train Standard Baseline Model (ERM)
Write-Host "`n[STEP 2/4] Training Standard Baseline Model (ConvNeXt-Tiny - ERM)..." -ForegroundColor Yellow
python train_baseline.py --epochs 20 --batch_size 64 --seed 42

# Step 3: Train Proposed Adversarial Invariant 'GAN' Model
Write-Host "`n[STEP 3/4] Training Proposed Adversarial Invariant 'GAN' Model (GRL + CLUB)..." -ForegroundColor Yellow
python train_gan.py --epochs 20 --batch_size 64 --adv_weight 0.30 --club_weight 0.05 --seed 42

# Step 4: Run Dual-Protocol Evaluation & Generate Markdown Report
Write-Host "`n[STEP 4/4] Running Dual-Protocol Audit (Seen vs. Unseen, SRI Probes, ECE)..." -ForegroundColor Yellow
python evaluate_dual.py --seed 42

Write-Host "`n[*] Generating final Markdown Report..." -ForegroundColor Yellow
python generate_report.py

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Benchmark completed successfully!" -ForegroundColor Green
Write-Host "Check detailed report at: 03_research_paper_specimen_invariance/EXPERIMENT_RESULTS_REPORT.md" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
