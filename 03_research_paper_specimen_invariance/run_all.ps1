# ==============================================================================
# run_all.ps1
# Master 1-Click Multi-Seed PowerShell Script for Specimen-Invariant GAN Timber Benchmark
# Directly scans: /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
# Evaluates 4 Seeds (42, 123, 456, 789) with Dynamic Per-Seed Holdout Partitioning
# ==============================================================================

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SPECIMEN-INVARIANT TIMBER FORENSICS: 4-SEED BENCHMARK" -ForegroundColor Cyan
Write-Host "  Protocol: Stress-Test 50% Specimen Holdout" -ForegroundColor Cyan
Write-Host "  Splits  : Train (44 vouchers) | Val (11 vouchers) | Test (55 vouchers = 5/sp)" -ForegroundColor Cyan
Write-Host "  Taxa    : 11 Clean Fabaceae Species (110 physical vouchers)" -ForegroundColor Cyan
Write-Host "  Backbone: ResNet-18 (Lightweight) | Device: CUDA" -ForegroundColor Cyan
Write-Host "  Direct  : /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" -ForegroundColor Cyan
Write-Host "  Seeds   : 42, 123, 456, 789 | Epochs: 20 | Batch Size: 64" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

python run_benchmark.py `
    --image_root "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" `
    --seeds 42 123 456 789 `
    --test_vouchers_per_species 5 `
    --val_vouchers_per_species 1 `
    --epochs 20 `
    --batch_size 64 `
    --lr 3e-4 `
    --adv_weight 0.30 `
    --club_weight 0.010 `
    --backbone "resnet18"

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Multi-Seed Benchmark completed successfully!" -ForegroundColor Green
Write-Host "Check publication report at: $ScriptDir\EXPERIMENT_RESULTS_REPORT.md" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
