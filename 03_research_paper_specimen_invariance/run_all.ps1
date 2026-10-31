# ==============================================================================
# run_all.ps1
# Master 1-Click Multi-Seed PowerShell Script for Specimen-Invariant GAN Timber Benchmark
# Directly scans: /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3
# Evaluates 3 Seeds (42, 123, 456) with Dynamic Per-Seed Holdout Partitioning
# ==============================================================================

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SPECIMEN-INVARIANT TIMBER FORENSICS: 3-SEED DUAL-PROTOCOL BENCHMARK" -ForegroundColor Cyan
Write-Host "  Clean 11-Species Dataset (110 physical vouchers, 10 vouchers/species)" -ForegroundColor Cyan
Write-Host "  Direct Path: /kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" -ForegroundColor Cyan
Write-Host "  Seeds: 42, 123, 456 | Epochs: 20 | Batch Size: 64" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

python run_benchmark_3seeds.py `
    --image_root "/kaggle/input/datasets/b23dckh002lvitanh/s3-origin/S3" `
    --seeds 42 123 456 `
    --epochs 20 `
    --batch_size 64 `
    --lr 3e-4 `
    --adv_weight 0.05 `
    --club_weight 0.005 `
    --backbone "convnext_tiny"

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Multi-Seed Benchmark completed successfully!" -ForegroundColor Green
Write-Host "Check publication report at: $ScriptDir\EXPERIMENT_RESULTS_REPORT.md" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
