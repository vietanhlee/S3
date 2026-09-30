"""
specimen_invariance_framework/train.py
======================================
CLI entry point to train a specimen-invariant model or baseline on a given fold.

Usage example:
    python train.py --method conditional_grl --backbone convnext_tiny --fold 0 --epochs 40
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Suppress multiple OpenMP runtime initialization errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure framework directory and root repository are always in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for p in [str(CURRENT_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import DatasetConfig, ModelConfig, TrainingConfig
    from datasets.dataset import TimberDataset, build_taxonomy_mappings
    from datasets.samplers import SpecimenBalancedBatchSampler
    from datasets.augmentations import build_train_transform, build_val_transform
    from datasets.splits import generate_round_robin_loso_splits
    from models.full_model import SpecimenInvariantModel
    from trainers.trainer_adversarial import InvarianceTrainer
except (ImportError, ValueError):
    from specimen_invariance_framework.config import DatasetConfig, ModelConfig, TrainingConfig
    from specimen_invariance_framework.datasets.dataset import TimberDataset, build_taxonomy_mappings
    from specimen_invariance_framework.datasets.samplers import SpecimenBalancedBatchSampler
    from specimen_invariance_framework.datasets.augmentations import build_train_transform, build_val_transform
    from specimen_invariance_framework.datasets.splits import generate_round_robin_loso_splits
    from specimen_invariance_framework.models.full_model import SpecimenInvariantModel
    from specimen_invariance_framework.trainers.trainer_adversarial import InvarianceTrainer


def resolve_metadata_path(given_path: str) -> Path:
    p = Path(given_path)
    if p.exists() and p.is_file():
        return p.resolve()
    
    candidates = [
        Path(given_path),
        Path("..") / given_path,
        REPO_ROOT / "01_data_paper_forensic_cites" / "out" / "metadata" / "metadata.csv",
        Path("/kaggle/working/S3/01_data_paper_forensic_cites/out/metadata/metadata.csv"),
        REPO_ROOT / "out" / "metadata" / "metadata.csv",
        Path("/kaggle/working/S3/out/metadata/metadata.csv"),
        Path("../out/metadata/metadata.csv"),
        Path("out/metadata/metadata.csv"),
        Path("metadata.csv"),
        Path("../metadata.csv"),
    ]
    # 1. Check known candidates
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    # 2. Check Kaggle input folders if available
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for csv_cand in kaggle_input.glob("**/metadata.csv"):
            if csv_cand.is_file():
                return csv_cand.resolve()
            
    # 3. Recursive search in REPO_ROOT
    for csv_cand in REPO_ROOT.glob("**/metadata.csv"):
        if csv_cand.is_file():
            return csv_cand.resolve()

    # 4. Recursive search in /kaggle working directory
    kaggle_root = Path("/kaggle")
    if kaggle_root.exists():
        for csv_cand in kaggle_root.glob("**/metadata.csv"):
            if csv_cand.is_file():
                return csv_cand.resolve()
            
    raise FileNotFoundError(
        f"Could not locate metadata CSV. Looked at: {[str(c) for c in candidates]}"
    )


def resolve_image_root(given_root: str, sample_image_relpath: str) -> Path:
    p = Path(given_root)
    if (p / sample_image_relpath).exists():
        return p.resolve()
    if (p / "images" / sample_image_relpath).exists():
        return (p / "images").resolve()
        
    # 1. Search in /kaggle/input (highest priority on Kaggle where image datasets are mounted)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root_cand in kaggle_input.glob("**"):
            if root_cand.is_dir():
                if (root_cand / sample_image_relpath).exists():
                    return root_cand.resolve()
                if (root_cand / "images" / sample_image_relpath).exists():
                    return (root_cand / "images").resolve()
                if (root_cand / "out" / "images" / sample_image_relpath).exists():
                    return (root_cand / "out" / "images").resolve()
                if (root_cand / "out" / sample_image_relpath).exists():
                    return (root_cand / "out").resolve()

    candidates = [
        Path(given_root),
        Path("..") / given_root,
        REPO_ROOT / given_root,
        REPO_ROOT / "01_data_paper_forensic_cites" / "out",
        REPO_ROOT / "01_data_paper_forensic_cites" / "out" / "images",
        Path("/kaggle/working/S3/01_data_paper_forensic_cites/out"),
        Path("/kaggle/working/S3/01_data_paper_forensic_cites/out/images"),
        REPO_ROOT / "out",
        REPO_ROOT / "out" / "images",
        Path("/kaggle/working/S3/out"),
        Path("/kaggle/working/S3/out/images"),
        Path("../out"),
        Path("../out/images"),
        Path("out"),
        Path("out/images"),
    ]
    for cand in candidates:
        if (cand / sample_image_relpath).exists():
            return cand.resolve()
        if (cand / "images" / sample_image_relpath).exists():
            return (cand / "images").resolve()
            
    # Fallback to local out directory
    fallback = (REPO_ROOT / "01_data_paper_forensic_cites" / "out").resolve() if (REPO_ROOT / "01_data_paper_forensic_cites" / "out").exists() else p.resolve()
    print(f"[*] Note: sample image '{sample_image_relpath}' not directly verified on disk. Defaulting image_root to: {fallback}")
    return fallback


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Specimen-Invariant Wood ID Models")
    parser.add_argument("--method", type=str, default="conditional_grl",
                        choices=["conditional_grl", "dann_unconditional", "club", "group_dro",
                                 "irm", "focal", "arcface", "strong_reg", "mixup", "supcon",
                                 "semihard_triplet", "frozen_linear", "ce"],
                        help="Training methodology or baseline")
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "resnet50", "tf_efficientnetv2_s", "swin_t"],
                        help="Visual backbone architecture")
    parser.add_argument("--fold", type=int, default=0, help="Round-robin LOSO fold index [0..R-1]")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of round-robin folds")
    parser.add_argument("--epochs", type=int, default=13, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate for visual backbone")
    parser.add_argument("--lr_head", type=float, default=5e-4, help="Learning rate for classification/adversarial heads")
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Max adversarial weight lambda_adv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv", help="Path to master metadata CSV")
    parser.add_argument("--image_root", type=str, default="out", help="Path to image directory root")
    parser.add_argument("--output_base_dir", type=str, default="specimen_invariance_outputs", help="Output directory")
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU index to use (e.g. 0, 1). If None, uses default cuda/cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if args.gpu is not None and args.gpu < device_count:
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(device)
            use_data_parallel = False
            print(f"[*] Executing on single pinned device: {device} | Seed: {args.seed} | Method: {args.method} | Backbone: {args.backbone}")
        else:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            use_data_parallel = (device_count > 1)
            if use_data_parallel:
                print(f"[+] Multi-GPU DataParallel Activated: Concurrently training 1 model across all {device_count} GPUs! | Method: {args.method}")
            else:
                print(f"[*] Executing on single device: {device} | Seed: {args.seed} | Method: {args.method} | Backbone: {args.backbone}")
    else:
        device = torch.device("cpu")
        use_data_parallel = False
        print(f"[*] Executing on device: cpu | Seed: {args.seed} | Method: {args.method} | Backbone: {args.backbone}")
    
    # 1. Load metadata
    meta_path = resolve_metadata_path(args.metadata_csv)
    print(f"[+] Ingesting metadata from: {meta_path}")
    meta_df = pd.read_csv(meta_path)
    
    # Resolve valid image_root
    sample_img_rel = meta_df["image_path"].iloc[0] if "image_path" in meta_df.columns else ""
    image_root = resolve_image_root(args.image_root, sample_img_rel)
    print(f"[+] Ingesting image directory from: {image_root}")
    
    # 2. Build taxonomy and specimen mappings
    mappings = build_taxonomy_mappings(meta_df)
    print(f"[+] Total species: {mappings['num_species']} | Total physical specimens: {mappings['num_total_specimens']}")
    
    # 3. Generate Round-Robin LOSO splits
    loso_folds = generate_round_robin_loso_splits(meta_df, num_folds=args.num_folds, seed=args.seed)
    selected_fold = loso_folds[args.fold]
    df_train = selected_fold["train"]
    df_val = selected_fold["val"]
    df_test = selected_fold["test"]
    
    print(f"[+] Fold {args.fold}: Train={len(df_train):,} | Val (Strict)={len(df_val):,} | Test (Strict)={len(df_test):,}")
    
    # 4. Datasets and Transforms
    train_transform = build_train_transform(image_size=224)
    val_transform = build_val_transform(image_size=224)
    
    train_dataset = TimberDataset(df_train, image_root=str(image_root), mappings=mappings, transform=train_transform)
    val_dataset = TimberDataset(df_val, image_root=str(image_root), mappings=mappings, transform=val_transform)
    test_dataset = TimberDataset(df_test, image_root=str(image_root), mappings=mappings, transform=val_transform)
    
    # 5. Specimen-balanced sampler for training
    train_sampler = SpecimenBalancedBatchSampler(
        df_train,
        batch_size=args.batch_size,
        samples_per_specimen=4,
        shuffle=True,
        seed=args.seed
    )
    
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    # 6. Configurations
    model_cfg = ModelConfig(
        backbone_name=args.backbone,
        max_lambda_adv=args.lambda_adv,
        species_head_type="arcface" if args.method == "arcface" else "linear",
    )
    
    train_cfg = TrainingConfig(
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        seed=args.seed,
    )
    
    # 7. Model instantiation
    model = SpecimenInvariantModel(
        config=model_cfg,
        specimen_counts=mappings["specimen_counts"],
        num_total_specimens=mappings["num_total_specimens"]
    )
    if use_data_parallel:
        model = nn.DataParallel(model)
        print(f"[+] Model wrapped in torch.nn.DataParallel across all visible GPUs.")
    
    # 8. Training loop
    run_dir = Path(args.output_base_dir) / f"{args.method}_{args.backbone}_fold{args.fold}_seed{args.seed}"
    trainer = InvarianceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_cfg,
        model_config=model_cfg,
        device=device,
        output_dir=str(run_dir),
    )
    
    results = trainer.train()
    
    # Save training record
    summary_path = run_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": args.method,
            "backbone": args.backbone,
            "fold": args.fold,
            "seed": args.seed,
            "best_epoch": results["best_epoch"],
            "best_val_macro_f1": results["best_val_metric"],
            "test_metrics": results["test_metrics"],
        }, f, indent=2)
        
    print(f"\n[SUCCESS] Completed run for {args.method}. Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
