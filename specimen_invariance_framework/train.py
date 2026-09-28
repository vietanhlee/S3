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
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import DatasetConfig, ModelConfig, TrainingConfig
from datasets.dataset import TimberDataset, build_taxonomy_mappings
from datasets.samplers import SpecimenBalancedBatchSampler
from datasets.augmentations import build_train_transform, build_val_transform
from datasets.splits import generate_round_robin_loso_splits
from models.full_model import SpecimenInvariantModel
from trainers.trainer_adversarial import InvarianceTrainer


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
    parser.add_argument("--epochs", type=int, default=17, help="Training epochs")
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
    
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing on device: {device} | Seed: {args.seed} | Method: {args.method} | Backbone: {args.backbone}")
    
    # 1. Load metadata
    meta_path = Path(args.metadata_csv)
    if not meta_path.exists():
        # Fallback to local project search
        candidates = [Path("../out/metadata/metadata.csv"), Path("metadata.csv")]
        for c in candidates:
            if c.exists():
                meta_path = c
                break
    print(f"[+] Ingesting metadata from: {meta_path}")
    meta_df = pd.read_csv(meta_path)
    
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
    
    train_dataset = TimberDataset(df_train, image_root=args.image_root, mappings=mappings, transform=train_transform)
    val_dataset = TimberDataset(df_val, image_root=args.image_root, mappings=mappings, transform=val_transform)
    test_dataset = TimberDataset(df_test, image_root=args.image_root, mappings=mappings, transform=val_transform)
    
    # 5. Specimen-balanced sampler for training
    train_sampler = SpecimenBalancedBatchSampler(
        df_train,
        batch_size=args.batch_size,
        samples_per_specimen=4,
        shuffle=True,
        seed=args.seed
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
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
