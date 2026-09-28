"""
specimen_invariance_framework/run_pareto_sweep.py
=================================================
Sweeps the adversarial hyperparameter lambda_adv to generate:
1. Pareto Frontier Curve: Strict Validation Accuracy vs. Specimen Recoverability Index (SRI).
2. Correlation Analysis: Specimen Recoverability Index vs. Generalization Gap (GGSL).

Usage:
    python run_pareto_sweep.py --backbone convnext_tiny --fold 0 --epochs 30 --lambdas 0.0 0.1 0.25 0.5 1.0 2.0
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from datasets.dataset import TimberDataset, build_taxonomy_mappings
from datasets.samplers import SpecimenBalancedBatchSampler
from datasets.augmentations import build_train_transform, build_val_transform
from datasets.splits import generate_round_robin_loso_splits, generate_leaky_stratified_split
from models.full_model import SpecimenInvariantModel
from trainers.trainer_adversarial import InvarianceTrainer
from evaluation.specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
from evaluation.metrics import compute_ggsl
from evaluation.visualizer import plot_pareto_curve, plot_correlation_sri_vs_ggsl


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep lambda_adv for Pareto Frontier & Correlation")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
                        help="List of lambda_adv values to sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv")
    parser.add_argument("--image_root", type=str, default="out")
    parser.add_argument("--output_dir", type=str, default="specimen_invariance_outputs/pareto_sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Commencing lambda_adv sweep across values: {args.lambdas}")
    
    # 1. Metadata and mappings
    meta_df = pd.read_csv(args.metadata_csv)
    mappings = build_taxonomy_mappings(meta_df)
    
    # 2. Splits
    loso_folds = generate_round_robin_loso_splits(meta_df, num_folds=5, seed=args.seed)
    loso_fold = loso_folds[args.fold]
    leaky_fold = generate_leaky_stratified_split(loso_fold, seed=args.seed)
    
    train_transform = build_train_transform(image_size=224)
    val_transform = build_val_transform(image_size=224)
    
    ds_train = TimberDataset(loso_fold["train"], args.image_root, mappings, transform=train_transform)
    ds_val = TimberDataset(loso_fold["val"], args.image_root, mappings, transform=val_transform)
    ds_test_loso = TimberDataset(loso_fold["test"], args.image_root, mappings, transform=val_transform)
    ds_test_leaky = TimberDataset(leaky_fold["test"], args.image_root, mappings, transform=val_transform)
    
    loader_train = DataLoader(
        ds_train,
        batch_sampler=SpecimenBalancedBatchSampler(loso_fold["train"], batch_size=args.batch_size, samples_per_specimen=4, seed=args.seed),
        num_workers=2,
        pin_memory=True
    )
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    loader_test_loso = DataLoader(ds_test_loso, batch_size=args.batch_size, shuffle=False, num_workers=2)
    loader_test_leaky = DataLoader(ds_test_leaky, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    sweep_results = []
    
    for lam in args.lambdas:
        print(f"\n" + "=" * 60)
        print(f"  TRAINING WITH lambda_adv = {lam:.2f}")
        print("=" * 60)
        
        run_name = f"pareto_lambda_{lam:.2f}"
        run_dir = out_dir / run_name
        
        model_cfg = ModelConfig(backbone_name=args.backbone, max_lambda_adv=lam)
        train_cfg = TrainingConfig(
            method="conditional_grl",
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        
        model = SpecimenInvariantModel(
            config=model_cfg,
            specimen_counts=mappings["specimen_counts"],
            num_total_specimens=mappings["num_total_specimens"]
        )
        
        trainer = InvarianceTrainer(
            model=model,
            train_loader=loader_train,
            val_loader=loader_val,
            test_loader=loader_test_loso,
            config=train_cfg,
            model_config=model_cfg,
            device=device,
            output_dir=str(run_dir),
        )
        
        train_out = trainer.train()
        
        # Load best checkpoint
        best_ckpt = run_dir / "best_model.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            
        # Evaluate LOSO vs Leaky
        test_loso_metrics = trainer.evaluate(loader_test_loso)
        test_leaky_metrics = trainer.evaluate(loader_test_leaky)
        ggsl = compute_ggsl(test_leaky_metrics["accuracy"], test_loso_metrics["accuracy"],
                            test_leaky_metrics["macro_f1"], test_loso_metrics["macro_f1"])
        
        # Evaluate Specimen Recoverability Probe
        embs_train, sp_train, spec_train, _ = extract_all_embeddings(model, loader_train, device)
        probe_res = evaluate_specimen_recoverability(
            embeddings=embs_train,
            species_labels=sp_train,
            specimen_labels=spec_train,
            specimen_counts=mappings["specimen_counts"],
            seed=args.seed
        )
        
        record = {
            "lambda_adv": lam,
            "val_acc": train_out["test_metrics"]["accuracy"],
            "val_macro_f1": train_out["test_metrics"]["macro_f1"],
            "sri": probe_res["mean_sri"],
            "ggsl_acc": ggsl["ggsl_acc"],
            "ggsl_macro_f1": ggsl["ggsl_macro_f1"],
            "method_name": f"λ={lam:.2f}",
        }
        sweep_results.append(record)
        print(f"[Result λ={lam:.2f}] Strict Acc: {record['val_acc']:.2f}% | SRI: {record['sri']:.4f} | GGSL Acc: {record['ggsl_acc']:+.2f} pp")
        
    # Save sweep json
    with open(out_dir / "pareto_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
        
    # Plot Pareto Curve
    pareto_plot_path = out_dir / "pareto_frontier_accuracy_vs_sri.png"
    plot_pareto_curve(sweep_results, str(pareto_plot_path))
    print(f"[+] Saved Pareto Curve to: {pareto_plot_path}")
    
    # Plot Correlation
    corr_plot_path = out_dir / "correlation_sri_vs_ggsl.png"
    plot_correlation_sri_vs_ggsl(sweep_results, str(corr_plot_path))
    print(f"[+] Saved Correlation Plot to: {corr_plot_path}")
    
    print("\n[SUCCESS] Completed Pareto sweep and correlation analyses!")


if __name__ == "__main__":
    main()
