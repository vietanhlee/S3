"""
specimen_invariance_framework/evaluate.py
=========================================
Comprehensive Module 3 Evaluation Script:
1. Calculates Generalization Gap from Specimen Leakage (GGSL): Acc_leaky - Acc_LOSO.
2. Trains 50/50 per-specimen probe to compute Specimen Recoverability Index (SRI).
3. Evaluates Model Calibration (ECE, MCE, Reliability Diagram).
4. Generates side-by-side t-SNE projections (Colored by Species vs. Colored by Specimen).

Usage:
    python evaluate.py --checkpoint specimen_invariance_outputs/conditional_grl_convnext_tiny_fold0_seed42/best_model.pth --fold 0
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import ModelConfig
from datasets.dataset import TimberDataset, build_taxonomy_mappings
from datasets.augmentations import build_val_transform
from datasets.splits import generate_round_robin_loso_splits, generate_leaky_stratified_split
from models.full_model import SpecimenInvariantModel
from evaluation.specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
from evaluation.calibration import compute_calibration_metrics, plot_reliability_diagram
from evaluation.metrics import compute_ggsl
from evaluation.visualizer import plot_tsne_species_vs_specimen


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Specimen-Invariance and Generalization Metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone architecture")
    parser.add_argument("--fold", type=int, default=0, help="Round-robin fold index")
    parser.add_argument("--num_folds", type=int, default=5, help="Total round-robin folds")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv", help="Metadata CSV path")
    parser.add_argument("--image_root", type=str, default="out", help="Image root folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to checkpoint parent)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Commencing Module 3 Evaluation for checkpoint: {ckpt_path}")
    
    # 1. Metadata and mappings
    meta_path = Path(args.metadata_csv)
    if not meta_path.exists():
        for c in [Path("../out/metadata/metadata.csv"), Path("metadata.csv")]:
            if c.exists():
                meta_path = c
                break
    meta_df = pd.read_csv(meta_path)
    mappings = build_taxonomy_mappings(meta_df)
    
    # 2. Recreate LOSO fold and matched Leaky Stratified split
    loso_folds = generate_round_robin_loso_splits(meta_df, num_folds=args.num_folds, seed=args.seed)
    loso_fold = loso_folds[args.fold]
    leaky_fold = generate_leaky_stratified_split(loso_fold, seed=args.seed)
    
    val_transform = build_val_transform(image_size=224)
    
    test_loso_dataset = TimberDataset(loso_fold["test"], image_root=args.image_root, mappings=mappings, transform=val_transform)
    test_leaky_dataset = TimberDataset(leaky_fold["test"], image_root=args.image_root, mappings=mappings, transform=val_transform)
    train_loso_dataset = TimberDataset(loso_fold["train"], image_root=args.image_root, mappings=mappings, transform=val_transform)
    
    loader_loso = DataLoader(test_loso_dataset, batch_size=64, shuffle=False, num_workers=2)
    loader_leaky = DataLoader(test_leaky_dataset, batch_size=64, shuffle=False, num_workers=2)
    loader_train = DataLoader(train_loso_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 3. Load Model
    model_cfg = ModelConfig(backbone_name=args.backbone)
    model = SpecimenInvariantModel(
        config=model_cfg,
        specimen_counts=mappings["specimen_counts"],
        num_total_specimens=mappings["num_total_specimens"]
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("[+] Model loaded successfully from checkpoint.")
    
    # 4. Evaluate LOSO Test (Strictly Unseen Specimens)
    def run_eval(loader):
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                targets = batch["species_idx"].to(device)
                logits = model(images)["species_logits"]
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        acc = float(accuracy_score(all_targets, all_preds) * 100.0)
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro") * 100.0)
        return acc, macro_f1, np.array(all_probs), np.array(all_preds), np.array(all_targets)
        
    acc_loso, f1_loso, probs_loso, preds_loso, targets_loso = run_eval(loader_loso)
    acc_leaky, f1_leaky, _, _, _ = run_eval(loader_leaky)
    
    # 5. Compute GGSL
    ggsl = compute_ggsl(acc_leaky, acc_loso, f1_leaky, f1_loso)
    print(f"\n[1] GENERALIZATION GAP FROM SPECIMEN LEAKAGE (GGSL):")
    print(f"    - Leaky Stratified Acc : {acc_leaky:.2f}% | Macro-F1: {f1_leaky:.2f}%")
    print(f"    - Strict LOSO Test Acc : {acc_loso:.2f}% | Macro-F1: {f1_loso:.2f}%")
    print(f"    - GGSL (Acc Inflation) : {ggsl['ggsl_acc']:+.2f} pp")
    print(f"    - GGSL (F1 Inflation)  : {ggsl['ggsl_macro_f1']:+.2f} pp")
    
    # 6. Specimen Recoverability Probe
    print(f"\n[2] RUNNING SPECIMEN RECOVERABILITY PROBE (50/50 per-specimen split):")
    train_embs, train_species, train_specimens, _ = extract_all_embeddings(model, loader_train, device)
    probe_results = evaluate_specimen_recoverability(
        embeddings=train_embs,
        species_labels=train_species,
        specimen_labels=train_specimens,
        specimen_counts=mappings["specimen_counts"],
        seed=args.seed
    )
    print(f"    - Mean Specimen Recoverability Index (SRI): {probe_results['mean_sri']:.4f} (0 = Invariant, 1 = Leaked)")
    
    # 7. Model Calibration (ECE & Reliability Diagram)
    print(f"\n[3] COMPUTING MODEL CALIBRATION (ECE, MCE):")
    confs_loso = np.max(probs_loso, axis=-1)
    calib_results = compute_calibration_metrics(confs_loso, preds_loso, targets_loso, num_bins=15)
    print(f"    - Expected Calibration Error (ECE) : {calib_results['ece']:.2f}%")
    print(f"    - Maximum Calibration Error (MCE)  : {calib_results['mce']:.2f}%")
    
    diagram_path = out_dir / "reliability_diagram.png"
    plot_reliability_diagram(calib_results, str(diagram_path), title=f"Reliability Diagram (LOSO Test - ECE: {calib_results['ece']:.2f}%)")
    print(f"    - Saved Reliability Diagram to: {diagram_path}")
    
    # 8. Dual t-SNE Visualization
    print(f"\n[4] GENERATING DUAL t-SNE PROJECTIONS (Species vs. Specimen):")
    tsne_path = out_dir / "tsne_species_vs_specimen.png"
    plot_tsne_species_vs_specimen(
        embeddings=train_embs,
        species_labels=train_species,
        specimen_labels=train_specimens,
        save_path=str(tsne_path),
        title_prefix=args.backbone
    )
    print(f"    - Saved Dual t-SNE plot to: {tsne_path}")
    
    # 9. Save JSON Evaluation Report
    report = {
        "checkpoint": str(ckpt_path),
        "backbone": args.backbone,
        "fold": args.fold,
        "metrics_loso": {"accuracy": acc_loso, "macro_f1": f1_loso},
        "metrics_leaky": {"accuracy": acc_leaky, "macro_f1": f1_leaky},
        "ggsl": ggsl,
        "probe": probe_results,
        "calibration": {
            "ece": calib_results["ece"],
            "mce": calib_results["mce"],
        },
    }
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[SUCCESS] Comprehensive evaluation complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()
