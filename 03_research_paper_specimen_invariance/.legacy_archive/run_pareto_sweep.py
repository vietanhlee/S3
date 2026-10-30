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
import os
import sys
from pathlib import Path

# Suppress multiple OpenMP runtime initialization errors and Hugging Face Hub unauthenticated warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
    from config import ModelConfig, TrainingConfig, safe_load_checkpoint
    from datasets.dataset import TimberDataset, build_taxonomy_mappings
    from datasets.samplers import SpecimenBalancedBatchSampler
    from datasets.augmentations import build_train_transform, build_val_transform
    from datasets.splits import generate_round_robin_loso_splits, generate_leaky_stratified_split
    from models.full_model import SpecimenInvariantModel
    from trainers.trainer_adversarial import InvarianceTrainer
    from evaluation.specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
    from evaluation.metrics import compute_ggsl
    from evaluation.visualizer import plot_pareto_curve, plot_correlation_sri_vs_ggsl
except (ImportError, ValueError):
    from specimen_invariance_framework.config import ModelConfig, TrainingConfig, safe_load_checkpoint
    from specimen_invariance_framework.datasets.dataset import TimberDataset, build_taxonomy_mappings
    from specimen_invariance_framework.datasets.samplers import SpecimenBalancedBatchSampler
    from specimen_invariance_framework.datasets.augmentations import build_train_transform, build_val_transform
    from specimen_invariance_framework.datasets.splits import generate_round_robin_loso_splits, generate_leaky_stratified_split
    from specimen_invariance_framework.models.full_model import SpecimenInvariantModel
    from specimen_invariance_framework.trainers.trainer_adversarial import InvarianceTrainer
    from specimen_invariance_framework.evaluation.specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
    from specimen_invariance_framework.evaluation.metrics import compute_ggsl
    from specimen_invariance_framework.evaluation.visualizer import plot_pareto_curve, plot_correlation_sri_vs_ggsl


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

    raise FileNotFoundError(f"Could not locate metadata CSV. Looked at: {[str(c) for c in candidates]}")


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
            
    fallback = (REPO_ROOT / "01_data_paper_forensic_cites" / "out").resolve() if (REPO_ROOT / "01_data_paper_forensic_cites" / "out").exists() else p.resolve()
    return fallback


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep lambda_adv (beta) and club_weight (mu) for Pareto & Sensitivity")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--method", type=str, default="conditional_grl_club",
                        choices=["conditional_grl_club", "conditional_grl", "club"],
                        help="Training method to sweep")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0],
                        help="List of lambda_adv (beta) values to sweep")
    parser.add_argument("--club_weights", nargs="+", type=float, default=[0.10],
                        help="List of club_weight (mu) values to sweep")
    parser.add_argument("--sweep_table4", action="store_true",
                        help="Execute exactly the 7 canonical (beta, mu) grid points defined in Paper Table 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv")
    parser.add_argument("--image_root", type=str, default="out")
    parser.add_argument("--output_dir", type=str, default="specimen_invariance_outputs/pareto_sweep")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model_with_calibration(model: nn.Module, loader: DataLoader, device: torch.device, num_bins: int = 15):
    """
    Evaluates classification accuracy, macro F1, and ECE/MCE calibration errors.
    """
    model.eval()
    all_confs = []
    all_preds = []
    all_targets = []
    
    feature_extractor = model.module if hasattr(model, "module") else model
    
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["species_idx"].to(device)
        out = feature_extractor(images)
        logits = out["species_logits"]
        probs = torch.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)
        all_confs.extend(confs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    all_confs = np.array(all_confs, dtype=np.float32)
    all_preds = np.array(all_preds, dtype=np.int64)
    all_targets = np.array(all_targets, dtype=np.int64)
    
    acc = float(np.mean(all_preds == all_targets) * 100.0)
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0)
    
    from evaluation.calibration import compute_calibration_metrics
    calib = compute_calibration_metrics(all_confs, all_preds, all_targets, num_bins=num_bins)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece": float(calib["ece"] * 100.0),
        "mce": float(calib["mce"] * 100.0),
    }


def main():
    args = parse_args()
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            device = torch.device("cuda")
            use_data_parallel = True
            print(f"[+] Multi-GPU DataParallel Activated across all {num_devices} GPUs!")
        else:
            device = torch.device("cuda:0")
            use_data_parallel = False
            print(f"[*] Executing on single device: {device}")
    else:
        device = torch.device("cpu")
        use_data_parallel = False
        print("[*] Executing on device: cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Define configurations to sweep
    if args.sweep_table4:
        # Canonical 7 grid points from Table 4 of the manuscript
        sweep_configs = [
            (0.20, 0.05),
            (0.50, 0.05),
            (1.00, 0.05),
            (1.00, 0.10),
            (1.00, 0.20),
            (1.50, 0.10),
            (2.00, 0.10),
        ]
        print(f"[*] TABLE 4 SENSITIVITY MODE: Sweeping 7 canonical grid points (beta, mu): {sweep_configs}")
    else:
        sweep_configs = [(lam, mu) for lam in args.lambdas for mu in args.club_weights]
        print(f"[*] Commencing sweep across (beta, mu) grid: {sweep_configs}")
    
    # 2. Metadata and mappings
    meta_path = resolve_metadata_path(args.metadata_csv)
    meta_df = pd.read_csv(meta_path)
    mappings = build_taxonomy_mappings(meta_df)
    
    sample_img_rel = meta_df["image_path"].iloc[0] if "image_path" in meta_df.columns else ""
    image_root = resolve_image_root(args.image_root, sample_img_rel)
    print(f"[+] Using image directory: {image_root}")
    
    # 3. Splits
    loso_folds = generate_round_robin_loso_splits(meta_df, num_folds=5, seed=args.seed)
    loso_fold = loso_folds[args.fold]
    leaky_fold = generate_leaky_stratified_split(loso_fold, seed=args.seed)
    
    train_transform = build_train_transform(image_size=224)
    val_transform = build_val_transform(image_size=224)
    
    ds_train = TimberDataset(loso_fold["train"], str(image_root), mappings, transform=train_transform)
    ds_val = TimberDataset(loso_fold["val"], str(image_root), mappings, transform=val_transform)
    ds_test_loso = TimberDataset(loso_fold["test"], str(image_root), mappings, transform=val_transform)
    ds_test_leaky = TimberDataset(leaky_fold["test"], str(image_root), mappings, transform=val_transform)
    
    num_train_workers = 2 if (os.name != 'nt' and torch.cuda.is_available()) else 0
    pin_mem = torch.cuda.is_available()
    
    loader_train = DataLoader(
        ds_train,
        batch_sampler=SpecimenBalancedBatchSampler(loso_fold["train"], batch_size=args.batch_size, samples_per_specimen=4, seed=args.seed),
        num_workers=num_train_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_train_workers > 0),
    )
    # Strictly num_workers=0 on validation and test to prevent host RAM explosion on Kaggle/Linux
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    loader_test_loso = DataLoader(ds_test_loso, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    loader_test_leaky = DataLoader(ds_test_leaky, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    sweep_results = []
    
    for (beta_val, mu_val) in sweep_configs:
        print(f"\n" + "=" * 65, flush=True)
        print(f"  RUNNING CONFIG: beta (adv) = {beta_val:.2f} | mu (club) = {mu_val:.2f} | Method: {args.method}", flush=True)
        print("=" * 65, flush=True)
        
        run_name = f"pareto_beta_{beta_val:.2f}_mu_{mu_val:.2f}"
        run_dir = out_dir / run_name
        best_ckpt = run_dir / "best_model.pth"
        train_summary_file = run_dir / "train_summary.json"
        
        model_cfg = ModelConfig(backbone_name=args.backbone, max_lambda_adv=beta_val)
        train_cfg = TrainingConfig(
            method=args.method,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            club_weight=mu_val,
        )
        
        model = SpecimenInvariantModel(
            config=model_cfg,
            specimen_counts=mappings["specimen_counts"],
            num_total_specimens=mappings["num_total_specimens"]
        )
        if use_data_parallel:
            model = nn.DataParallel(model)
        
        # Check if run already completed (Cache hit)
        if best_ckpt.exists() and train_summary_file.exists():
            print(f"[*] [CACHE HIT] Found completed checkpoint and summary at {run_dir}. Skipping training!", flush=True)
        else:
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
            trainer.train()
            # Explicit memory cleanup
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Load best checkpoint
        if best_ckpt.exists():
            ckpt = safe_load_checkpoint(best_ckpt, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model
            raw_state_dict = ckpt["model_state_dict"]
            cleaned_state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in raw_state_dict.items()
            }
            raw_model.load_state_dict(cleaned_state_dict)
            print(f"[+] Loaded best checkpoint from epoch {ckpt.get('epoch', 'N/A')}")
            
        # Evaluate LOSO with Calibration
        test_loso_metrics = evaluate_model_with_calibration(model, loader_test_loso, device)
        test_leaky_metrics = evaluate_model_with_calibration(model, loader_test_leaky, device)
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
            "beta": float(beta_val),
            "mu": float(mu_val),
            "lambda_adv": float(beta_val),
            "val_acc": float(test_loso_metrics["accuracy"]),
            "val_macro_f1": float(test_loso_metrics["macro_f1"]),
            "sri": float(probe_res["mean_sri"]),
            "ece": float(test_loso_metrics["ece"]),
            "mce": float(test_loso_metrics["mce"]),
            "ggsl_acc": float(ggsl["ggsl_acc"]),
            "ggsl_macro_f1": float(ggsl["ggsl_macro_f1"]),
            "method_name": f"β={beta_val:.2f}, μ={mu_val:.2f}",
        }
        sweep_results.append(record)
        print(f"[Result β={beta_val:.2f}, μ={mu_val:.2f}] Strict Acc: {record['val_acc']:.2f}% | "
              f"Macro-F1: {record['val_macro_f1']:.2f}% | SRI: {record['sri']:.4f} | ECE: {record['ece']:.2f}% | "
              f"GGSL Acc: {record['ggsl_acc']:+.2f} pp", flush=True)
        
        # Free memory and clear cache before next iteration to prevent OOM
        del model, embs_train, probe_res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    # Save sweep json & csv
    with open(out_dir / "pareto_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    pd.DataFrame(sweep_results).to_csv(out_dir / "pareto_sweep_results.csv", index=False)
    
    # Generate LaTeX rows for Table 4 (Hyperparameter Sensitivity)
    tex_rows = []
    for r in sweep_results:
        beta_str = f"{r['beta']:.2f}"
        mu_str = f"{r['mu']:.2f}"
        acc_str = f"{r['val_acc']:.2f}"
        sri_str = f"{r['sri']:.3f}"
        ece_str = f"{r['ece']:.2f}"
        if abs(r['beta'] - 1.00) < 1e-4 and abs(r['mu'] - 0.10) < 1e-4:
            tex_rows.append(f"\\textbf{{{beta_str}}} & \\textbf{{{mu_str}}} & \\textbf{{{acc_str}}} & \\textbf{{{sri_str}}} & \\textbf{{{ece_str}}} \\\\")
        else:
            tex_rows.append(f"{beta_str} & {mu_str} & {acc_str} & {sri_str} & {ece_str} \\\\")
    
    latex_snippet_path = out_dir / "pareto_sensitivity_table.tex"
    with open(latex_snippet_path, "w", encoding="utf-8") as f:
        f.write("% Table 4: Hyperparameter Sensitivity LaTeX rows\n")
        f.write("\n".join(tex_rows) + "\n")
    print(f"[+] Saved formatted LaTeX table snippet to: {latex_snippet_path}", flush=True)
        
    # Plot Pareto Curve
    try:
        pareto_plot_path = out_dir / "pareto_frontier_accuracy_vs_sri.png"
        plot_pareto_curve(sweep_results, str(pareto_plot_path))
        print(f"[+] Saved Pareto Curve to: {pareto_plot_path}", flush=True)
        
        corr_plot_path = out_dir / "correlation_sri_vs_ggsl.png"
        plot_correlation_sri_vs_ggsl(sweep_results, str(corr_plot_path))
        print(f"[+] Saved Correlation Plot to: {corr_plot_path}", flush=True)
    except Exception as e:
        print(f"[!] Warning: Plot generation failed with: {e}", flush=True)
        
    print("\n[SUCCESS] Completed Pareto sweep and correlation analyses!", flush=True)


if __name__ == "__main__":
    main()
