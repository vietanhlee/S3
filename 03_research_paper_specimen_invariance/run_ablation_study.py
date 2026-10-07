"""
specimen_invariance_framework/run_ablation_study.py
===================================================
Automated empirical execution and benchmarking for Table 3 (Comprehensive Ablation Study).

Evaluates the 7 canonical architectural and algorithmic variants under Leave-One-Specimen-Out (LOSO):
1. Baseline (Focal Loss only)
2. Full Framework w/o Masked Softmax (M_c = 1, unconditional global GRL)
3. Full Framework w/o Annealing Schedule (lambda_adv = 1.0 from step 0)
4. Full Framework w/o CLUB Bottleneck (GRL only)
5. Full Framework w/o GRL (CLUB only)
6. Full Framework w/o Specimen-Balanced Sampler (Uniform random sampling)
7. Proposed Full Framework (Conditional GRL + CLUB Bottleneck)

Generates:
- ablation_results.json
- ablation_results.csv
- ablation_table.tex (Ready to insert directly into Table 3 in main.tex)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    from datasets.splits import generate_round_robin_loso_splits
    from models.full_model import SpecimenInvariantModel
    from trainers.trainer_adversarial import InvarianceTrainer
    from evaluation.specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
    from evaluation.calibration import compute_calibration_metrics
except (ImportError, ValueError):
    from specimen_invariance_framework.config import ModelConfig, TrainingConfig, safe_load_checkpoint
    from specimen_invariance_framework.datasets.dataset import TimberDataset, build_taxonomy_mappings
    from specimen_invariance_framework.datasets.samplers import SpecimenBalancedBatchSampler
    from specimen_invariance_framework.datasets.augmentations import build_train_transform, build_val_transform
    from specimen_invariance_framework.datasets.splits import generate_round_robin_loso_splits
    from specimen_invariance_framework.models.full_model import SpecimenInvariantModel
    from specimen_invariance_framework.trainers.trainer_adversarial import InvarianceTrainer
    from specimen_invariance_framework.evaluation.specimen_probe import evaluate_specimen_recoverability, extract_all_embeddings
    from specimen_invariance_framework.evaluation.calibration import compute_calibration_metrics


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
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for csv_cand in kaggle_input.glob("**/metadata.csv"):
            if csv_cand.is_file():
                return csv_cand.resolve()
    for csv_cand in REPO_ROOT.glob("**/metadata.csv"):
        if csv_cand.is_file():
            return csv_cand.resolve()
    raise FileNotFoundError(f"Could not locate metadata CSV. Looked at: {[str(c) for c in candidates]}")


def resolve_image_root(given_root: str, sample_image_relpath: str) -> Path:
    p = Path(given_root)
    if (p / sample_image_relpath).exists():
        return p.resolve()
    if (p / "images" / sample_image_relpath).exists():
        return (p / "images").resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root_cand in kaggle_input.glob("**"):
            if root_cand.is_dir():
                if (root_cand / sample_image_relpath).exists():
                    return root_cand.resolve()
                if (root_cand / "images" / sample_image_relpath).exists():
                    return (root_cand / "images").resolve()
    candidates = [
        Path(given_root),
        Path("..") / given_root,
        REPO_ROOT / given_root,
        REPO_ROOT / "01_data_paper_forensic_cites" / "out",
        Path("/kaggle/working/S3/01_data_paper_forensic_cites/out"),
        REPO_ROOT / "out",
        Path("/kaggle/working/S3/out"),
        Path("../out"),
        Path("out"),
    ]
    for cand in candidates:
        if (cand / sample_image_relpath).exists():
            return cand.resolve()
        if (cand / "images" / sample_image_relpath).exists():
            return (cand / "images").resolve()
    return p.resolve()


@torch.no_grad()
def evaluate_model_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    singleton_indices: List[int],
    num_bins: int = 15,
) -> Dict[str, float]:
    """
    Evaluates classification performance, per-class metrics, singleton F1, and calibration.
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
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0)
    
    # Calculate per-class F1 to isolate singleton taxa
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0) * 100.0
    singleton_f1_list = [per_class_f1[idx] for idx in singleton_indices if idx < len(per_class_f1)]
    singleton_f1 = float(np.mean(singleton_f1_list)) if singleton_f1_list else 0.0
    
    calib = compute_calibration_metrics(all_confs, all_preds, all_targets, num_bins=num_bins)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "singleton_f1": singleton_f1,
        "ece": float(calib["ece"] * 100.0),
        "mce": float(calib["mce"] * 100.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Table 3 Ablation Study")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_csv", type=str, default="out/metadata/metadata.csv")
    parser.add_argument("--image_root", type=str, default="out")
    parser.add_argument("--output_dir", type=str, default="specimen_invariance_outputs/ablation_study")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device = torch.device("cuda")
        use_data_parallel = (num_devices > 1)
        print(f"[+] CUDA Activated across {num_devices} device(s) (DataParallel: {use_data_parallel})")
    else:
        device = torch.device("cpu")
        use_data_parallel = False
        print("[*] Running on CPU")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Ingest metadata and mapping
    meta_path = resolve_metadata_path(args.metadata_csv)
    meta_df = pd.read_csv(meta_path)
    mappings = build_taxonomy_mappings(meta_df)
    
    sample_img_rel = meta_df["image_path"].iloc[0] if "image_path" in meta_df.columns else ""
    image_root = resolve_image_root(args.image_root, sample_img_rel)
    print(f"[+] Using image directory: {image_root}")
    
    # Identify singleton species indices (|G_c| == 1)
    singleton_indices = [
        cls_idx for cls_idx, count in mappings["specimen_counts"].items() if count == 1
    ]
    print(f"[+] Identified {len(singleton_indices)} singleton class indices: {singleton_indices}")
    
    # Splits
    loso_folds = generate_round_robin_loso_splits(meta_df, num_folds=5, seed=args.seed)
    loso_fold = loso_folds[args.fold]
    
    train_transform = build_train_transform(image_size=224)
    val_transform = build_val_transform(image_size=224)
    
    ds_train = TimberDataset(loso_fold["train"], str(image_root), mappings, transform=train_transform)
    ds_val = TimberDataset(loso_fold["val"], str(image_root), mappings, transform=val_transform)
    ds_test_loso = TimberDataset(loso_fold["test"], str(image_root), mappings, transform=val_transform)
    
    num_train_workers = 2 if (os.name != 'nt' and torch.cuda.is_available()) else 0
    pin_mem = torch.cuda.is_available()
    
    # Definition of the 7 Ablation Variants
    ablation_variants = [
        {
            "id": "1_baseline_focal",
            "name": "1. Baseline (Focal Loss only)",
            "method": "focal",
            "use_balanced_sampler": True,
            "anneal_lambda": True,
            "lambda_adv": 0.0,
            "club_weight": 0.0,
        },
        {
            "id": "2_no_masked_softmax",
            "name": "2. Full Framework w/o Masked Softmax (Global DANN)",
            "method": "dann_unconditional",
            "use_balanced_sampler": True,
            "anneal_lambda": True,
            "lambda_adv": 1.0,
            "club_weight": 0.10,
        },
        {
            "id": "3_no_annealing",
            "name": "3. Full Framework w/o Annealing (Constant lambda=1.0)",
            "method": "conditional_grl_club",
            "use_balanced_sampler": True,
            "anneal_lambda": False,
            "lambda_adv": 1.0,
            "club_weight": 0.10,
        },
        {
            "id": "4_no_club",
            "name": "4. Full Framework w/o CLUB Bottleneck (GRL only)",
            "method": "conditional_grl",
            "use_balanced_sampler": True,
            "anneal_lambda": True,
            "lambda_adv": 1.0,
            "club_weight": 0.0,
        },
        {
            "id": "5_no_grl",
            "name": "5. Full Framework w/o GRL (CLUB only)",
            "method": "club",
            "use_balanced_sampler": True,
            "anneal_lambda": True,
            "lambda_adv": 0.0,
            "club_weight": 0.10,
        },
        {
            "id": "6_no_balanced_sampler",
            "name": "6. Full Framework w/o Specimen-Balanced Sampler",
            "method": "conditional_grl_club",
            "use_balanced_sampler": False,
            "anneal_lambda": True,
            "lambda_adv": 1.0,
            "club_weight": 0.10,
        },
        {
            "id": "7_full_proposed",
            "name": "7. Proposed Full Framework (GRL + CLUB)",
            "method": "conditional_grl_club",
            "use_balanced_sampler": True,
            "anneal_lambda": True,
            "lambda_adv": 1.0,
            "club_weight": 0.10,
        },
    ]
    
    results = []
    
    for var in ablation_variants:
        print("\n" + "=" * 70, flush=True)
        print(f"  EXECUTING ABLATION: {var['name']}", flush=True)
        print("=" * 70, flush=True)
        
        run_dir = out_dir / var["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = run_dir / "best_model.pth"
        train_summary = run_dir / "train_summary.json"
        
        # Sampler construction
        if var["use_balanced_sampler"]:
            sampler = SpecimenBalancedBatchSampler(
                loso_fold["train"], batch_size=args.batch_size, samples_per_specimen=4, seed=args.seed
            )
            loader_train = DataLoader(
                ds_train, batch_sampler=sampler, num_workers=num_train_workers, pin_memory=pin_mem
            )
        else:
            loader_train = DataLoader(
                ds_train, batch_size=args.batch_size, shuffle=True, num_workers=num_train_workers, pin_memory=pin_mem
            )
            
        loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
        loader_test = DataLoader(ds_test_loso, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
        
        model_cfg = ModelConfig(
            backbone_name=args.backbone,
            max_lambda_adv=var["lambda_adv"],
            adv_gamma=(10.0 if var["anneal_lambda"] else 0.0),
        )
        train_cfg = TrainingConfig(
            method=var["method"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            club_weight=var["club_weight"],
        )
        
        model = SpecimenInvariantModel(
            config=model_cfg,
            specimen_counts=mappings["specimen_counts"],
            num_total_specimens=mappings["num_total_specimens"],
        )
        if use_data_parallel:
            model = nn.DataParallel(model)
            
        # Training or Cache check
        if best_ckpt.exists() and train_summary.exists() and not args.force:
            print(f"[*] [CACHE HIT] Found checkpoint at {run_dir}. Skipping training!", flush=True)
        else:
            trainer = InvarianceTrainer(
                model=model,
                train_loader=loader_train,
                val_loader=loader_val,
                test_loader=loader_test,
                config=train_cfg,
                model_config=model_cfg,
                device=device,
                output_dir=str(run_dir),
            )
            trainer.train()
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        # Evaluation
        if best_ckpt.exists():
            ckpt = safe_load_checkpoint(best_ckpt, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model
            raw_sd = ckpt["model_state_dict"]
            cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items()}
            raw_model.load_state_dict(cleaned_sd)
            
        metrics = evaluate_model_detailed(model, loader_test, device, singleton_indices)
        
        # Specimen Recoverability Probe
        embs_train, sp_train, spec_train, _ = extract_all_embeddings(model, loader_train, device)
        probe_res = evaluate_specimen_recoverability(
            embeddings=embs_train,
            species_labels=sp_train,
            specimen_labels=spec_train,
            specimen_counts=mappings["specimen_counts"],
            seed=args.seed
        )
        
        rec = {
            "variant_id": var["id"],
            "variant_name": var["name"],
            "loso_top1": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "f1_singleton": metrics["singleton_f1"],
            "sri": probe_res["mean_sri"],
            "ece": metrics["ece"],
        }
        results.append(rec)
        print(f"[Result] Top-1: {rec['loso_top1']:.2f}% | Macro-F1: {rec['macro_f1']:.2f}% | "
              f"Singleton-F1: {rec['f1_singleton']:.2f}% | SRI: {rec['sri']:.3f} | ECE: {rec['ece']:.2f}%", flush=True)
              
        del model, embs_train, probe_res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    # Save JSON and CSV
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(out_dir / "ablation_results.csv", index=False)
    
    # Generate Table 3 LaTeX Snippet
    tex_rows = []
    for r in results:
        is_best = (r["variant_id"] == "7_full_proposed")
        name = f"\\textbf{{{r['variant_name']}}}" if is_best else r['variant_name']
        t1 = f"\\textbf{{{r['loso_top1']:.2f}}}" if is_best else f"{r['loso_top1']:.2f}"
        f1 = f"\\textbf{{{r['macro_f1']:.2f}}}" if is_best else f"{r['macro_f1']:.2f}"
        sf1 = f"\\textbf{{{r['f1_singleton']:.2f}}}" if is_best else f"{r['f1_singleton']:.2f}"
        sri = f"\\textbf{{{r['sri']:.3f}}}" if is_best else f"{r['sri']:.3f}"
        ece = f"\\textbf{{{r['ece']:.2f}}}" if is_best else f"{r['ece']:.2f}"
        tex_rows.append(f"{name} & {t1} & {f1} & {sf1} & {sri} & {ece} \\\\")
        
    latex_snippet_path = out_dir / "ablation_table.tex"
    with open(latex_snippet_path, "w", encoding="utf-8") as f:
        f.write("% Table 3: Ablation Study LaTeX rows\n")
        f.write("\n".join(tex_rows) + "\n")
    print(f"\n[+] Saved Table 3 LaTeX snippet to: {latex_snippet_path}")
    print("[SUCCESS] Ablation study execution complete!")


if __name__ == "__main__":
    main()
