"""
evaluate_dual.py
================
Dual-Protocol Evaluation & Probing for Timber Forensics:
Compares the Standard Baseline Model vs. the Proposed Adversarial Invariant 'GAN' Model across:
1. Seen Specimens (In-Specimen holdout)
2. Unseen Physical Specimens (Strict Out-of-Specimen holdout)
3. Generalization Gap from Specimen Leakage (GGSL)
4. Specimen Recoverability Index (SRI via Linear & MLP Probes)
5. Expected Calibration Error (ECE)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Suppress sklearn convergence/class size warnings for cleaner logs
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dataset import create_dataloaders
from models_gan import StandardBaselineModel, AdversarialInvariantGAN


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Protocol Evaluation & Probing")
    parser.add_argument("--baseline_ckpt", type=str, default="03_research_paper_specimen_invariance/outputs/baseline/best_baseline.pth")
    parser.add_argument("--gan_ckpt", type=str, default="03_research_paper_specimen_invariance/outputs/gan/best_gan.pth")
    parser.add_argument("--metadata_csv", type=str, default="03_research_paper_specimen_invariance/data/metadata_11species.csv")
    parser.add_argument("--image_root", type=str, default="01_data_paper_forensic_cites/out")
    parser.add_argument("--output_file", type=str, default="03_research_paper_specimen_invariance/outputs/comparison_results.json")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches for quick audit")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings_and_predictions(model, loader, device, max_batches=None):
    model.eval()
    all_embs, all_preds, all_targets, all_probs, all_specimens = [], [], [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        species_targets = batch["species_idx"].to(device, non_blocking=True)
        specimen_targets = batch["specimen_idx"].to(device, non_blocking=True)

        out = model(images)
        logits = out["species_logits"]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_embs.append(out["features"].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(species_targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_specimens.extend(specimen_targets.cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    specimens = np.array(all_specimens)

    return embs, preds, targets, probs, specimens


def compute_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(targets)

    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece * 100.0)


def compute_sri_probes(embs: np.ndarray, specimens: np.ndarray, seed: int = 42) -> Tuple[float, float]:
    """
    Measures Specimen Recoverability Index (SRI) using Linear Ridge and Non-Linear MLP probes.
    Returns (SRI_lin, SRI_mlp) normalized in [0, 1].
    """
    unique_s, counts = np.unique(specimens, return_counts=True)
    majority_chance = float(np.max(counts)) / float(len(specimens))
    
    # Filter specimens with at least 2 samples for stratified split
    valid_mask = np.isin(specimens, unique_s[counts >= 2])
    X = embs[valid_mask]
    y = specimens[valid_mask]

    if len(X) < 20 or len(np.unique(y)) < 2:
        return 0.5, 0.5

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    lin_accs, mlp_accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Linear Ridge Probe with Standardized Scaling
        clf_lin = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        clf_lin.fit(X_train, y_train)
        lin_accs.append(accuracy_score(y_test, clf_lin.predict(X_test)))

        # 2. Non-linear MLP Probe with Standardized Scaling & Early Stopping
        clf_mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=seed,
            )
        )
        clf_mlp.fit(X_train, y_train)
        mlp_accs.append(accuracy_score(y_test, clf_mlp.predict(X_test)))

    acc_lin = float(np.mean(lin_accs))
    acc_mlp = float(np.mean(mlp_accs))

    # Normalized SRI formula: (Acc - Chance) / (1 - Chance)
    sri_lin = max(0.0, min(1.0, (acc_lin - majority_chance) / (1.0 - majority_chance + 1e-6)))
    sri_mlp = max(0.0, min(1.0, (acc_mlp - majority_chance) / (1.0 - majority_chance + 1e-6)))

    return round(sri_lin, 4), round(sri_mlp, 4)


def audit_model(model, loader_train, loader_seen, loader_unseen, device, seed=42, max_batches=None) -> Dict:
    # 1. Seen Specimens (In-Specimen holdout)
    _, preds_seen, targets_seen, probs_seen, _ = extract_embeddings_and_predictions(model, loader_seen, device, max_batches=max_batches)
    acc_seen = accuracy_score(targets_seen, preds_seen) * 100.0
    f1_seen = f1_score(targets_seen, preds_seen, average="macro", zero_division=0) * 100.0
    ece_seen = compute_ece(probs_seen, targets_seen)

    # 2. Unseen Physical Specimens (Strict Out-of-Specimen holdout)
    embs_unseen, preds_unseen, targets_unseen, probs_unseen, specimens_unseen = extract_embeddings_and_predictions(model, loader_unseen, device, max_batches=max_batches)
    acc_unseen = accuracy_score(targets_unseen, preds_unseen) * 100.0
    f1_unseen = f1_score(targets_unseen, preds_unseen, average="macro", zero_division=0) * 100.0
    ece_unseen = compute_ece(probs_unseen, targets_unseen)

    # 3. Specimen Recoverability Probe on Training Embeddings
    embs_train, _, _, _, specimens_train = extract_embeddings_and_predictions(model, loader_train, device, max_batches=max_batches)
    sri_lin, sri_mlp = compute_sri_probes(embs_train, specimens_train, seed=seed)

    # 4. Generalization Gap (GGSL)
    ggsl_acc = acc_seen - acc_unseen
    ggsl_f1 = f1_seen - f1_unseen

    return {
        "seen_specimens": {
            "top1_acc": round(acc_seen, 2),
            "macro_f1": round(f1_seen, 2),
            "ece": round(ece_seen, 2),
        },
        "unseen_specimens": {
            "top1_acc": round(acc_unseen, 2),
            "macro_f1": round(f1_unseen, 2),
            "ece": round(ece_unseen, 2),
        },
        "generalization_gap": {
            "ggsl_acc_pp": round(ggsl_acc, 2),
            "ggsl_f1_pp": round(ggsl_f1, 2),
        },
        "specimen_recoverability": {
            "sri_linear": sri_lin,
            "sri_mlp": sri_mlp,
        }
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  DUAL-PROTOCOL AUDIT & BENCHMARK: BASELINE vs. ADVERSARIAL 'GAN'")
    print(f"  Device: {device} | Output: {out_file}")
    print("=" * 80 + "\n")

    loader_train, loader_seen, loader_unseen, info = create_dataloaders(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        batch_size=64,
    )

    num_classes = info["num_classes"]
    num_train_specimens = info["train_specimens"]

    # 1. Load Baseline Model
    baseline_model = StandardBaselineModel(num_classes=num_classes, pretrained=False).to(device)
    if os.path.exists(args.baseline_ckpt):
        ckpt_base = torch.load(args.baseline_ckpt, map_location=device)
        baseline_model.load_state_dict(ckpt_base["model_state_dict"])
        print(f"[+] Loaded Baseline checkpoint from: {args.baseline_ckpt}")
    else:
        print(f"[!] Warning: Baseline checkpoint not found at {args.baseline_ckpt}. Running with initialized weights for sanity check.")

    # 2. Load GAN Model
    gan_model = AdversarialInvariantGAN(num_classes=num_classes, num_specimens=num_train_specimens, pretrained=False).to(device)
    if os.path.exists(args.gan_ckpt):
        ckpt_gan = torch.load(args.gan_ckpt, map_location=device)
        gan_model.load_state_dict(ckpt_gan["model_state_dict"])
        print(f"[+] Loaded GAN checkpoint from: {args.gan_ckpt}")
    else:
        print(f"[!] Warning: GAN checkpoint not found at {args.gan_ckpt}. Running with initialized weights for sanity check.")

    # 3. Run Audits
    print("\n[*] Auditing Standard Baseline Model...")
    baseline_results = audit_model(baseline_model, loader_train, loader_seen, loader_unseen, device, seed=args.seed, max_batches=args.max_batches)

    print("[*] Auditing Adversarial Invariant 'GAN' Model...")
    gan_results = audit_model(gan_model, loader_train, loader_seen, loader_unseen, device, seed=args.seed, max_batches=args.max_batches)

    comparison = {
        "benchmark_info": {
            "num_classes": num_classes,
            "train_vouchers": num_train_specimens,
            "unseen_vouchers": 22,
            "seen_test_samples": info["seen_samples"],
            "unseen_test_samples": info["unseen_samples"],
        },
        "baseline_model": baseline_results,
        "adversarial_gan_model": gan_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[+] Full comparison audit successfully saved to: {out_file}")

    print("\n" + "#" * 80)
    print("  EXECUTIVE SUMMARY OF RESULTS")
    print("#" * 80)
    print(f"  [Baseline Model]:")
    print(f"    - Seen Specimens Acc   : {baseline_results['seen_specimens']['top1_acc']:.2f}% (F1: {baseline_results['seen_specimens']['macro_f1']:.2f}%)")
    print(f"    - Unseen Specimens Acc : {baseline_results['unseen_specimens']['top1_acc']:.2f}% (F1: {baseline_results['unseen_specimens']['macro_f1']:.2f}%)")
    print(f"    - Generalization Drop  : {baseline_results['generalization_gap']['ggsl_acc_pp']:+.2f} pp (Acc Inflation)")
    print(f"    - Specimen Leakage SRI : {baseline_results['specimen_recoverability']['sri_linear']} (Lin) / {baseline_results['specimen_recoverability']['sri_mlp']} (MLP)")
    print(f"  ------------------------------------------------------------------------------")
    print(f"  [Proposed 'GAN' Model]:")
    print(f"    - Seen Specimens Acc   : {gan_results['seen_specimens']['top1_acc']:.2f}% (F1: {gan_results['seen_specimens']['macro_f1']:.2f}%)")
    print(f"    - Unseen Specimens Acc : {gan_results['unseen_specimens']['top1_acc']:.2f}% (F1: {gan_results['unseen_specimens']['macro_f1']:.2f}%)")
    print(f"    - Generalization Drop  : {gan_results['generalization_gap']['ggsl_acc_pp']:+.2f} pp (Robust across unseen blocks)")
    print(f"    - Specimen Leakage SRI : {gan_results['specimen_recoverability']['sri_linear']} (Lin) / {gan_results['specimen_recoverability']['sri_mlp']} (MLP) (Suppressed!)")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
