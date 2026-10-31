"""
run_benchmark_3seeds.py
=======================
Master Multi-Seed Execution Pipeline for Specimen-Invariant Timber Identification.
Strictly implements the dynamic per-seed protocol with 3 evaluation sets and 3 unseen vouchers/species:
  For each seed s in [42, 123, 456, 789]:
    1. Set random seed s.
    2. Partition dataset dynamically by seed s:
       - 3 vouchers per species -> Strictly Out-of-Specimen Unseen Physical Test (33 vouchers)
       - 7 vouchers per species -> 70% Train, 15% In-Specimen Validation, 15% In-Specimen Seen Test (77 vouchers)
    3. Train Standard Baseline Model (ResNet-18 ERM) for 20 epochs.
    4. Train Proposed Adversarial Invariant 'GAN' Model (ResNet-18 GRL + CLUB) for 20 epochs.
    5. Evaluate both models across all 3 evaluation sets:
       - Set 1: In-Specimen Validation Set (Val Acc, Val F1)
       - Set 2: In-Specimen Seen Test Set (Seen Test Acc, Seen Test F1)
       - Set 3: Out-of-Specimen Unseen Test Set (Unseen Acc, Unseen F1, ECE, Linear SRI, MLP SRI)
       - Generalization Gaps: (Seen Test - Unseen Test) and (Val - Unseen Test)
  Finally: Aggregates Mean ± Std across 4 seeds and compiles publication Markdown report.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUNBUFFERED"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dataset import (
    DEFAULT_IMAGE_ROOT,
    TARGET_SPECIES,
    scan_and_build_dataset,
    split_dataset_by_seed,
    create_seed_dataloaders,
)
from models_gan import (
    StandardBaselineModel,
    AdversarialInvariantGAN,
)

CURRENT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int):
    """Sets deterministic random state across python, numpy, torch, and cuDNN."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_grl_alpha(epoch: int, total_epochs: int) -> float:
    """Standard dynamic sigmoid schedule for GRL alpha in [0, 1]."""
    p = float(epoch) / float(max(1, total_epochs))
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ==============================================================================
# EVALUATION & PROBING FUNCTIONS
# ==============================================================================
@torch.no_grad()
def extract_embeddings_and_predictions(model, loader, device):
    model.eval()
    all_embs, all_preds, all_targets, all_probs, all_specimens = [], [], [], [], []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        species_targets = batch["species_idx"].to(device, non_blocking=True)
        specimen_targets = batch["specimen_idx"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            out = model(images)

        logits = out["species_logits"]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_embs.append(out["features"].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(species_targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_specimens.extend(specimen_targets.cpu().numpy())

    feat_dim = all_embs[0].shape[-1] if all_embs else 2048
    return {
        "embeddings": np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, feat_dim)),
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
        "probabilities": np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 11)),
        "specimens": np.array(all_specimens),
    }


def compute_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) in percentage [0, 100]."""
    if len(targets) == 0:
        return 0.0
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = predictions == targets

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
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
    if len(specimens) == 0:
        return 0.0, 0.0
    majority_chance = float(np.max(counts)) / float(len(specimens))

    valid_mask = np.isin(specimens, unique_s[counts >= 2])
    X = embs[valid_mask]
    y = specimens[valid_mask]

    if len(X) < 20 or len(np.unique(y)) < 2:
        return 0.0, 0.0

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    lin_accs, mlp_accs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf_lin = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        clf_lin.fit(X_train, y_train)
        lin_accs.append(accuracy_score(y_test, clf_lin.predict(X_test)))

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

    sri_lin = max(0.0, min(1.0, (acc_lin - majority_chance) / (1.0 - majority_chance + 1e-6)))
    sri_mlp = max(0.0, min(1.0, (acc_mlp - majority_chance) / (1.0 - majority_chance + 1e-6)))
    return sri_lin, sri_mlp


def evaluate_model_full(
    model,
    loader_val,
    loader_seen_test,
    loader_unseen,
    device,
    seed: int = 42,
) -> Dict:
    """
    Evaluates a trained model across all 3 evaluation sets:
      1. In-Specimen Validation Set (loader_val)
      2. In-Specimen Seen Test Set (loader_seen_test)
      3. Out-of-Specimen Unseen Physical Test Set (loader_unseen)
    """
    val_res = extract_embeddings_and_predictions(model, loader_val, device)
    seen_res = extract_embeddings_and_predictions(model, loader_seen_test, device)
    unseen_res = extract_embeddings_and_predictions(model, loader_unseen, device)

    # 1. Validation Set (Seen)
    val_acc = float(accuracy_score(val_res["targets"], val_res["predictions"]) * 100.0)
    val_f1 = float(f1_score(val_res["targets"], val_res["predictions"], average="macro", zero_division=0) * 100.0)

    # 2. Seen Test Set (Seen)
    seen_acc = float(accuracy_score(seen_res["targets"], seen_res["predictions"]) * 100.0)
    seen_f1 = float(f1_score(seen_res["targets"], seen_res["predictions"], average="macro", zero_division=0) * 100.0)

    # 3. Unseen Test Set (Unseen - 3 vouchers/species)
    unseen_acc = float(accuracy_score(unseen_res["targets"], unseen_res["predictions"]) * 100.0)
    unseen_f1 = float(f1_score(unseen_res["targets"], unseen_res["predictions"], average="macro", zero_division=0) * 100.0)
    unseen_ece = float(compute_ece(unseen_res["probabilities"], unseen_res["targets"]))

    # SRI Probing on Unseen Physical Specimen Embeddings
    sri_lin, sri_mlp = compute_sri_probes(unseen_res["embeddings"], unseen_res["specimens"], seed=seed)

    # Generalization Gaps
    gen_gap_seen_unseen_acc = seen_acc - unseen_acc
    gen_gap_seen_unseen_f1 = seen_f1 - unseen_f1
    gen_gap_val_unseen_acc = val_acc - unseen_acc
    gen_gap_val_unseen_f1 = val_f1 - unseen_f1

    return {
        "val_top1_acc": val_acc,
        "val_macro_f1": val_f1,
        "seen_top1_acc": seen_acc,
        "seen_macro_f1": seen_f1,
        "unseen_top1_acc": unseen_acc,
        "unseen_macro_f1": unseen_f1,
        "unseen_ece": unseen_ece,
        "gen_gap_seen_unseen_acc": gen_gap_seen_unseen_acc,
        "gen_gap_seen_unseen_f1": gen_gap_seen_unseen_f1,
        "gen_gap_val_unseen_acc": gen_gap_val_unseen_acc,
        "gen_gap_val_unseen_f1": gen_gap_val_unseen_f1,
        "sri_linear": sri_lin,
        "sri_mlp": sri_mlp,
    }


# ==============================================================================
# TRAINING ROUTINES FOR SINGLE SEED
# ==============================================================================
def train_baseline_one_seed(loader_train, loader_val, epochs: int, lr: float, device, backbone: str, seed: int):
    """Trains Standard Baseline (ERM) for one seed."""
    print(f"\n[*] [Seed {seed}] Training Standard Baseline Model ({backbone} ERM)...")
    model = StandardBaselineModel(backbone_name=backbone, num_classes=len(TARGET_SPECIES), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in loader_train:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["species_idx"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                out = model(images)
                loss = criterion(out["species_logits"], targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(out["species_logits"], dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

        scheduler.step()
        train_acc = accuracy_score(all_targets, all_preds) * 100.0

        # Validate on In-Specimen Validation set
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in loader_val:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["species_idx"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                    out = model(images)
                preds = torch.argmax(out["species_logits"], dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0

        if val_f1 >= best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (F1: {val_f1:.2f}%)")

    model.load_state_dict(best_weights)
    return model


def train_gan_one_seed(loader_train, loader_val, epochs: int, lr: float, adv_weight: float, club_weight: float, device, backbone: str, seed: int, num_specimens: int = 77):
    """Trains Proposed Adversarial Invariant 'GAN' Model (GRL + CLUB) for one seed."""
    print(f"\n[*] [Seed {seed}] Training Proposed Adversarial Invariant 'GAN' Model ({backbone} GRL + CLUB | {num_specimens} train specimens)...")
    model = AdversarialInvariantGAN(
        backbone_name=backbone,
        num_classes=len(TARGET_SPECIES),
        num_specimens=num_specimens,
        pretrained=True,
        adv_weight=adv_weight,
        club_weight=club_weight,
    ).to(device)

    criterion_species = nn.CrossEntropyLoss()
    criterion_specimen = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

    best_val_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        alpha = compute_grl_alpha(epoch, epochs)
        model.train()
        model.set_alpha(alpha)

        all_preds, all_targets = [], []
        for batch in loader_train:
            images = batch["image"].to(device, non_blocking=True)
            species_targets = batch["species_idx"].to(device, non_blocking=True)
            specimen_targets = batch["specimen_idx"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                out = model(images, specimen_labels=specimen_targets, species_labels=species_targets)
                loss_species = criterion_species(out["species_logits"], species_targets)
                loss_specimen = criterion_specimen(out["specimen_logits"], specimen_targets)
                loss_club = out.get("club_loss", torch.tensor(0.0, device=device))

                loss_total = loss_species + (adv_weight * alpha * loss_specimen) + (club_weight * loss_club)

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(out["species_logits"], dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(species_targets.detach().cpu().numpy())

        scheduler.step()
        train_acc = accuracy_score(all_targets, all_preds) * 100.0

        # Validate on In-Specimen Validation set
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in loader_val:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["species_idx"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                    out = model(images)
                preds = torch.argmax(out["species_logits"], dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0

        if val_f1 >= best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] (alpha={alpha:.2f}) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (F1: {val_f1:.2f}%)")

    model.load_state_dict(best_weights)
    return model


# ==============================================================================
# MAIN MULTI-SEED WORKFLOW
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Specimen-Invariant Benchmark Pipeline")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT, help="Root folder of raw images")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789], help="List of random seeds to evaluate")
    parser.add_argument("--unseen_vouchers_per_species", type=int, default=3, help="Number of unseen vouchers per species (default: 3)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--adv_weight", type=float, default=0.05, help="GRL discriminator loss weight beta")
    parser.add_argument("--club_weight", type=float, default=0.005, help="Variational CLUB MI loss weight mu")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Visual backbone")
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "outputs"), help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  SPECIMEN-INVARIANT TIMBER IDENTIFICATION: MULTI-SEED SCIENTIFIC BENCHMARK")
    print(f"  Target Taxa : 11 Clean Fabaceae Species (110 physical vouchers)")
    print(f"  Protocol    : 3 Evaluation Sets (Val, Seen Test, Unseen Test)")
    print(f"  Holdout     : {args.unseen_vouchers_per_species} Unseen Vouchers/Species ({args.unseen_vouchers_per_species * len(TARGET_SPECIES)} physical vouchers total)")
    print(f"  Backbone    : {args.backbone}")
    print(f"  Device      : {device} (cuDNN enabled: {torch.cuda.is_available()})")
    print(f"  Seeds       : {args.seeds} (Total {len(args.seeds)} experimental runs)")
    print(f"  Image Root  : {args.image_root}")
    print("=" * 80 + "\n")

    # Step 1: Scan and build dataset directly from disk
    print("[STEP 1] Scanning and building dataset directly from image folder...")
    df_raw = scan_and_build_dataset(args.image_root)

    results_by_seed = []

    # Step 2: Loop through seeds
    for seed_idx, seed in enumerate(args.seeds, 1):
        print("\n" + "#" * 80)
        print(f"  [RUN {seed_idx}/{len(args.seeds)}] EXECUTING EXPERIMENTAL SEED {seed}")
        print("#" * 80)

        # 1. Set seed
        set_seed(seed)

        # 2. Dynamic partitioning for this seed into 3 evaluation sets
        df_train, df_val, df_seen_test, df_unseen = split_dataset_by_seed(
            df_raw,
            seed=seed,
            unseen_vouchers_per_species=args.unseen_vouchers_per_species,
            val_ratio=0.15,
            seen_test_ratio=0.15,
        )

        # 3. Create loaders for all sets
        loader_train, loader_val, loader_seen_test, loader_unseen, info = create_seed_dataloaders(
            df_train, df_val, df_seen_test, df_unseen, batch_size=args.batch_size
        )

        # 4. Train Standard Baseline
        baseline_model = train_baseline_one_seed(
            loader_train, loader_val, epochs=args.epochs, lr=args.lr, device=device, backbone=args.backbone, seed=seed
        )

        # 5. Train Proposed Adversarial Invariant 'GAN'
        gan_model = train_gan_one_seed(
            loader_train, loader_val, epochs=args.epochs, lr=args.lr,
            adv_weight=args.adv_weight, club_weight=args.club_weight,
            device=device, backbone=args.backbone, seed=seed,
            num_specimens=info["train_specimens"],
        )

        # 6. Evaluate Baseline across 3 evaluation sets
        print(f"\n[*] [Seed {seed}] Auditing Standard Baseline Model across 3 Evaluation Sets (Val, Seen Test, Unseen Test)...")
        base_metrics = evaluate_model_full(baseline_model, loader_val, loader_seen_test, loader_unseen, device=device, seed=seed)

        # 7. Evaluate GAN across 3 evaluation sets
        print(f"[*] [Seed {seed}] Auditing Proposed 'GAN' Model across 3 Evaluation Sets (Val, Seen Test, Unseen Test)...")
        gan_metrics = evaluate_model_full(gan_model, loader_val, loader_seen_test, loader_unseen, device=device, seed=seed)

        # Print detailed 3-set summary for this seed
        print(f"\n--- [Seed {seed} Executive Summary: 3 Evaluation Sets] ---")
        print(f"  Baseline -> Val Acc: {base_metrics['val_top1_acc']:.2f}% | Seen Test Acc: {base_metrics['seen_top1_acc']:.2f}% | Unseen Test Acc: {base_metrics['unseen_top1_acc']:.2f}% | Gap (Seen-Unseen): +{base_metrics['gen_gap_seen_unseen_acc']:.2f} pp | SRI (MLP): {base_metrics['sri_mlp']:.4f} | ECE: {base_metrics['unseen_ece']:.2f}%")
        print(f"  'GAN'    -> Val Acc: {gan_metrics['val_top1_acc']:.2f}% | Seen Test Acc: {gan_metrics['seen_top1_acc']:.2f}% | Unseen Test Acc: {gan_metrics['unseen_top1_acc']:.2f}% | Gap (Seen-Unseen): +{gan_metrics['gen_gap_seen_unseen_acc']:.2f} pp | SRI (MLP): {gan_metrics['sri_mlp']:.4f} | ECE: {gan_metrics['unseen_ece']:.2f}%")

        results_by_seed.append({
            "seed": seed,
            "baseline": base_metrics,
            "gan": gan_metrics,
        })

    # Step 3: Compute aggregate statistics across all seeds
    print("\n" + "=" * 80)
    print("  AGGREGATING MULTI-SEED BENCHMARK STATISTICS")
    print("=" * 80)

    metric_keys = [
        "val_top1_acc", "val_macro_f1",
        "seen_top1_acc", "seen_macro_f1",
        "unseen_top1_acc", "unseen_macro_f1", "unseen_ece",
        "gen_gap_seen_unseen_acc", "gen_gap_seen_unseen_f1",
        "gen_gap_val_unseen_acc", "gen_gap_val_unseen_f1",
        "sri_linear", "sri_mlp",
    ]

    summary = {"baseline": {}, "gan": {}, "comparison": {}}

    for key in metric_keys:
        b_vals = [r["baseline"][key] for r in results_by_seed]
        g_vals = [r["gan"][key] for r in results_by_seed]

        summary["baseline"][key] = {
            "mean": float(np.mean(b_vals)),
            "std": float(np.std(b_vals)),
            "values": b_vals,
        }
        summary["gan"][key] = {
            "mean": float(np.mean(g_vals)),
            "std": float(np.std(g_vals)),
            "values": g_vals,
        }
        summary["comparison"][key] = {
            "delta_mean": float(np.mean(g_vals) - np.mean(b_vals)),
        }

    unseen_v_count = args.unseen_vouchers_per_species * len(TARGET_SPECIES)
    train_v_count = 110 - unseen_v_count

    full_output = {
        "benchmark_meta": {
            "seeds": args.seeds,
            "num_seeds": len(args.seeds),
            "epochs": args.epochs,
            "backbone": args.backbone,
            "num_species": len(TARGET_SPECIES),
            "total_physical_vouchers": 110,
            "unseen_vouchers_per_species": args.unseen_vouchers_per_species,
            "num_unseen_vouchers": unseen_v_count,
            "num_train_vouchers": train_v_count,
            "eval_sets": [
                "Bộ 1: Validation Set (In-Specimen)",
                "Bộ 2: Seen Test Set (In-Specimen)",
                "Bộ 3: Unseen Test Set (Out-of-Specimen, 3 vouchers/species)",
            ],
            "image_root": str(args.image_root),
        },
        "per_seed_results": results_by_seed,
        "aggregate_summary": summary,
    }

    results_json_path = out_dir / "multi_seed_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[+] Lưu kết quả multi-seed JSON thành công tại: {results_json_path}")

    # Step 4: Generate publication markdown report
    import generate_report
    generate_report.generate_multiseed_report(results_json_path, CURRENT_DIR / "EXPERIMENT_RESULTS_REPORT.md")


if __name__ == "__main__":
    main()
