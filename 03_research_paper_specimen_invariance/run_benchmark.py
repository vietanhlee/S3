"""
run_benchmark.py
================
Master Multi-Seed Execution Pipeline for Specimen-Invariant Timber Identification.
Evaluates 4 Random Seeds (42, 123, 456, 789) under Stress-Test 50% Specimen Holdout:
  - 5 vouchers/species (55 vouchers) -> Strictly Out-of-Specimen Unseen Physical Test
  - 1 voucher/species  (11 vouchers) -> Strictly Out-of-Specimen Validation Set (honest checkpointing)
  - 4 vouchers/species (44 vouchers) -> Train Pool (80% Train, 20% In-Specimen Seen Test)
Measures:
  - Seen Test Acc & F1
  - Out-of-Specimen Val Acc & F1
  - Out-of-Specimen Unseen Test Acc, F1, ECE
  - Train SRI & Unseen SRI (Linear & MLP Probes)
  - Generalization Gap (GGSL: Seen - Unseen)
  - Paired t-test statistical significance
Outputs:
  - outputs/multi_seed_results.json
  - EXPERIMENT_RESULTS_REPORT.md
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
from typing import Dict, List, Tuple, Optional, Any
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

CURRENT_DIR = Path(__file__).resolve().parent

from scipy import stats

from dataset import (
    DEFAULT_IMAGE_ROOT,
    TARGET_SPECIES,
    scan_and_build_dataset,
    split_dataset_by_seed,
    create_seed_dataloaders,
    split_dataset_naive_random,
    create_naive_dataloaders,
)
from models_gan import (
    StandardBaselineModel,
    AdversarialInvariantGAN,
)
from generate_report import generate_multiseed_report

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
    all_embs, all_preds, all_targets, all_probs, all_raw_probs, all_logits, all_specimens = [], [], [], [], [], [], []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        species_targets = batch["species_idx"].to(device, non_blocking=True)
        specimen_targets = batch["specimen_idx"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            out = model(images)

        logits = out["species_logits"]
        calibrated_logits = out.get("calibrated_logits", logits)
        raw_probs = F.softmax(logits, dim=-1)
        probs = F.softmax(calibrated_logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_embs.append(out["features"].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(species_targets.cpu().numpy())
        all_raw_probs.append(raw_probs.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_specimens.extend(specimen_targets.cpu().numpy())

    feat_dim = all_embs[0].shape[-1] if all_embs else 512
    return {
        "embeddings": np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, feat_dim)),
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
        "logits": np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 11)),
        "raw_probabilities": np.concatenate(all_raw_probs, axis=0) if all_raw_probs else np.empty((0, 11)),
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


def compute_voucher_cluster_bootstrap(
    targets: np.ndarray,
    preds_b: np.ndarray,
    preds_g: np.ndarray,
    probs_b: np.ndarray,
    probs_g: np.ndarray,
    specimens: np.ndarray,
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Voucher-Level Cluster Bootstrap (Physical Specimens as Independent Sampling Units).
    Resamples physical vouchers with replacement (K = 55), gathers all associated tiles,
    and computes empirical 95% Confidence Intervals for:
      - Baseline Unseen Top-1 Acc & F1
      - 'GAN' Unseen Top-1 Acc & F1
      - Delta (GAN - Baseline)
      - Cluster bootstrap two-tailed p-value
    """
    unique_vouchers = np.unique(specimens)
    k = len(unique_vouchers)
    if k < 2 or len(targets) == 0:
        return {}

    rng = np.random.RandomState(seed)
    v_to_indices = {v: np.where(specimens == v)[0] for v in unique_vouchers}

    b_accs, g_accs, d_accs = [], [], []
    b_f1s, g_f1s, d_f1s = [], [], []
    b_eces, g_eces, d_eces = [], [], []

    for _ in range(num_bootstrap):
        sampled_vouchers = rng.choice(unique_vouchers, size=k, replace=True)
        idx_pool = np.concatenate([v_to_indices[v] for v in sampled_vouchers])

        y_t = targets[idx_pool]
        y_b = preds_b[idx_pool]
        y_g = preds_g[idx_pool]
        pr_b = probs_b[idx_pool]
        pr_g = probs_g[idx_pool]

        acc_b = accuracy_score(y_t, y_b) * 100.0
        acc_g = accuracy_score(y_t, y_g) * 100.0
        f1_b = f1_score(y_t, y_b, average="macro", zero_division=0) * 100.0
        f1_g = f1_score(y_t, y_g, average="macro", zero_division=0) * 100.0
        ece_b = compute_ece(pr_b, y_t)
        ece_g = compute_ece(pr_g, y_t)

        b_accs.append(acc_b)
        g_accs.append(acc_g)
        d_accs.append(acc_g - acc_b)

        b_f1s.append(f1_b)
        g_f1s.append(f1_g)
        d_f1s.append(f1_g - f1_b)

        b_eces.append(ece_b)
        g_eces.append(ece_g)
        d_eces.append(ece_g - ece_b)

    d_acc_arr = np.array(d_accs)
    p_val_acc = 2.0 * min(np.mean(d_acc_arr <= 0), np.mean(d_acc_arr >= 0))
    p_val_acc = float(min(1.0, max(0.0, p_val_acc)))

    return {
        "num_bootstrap_replicates": num_bootstrap,
        "num_physical_vouchers": int(k),
        "baseline_unseen_acc_ci95": [float(np.percentile(b_accs, 2.5)), float(np.percentile(b_accs, 97.5))],
        "gan_unseen_acc_ci95": [float(np.percentile(g_accs, 2.5)), float(np.percentile(g_accs, 97.5))],
        "delta_unseen_acc_ci95": [float(np.percentile(d_accs, 2.5)), float(np.percentile(d_accs, 97.5))],
        "baseline_unseen_f1_ci95": [float(np.percentile(b_f1s, 2.5)), float(np.percentile(b_f1s, 97.5))],
        "gan_unseen_f1_ci95": [float(np.percentile(g_f1s, 2.5)), float(np.percentile(g_f1s, 97.5))],
        "delta_unseen_f1_ci95": [float(np.percentile(d_f1s, 2.5)), float(np.percentile(d_f1s, 97.5))],
        "baseline_unseen_ece_ci95": [float(np.percentile(b_eces, 2.5)), float(np.percentile(b_eces, 97.5))],
        "gan_unseen_ece_ci95": [float(np.percentile(g_eces, 2.5)), float(np.percentile(g_eces, 97.5))],
        "delta_unseen_ece_ci95": [float(np.percentile(d_eces, 2.5)), float(np.percentile(d_eces, 97.5))],
        "cluster_bootstrap_p_value": p_val_acc,
    }


def evaluate_model_full(
    model,
    loader_train,
    loader_val,
    loader_seen_test,
    loader_unseen,
    device,
    seed: int = 42,
) -> Dict:
    """
    Evaluates a trained model across all 4 data partitions:
      1. Train Set (loader_train): Measures Train SRI to verify specimen disentanglement.
      2. Out-of-Specimen Val Set (loader_val, 11 vouchers): Honest Out-of-Specimen checkpoint validation.
      3. In-Specimen Seen Test Set (loader_seen_test): In-Distribution generalizability.
      4. Out-of-Specimen Unseen Physical Test Set (loader_unseen, 55 vouchers): 5-voucher real-world stress test.
    """
    train_res = extract_embeddings_and_predictions(model, loader_train, device)
    val_res = extract_embeddings_and_predictions(model, loader_val, device)
    seen_res = extract_embeddings_and_predictions(model, loader_seen_test, device)
    unseen_res = extract_embeddings_and_predictions(model, loader_unseen, device)

    # 1. Validation Set (Out-of-Specimen, 11 vouchers)
    val_acc = float(accuracy_score(val_res["targets"], val_res["predictions"]) * 100.0)
    val_f1 = float(f1_score(val_res["targets"], val_res["predictions"], average="macro", zero_division=0) * 100.0)

    # 2. Seen Test Set (In-Specimen, 44 vouchers)
    seen_acc = float(accuracy_score(seen_res["targets"], seen_res["predictions"]) * 100.0)
    seen_f1 = float(f1_score(seen_res["targets"], seen_res["predictions"], average="macro", zero_division=0) * 100.0)

    # 3. Unseen Test Set (Out-of-Specimen, 5 vouchers/species = 55 vouchers)
    unseen_acc = float(accuracy_score(unseen_res["targets"], unseen_res["predictions"]) * 100.0)
    unseen_f1 = float(f1_score(unseen_res["targets"], unseen_res["predictions"], average="macro", zero_division=0) * 100.0)
    unseen_ece = float(compute_ece(unseen_res["probabilities"], unseen_res["targets"]))
    unseen_raw_ece = float(compute_ece(unseen_res["raw_probabilities"], unseen_res["targets"]))
    opt_temp = float(getattr(model, "temperature", torch.tensor(1.0)).item())

    # 4. SRI Probing on Train Set (Primary metric: Does the model forget specimen identity of learned wood?)
    sri_train_lin, sri_train_mlp = compute_sri_probes(train_res["embeddings"], train_res["specimens"], seed=seed)

    # 5. SRI Probing on Unseen Test Set (Clustering propensity on unseen wood blocks)
    sri_unseen_lin, sri_unseen_mlp = compute_sri_probes(unseen_res["embeddings"], unseen_res["specimens"], seed=seed)

    # Generalization Gaps
    gen_gap_seen_unseen_acc = seen_acc - unseen_acc
    gen_gap_seen_unseen_f1 = seen_f1 - unseen_f1
    gen_gap_val_unseen_acc = val_acc - unseen_acc

    return {
        "val_top1_acc": val_acc,
        "val_macro_f1": val_f1,
        "seen_top1_acc": seen_acc,
        "seen_macro_f1": seen_f1,
        "unseen_top1_acc": unseen_acc,
        "unseen_macro_f1": unseen_f1,
        "unseen_ece": unseen_ece,
        "unseen_raw_ece": unseen_raw_ece,
        "temperature": opt_temp,
        "gen_gap_seen_unseen_acc": gen_gap_seen_unseen_acc,
        "gen_gap_seen_unseen_f1": gen_gap_seen_unseen_f1,
        "gen_gap_val_unseen_acc": gen_gap_val_unseen_acc,
        "sri_train_linear": sri_train_lin,
        "sri_train_mlp": sri_train_mlp,
        "sri_unseen_linear": sri_unseen_lin,
        "sri_unseen_mlp": sri_unseen_mlp,
        "_unseen_targets": unseen_res["targets"],
        "_unseen_predictions": unseen_res["predictions"],
        "_unseen_probabilities": unseen_res["probabilities"],
        "_unseen_specimens": unseen_res["specimens"],
    }


# ==============================================================================
# TRAINING ROUTINES FOR SINGLE SEED
# ==============================================================================
def train_baseline_one_seed(
    loader_train,
    loader_val,
    epochs: int,
    lr: float,
    device,
    backbone: str,
    seed: int,
    label_smoothing: float = 0.05,
):
    """Trains Standard Baseline (ERM) with honest Out-of-Specimen checkpointing and post-hoc temperature calibration."""
    print(f"\n[*] [Seed {seed}] Training Standard Baseline Model ({backbone} ERM | Label Smoothing: {label_smoothing})...")
    model = StandardBaselineModel(backbone_name=backbone, num_classes=len(TARGET_SPECIES), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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

        # Honest validation on Out-of-Specimen Val Set (11 unseen vouchers)
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

        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] | Train Acc: {train_acc:.2f}% | Out-of-Specimen Val Acc: {val_acc:.2f}% (F1: {val_f1:.2f}%)")

    model.load_state_dict(best_weights)

    # Post-hoc Temperature Calibration on Out-of-Specimen Val Set (Guo et al., ICML 2017)
    val_res = extract_embeddings_and_predictions(model, loader_val, device)
    opt_t = model.calibrate_temperature(
        torch.tensor(val_res["logits"], device=device),
        torch.tensor(val_res["targets"], device=device),
    )
    print(f"    [Post-Hoc Calibration] Baseline Temperature on Out-of-Specimen Val: T = {opt_t:.3f}")
    return model


def train_gan_one_seed(
    loader_train,
    loader_val,
    epochs: int,
    lr: float,
    adv_weight: float,
    club_weight: float,
    device,
    backbone: str,
    seed: int,
    num_specimens: int = 44,
    species_specimen_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.05,
):
    """Trains Proposed Adversarial Invariant 'GAN' Model with Species-Conditioned Masked Softmax & Vectorized CLUB."""
    print(f"\n[*] [Seed {seed}] Training Proposed Adversarial Invariant 'GAN' Model ({backbone} GRL + Vectorized CLUB | {num_specimens} train specimens | beta={adv_weight}, mu={club_weight}, label_smoothing={label_smoothing})...")
    if species_specimen_mask is not None:
        species_specimen_mask = species_specimen_mask.to(device)

    model = AdversarialInvariantGAN(
        backbone_name=backbone,
        num_classes=len(TARGET_SPECIES),
        num_specimens=num_specimens,
        pretrained=True,
        adv_weight=adv_weight,
        club_weight=club_weight,
        species_specimen_mask=species_specimen_mask,
    ).to(device)

    criterion_species = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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

        # Honest validation on Out-of-Specimen Val Set (11 unseen vouchers)
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

        if val_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] (alpha={alpha:.2f}) | Train Acc: {train_acc:.2f}% | Out-of-Specimen Val Acc: {val_acc:.2f}% (F1: {val_f1:.2f}%)")

    model.load_state_dict(best_weights)

    # Post-hoc Temperature Calibration on Out-of-Specimen Val Set (Guo et al., ICML 2017)
    val_res = extract_embeddings_and_predictions(model, loader_val, device)
    opt_t = model.calibrate_temperature(
        torch.tensor(val_res["logits"], device=device),
        torch.tensor(val_res["targets"], device=device),
    )
    print(f"    [Post-Hoc Calibration] 'GAN' Temperature on Out-of-Specimen Val: T = {opt_t:.3f}")
    return model


# ==============================================================================
# MAIN MULTI-SEED WORKFLOW & STATISTICAL AUDIT
# ==============================================================================
CANDIDATE_SEEDS = [
    42, 123, 456, 789, 101, 202, 303, 404, 505, 606,
    707, 808, 909, 111, 222, 333, 444, 555, 666, 777,
]


def train_naive_random_leakage_baseline(
    df_raw: pd.DataFrame,
    device,
    backbone: str = "resnet18",
    epochs: int = 15,
    lr: float = 3e-4,
    seed: int = 42,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Trains a Baseline on Naive Image-Level Random Split (where same-specimen pictures are leaked).
    Empirically quantifies the Performance Inflation Gap / Optimism Bias caused by data leakage.
    """
    print("\n" + "=" * 80)
    print("  [LEAKAGE AUDIT] TRAINING NAIVE IMAGE-LEVEL RANDOM SPLIT BASELINE")
    print("  (Quantifying Performance Inflation / Optimism Bias under Specimen Leakage)")
    print("=" * 80)

    df_train_n, df_val_n, df_test_n = split_dataset_naive_random(df_raw, seed=seed)
    loader_train, loader_val, loader_test = create_naive_dataloaders(df_train_n, df_val_n, df_test_n, batch_size=batch_size)

    model = train_baseline_one_seed(
        loader_train,
        loader_val,
        epochs=epochs,
        lr=lr,
        device=device,
        backbone=backbone,
        seed=seed,
        label_smoothing=0.05,
    )

    test_res = extract_embeddings_and_predictions(model, loader_test, device)
    acc = float(accuracy_score(test_res["targets"], test_res["predictions"]) * 100.0)
    f1 = float(f1_score(test_res["targets"], test_res["predictions"], average="macro", zero_division=0) * 100.0)
    ece = float(compute_ece(test_res["probabilities"], test_res["targets"]))

    print(f"\n[+] [LEAKAGE AUDIT RESULT] Naive Random Split Test Acc: {acc:.2f}% | F1: {f1:.2f}% | ECE: {ece:.2f}%")
    return {
        "naive_test_top1_acc": acc,
        "naive_test_macro_f1": f1,
        "naive_test_ece": ece,
        "naive_train_samples": len(df_train_n),
        "naive_test_samples": len(df_test_n),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Specimen-Invariant Benchmark Pipeline")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT, help="Root folder of raw images")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds/splits to evaluate (default: 10, recommended 10-20)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit list of random seeds (overrides --num_seeds if provided)")
    parser.add_argument("--test_vouchers_per_species", type=int, default=5, help="Number of unseen vouchers per species for Test (default: 5 = 55 vouchers, 50%% holdout)")
    parser.add_argument("--val_vouchers_per_species", type=int, default=1, help="Number of unseen vouchers per species for Val (default: 1 = 11 vouchers)")
    parser.add_argument("--epochs", type=int, default=17, help="Training epochs per model (default: 17)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--adv_weight", type=float, default=0.05, help="GRL discriminator loss weight beta (default: 0.05)")
    parser.add_argument("--club_weight", type=float, default=0.001, help="Variational CLUB MI loss weight mu (default: 0.001)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing epsilon for classification loss (default: 0.05)")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Visual backbone architecture (default: resnet18)")
    parser.add_argument("--run_leakage_audit", action="store_true", default=True, help="Train Naive Random Split baseline to measure voucher leakage inflation")
    parser.add_argument("--no_leakage_audit", dest="run_leakage_audit", action="store_false", help="Skip naive random split audit")
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "outputs"), help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds is not None and len(args.seeds) > 0:
        active_seeds = args.seeds
    else:
        active_seeds = CANDIDATE_SEEDS[: args.num_seeds]

    test_v_count = args.test_vouchers_per_species * len(TARGET_SPECIES)
    val_v_count = args.val_vouchers_per_species * len(TARGET_SPECIES)
    train_v_count = 110 - test_v_count - val_v_count

    print("\n" + "=" * 80)
    print("  VOUCHER-AWARE TIMBER FORENSICS BENCHMARK & SPECIMEN LEAKAGE AUDIT")
    print(f"  Target Taxa : 11 Clean Fabaceae Species (110 physical vouchers)")
    print(f"  Protocol    : Stress-Test 50% Specimen Holdout")
    print(f"  Splits      : Train ({train_v_count} vouchers) | Out-of-Specimen Val ({val_v_count} vouchers) | Out-of-Specimen Test ({test_v_count} vouchers)")
    print(f"  Backbone    : {args.backbone} (ResNet-18 Default) | GRL Weight: {args.adv_weight} | CLUB Weight: {args.club_weight} | Label Smoothing: {args.label_smoothing}")
    print(f"  Device      : {device} (cuDNN enabled: {torch.cuda.is_available()})")
    print(f"  Seeds       : {active_seeds} (Total {len(active_seeds)} experimental runs)")
    print(f"  Image Root  : {args.image_root}")
    print("=" * 80 + "\n")

    # Step 1: Scan and build dataset directly from disk
    print("[STEP 1] Scanning and building dataset directly from image folder...")
    df_raw = scan_and_build_dataset(args.image_root)

    # Step 2: Specimen Leakage Audit (Naive Image-Level Random Split Baseline)
    leakage_audit_data = None
    if args.run_leakage_audit:
        try:
            leakage_audit_data = train_naive_random_leakage_baseline(
                df_raw,
                device=device,
                backbone=args.backbone,
                epochs=min(15, args.epochs),
                lr=args.lr,
                seed=active_seeds[0],
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"[!] Warning on Leakage Audit run: {e}")

    results_by_seed = []

    # Step 3: Loop through voucher-level splits
    for seed_idx, seed in enumerate(active_seeds, 1):
        print("\n" + "#" * 80)
        print(f"  [RUN {seed_idx}/{len(active_seeds)}] EXECUTING VOUCHER SPLIT SEED {seed}")
        print("#" * 80)

        # 1. Set seed
        set_seed(seed)

        # 2. Dynamic partitioning for this seed into 4 sets
        df_train, df_val, df_seen_test, df_unseen, species_specimen_mask = split_dataset_by_seed(
            df_raw,
            seed=seed,
            test_vouchers_per_species=args.test_vouchers_per_species,
            val_vouchers_per_species=args.val_vouchers_per_species,
        )

        # 3. Create loaders for all sets
        loader_train, loader_val, loader_seen_test, loader_unseen, info = create_seed_dataloaders(
            df_train,
            df_val,
            df_seen_test,
            df_unseen,
            species_specimen_mask=species_specimen_mask,
            batch_size=args.batch_size,
        )

        # 4. Train Standard Baseline
        baseline_model = train_baseline_one_seed(
            loader_train,
            loader_val,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            backbone=args.backbone,
            seed=seed,
            label_smoothing=args.label_smoothing,
        )

        # 5. Train Proposed Adversarial Invariant 'GAN'
        gan_model = train_gan_one_seed(
            loader_train,
            loader_val,
            epochs=args.epochs,
            lr=args.lr,
            adv_weight=args.adv_weight,
            club_weight=args.club_weight,
            device=device,
            backbone=args.backbone,
            seed=seed,
            num_specimens=info["train_specimens"],
            species_specimen_mask=species_specimen_mask,
            label_smoothing=args.label_smoothing,
        )

        # 6. Evaluate Baseline across all partitions
        print(f"\n[*] [Seed {seed}] Auditing Standard Baseline Model across all partitions...")
        base_metrics = evaluate_model_full(baseline_model, loader_train, loader_val, loader_seen_test, loader_unseen, device=device, seed=seed)

        # 7. Evaluate GAN across all partitions
        print(f"[*] [Seed {seed}] Auditing Proposed 'GAN' Model across all partitions...")
        gan_metrics = evaluate_model_full(gan_model, loader_train, loader_val, loader_seen_test, loader_unseen, device=device, seed=seed)

        # 8. Voucher-Level Cluster Bootstrap for this seed
        boot_ci = compute_voucher_cluster_bootstrap(
            targets=base_metrics["_unseen_targets"],
            preds_b=base_metrics["_unseen_predictions"],
            preds_g=gan_metrics["_unseen_predictions"],
            probs_b=base_metrics["_unseen_probabilities"],
            probs_g=gan_metrics["_unseen_probabilities"],
            specimens=base_metrics["_unseen_specimens"],
            num_bootstrap=1000,
            seed=seed,
        )

        # Print detailed summary for this seed
        print(f"\n--- [Seed {seed} Executive Summary: Stress-Test 55 Unseen Vouchers] ---")
        print(f"  Baseline -> Val Acc: {base_metrics['val_top1_acc']:.2f}% | Seen Test Acc: {base_metrics['seen_top1_acc']:.2f}% | Unseen Test Acc: {base_metrics['unseen_top1_acc']:.2f}% (95% CI: [{boot_ci.get('baseline_unseen_acc_ci95', [0,0])[0]:.2f}%, {boot_ci.get('baseline_unseen_acc_ci95', [0,0])[1]:.2f}%]) | ECE: {base_metrics['unseen_ece']:.2f}%")
        print(f"  'GAN'    -> Val Acc: {gan_metrics['val_top1_acc']:.2f}% | Seen Test Acc: {gan_metrics['seen_top1_acc']:.2f}% | Unseen Test Acc: {gan_metrics['unseen_top1_acc']:.2f}% (95% CI: [{boot_ci.get('gan_unseen_acc_ci95', [0,0])[0]:.2f}%, {boot_ci.get('gan_unseen_acc_ci95', [0,0])[1]:.2f}%]) | ECE: {gan_metrics['unseen_ece']:.2f}%")
        print(f"  Diff (GAN - Base): Delta Acc = {gan_metrics['unseen_top1_acc'] - base_metrics['unseen_top1_acc']:+.2f} pp (Cluster Bootstrap 95% CI: [{boot_ci.get('delta_unseen_acc_ci95', [0,0])[0]:+.2f} pp, {boot_ci.get('delta_unseen_acc_ci95', [0,0])[1]:+.2f} pp], p_boot = {boot_ci.get('cluster_bootstrap_p_value', 1.0):.4f})")

        # Clean raw arrays before saving JSON
        clean_base = {k: v for k, v in base_metrics.items() if not k.startswith("_")}
        clean_gan = {k: v for k, v in gan_metrics.items() if not k.startswith("_")}

        results_by_seed.append({
            "seed": seed,
            "baseline": clean_base,
            "gan": clean_gan,
            "voucher_cluster_bootstrap": boot_ci,
        })

    # Step 4: Compute aggregate statistics across all seeds
    print("\n" + "=" * 80)
    print("  AGGREGATING MULTI-SEED BENCHMARK STATISTICS & STATISTICAL TESTING")
    print("=" * 80)

    metric_keys = [
        "val_top1_acc", "val_macro_f1",
        "seen_top1_acc", "seen_macro_f1",
        "unseen_top1_acc", "unseen_macro_f1",
        "unseen_ece", "unseen_raw_ece", "temperature",
        "gen_gap_seen_unseen_acc", "gen_gap_seen_unseen_f1",
        "gen_gap_val_unseen_acc",
        "sri_train_linear", "sri_train_mlp",
        "sri_unseen_linear", "sri_unseen_mlp",
    ]

    n_runs = len(active_seeds)
    t_crit = float(stats.t.ppf(0.975, df=n_runs - 1)) if n_runs > 1 else 1.96

    def calc_ci(values: List[float]) -> Dict[str, Any]:
        arr = np.array(values, dtype=float)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_v = float(std_v / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        ci95 = [float(mean_v - t_crit * sem_v), float(mean_v + t_crit * sem_v)]
        return {
            "mean": mean_v,
            "std": std_v,
            "sem": sem_v,
            "ci95": ci95,
            "values": [float(x) for x in arr],
        }

    summary = {"baseline": {}, "gan": {}, "comparison": {}, "statistical_tests": {}}

    for key in metric_keys:
        b_vals = [r["baseline"][key] for r in results_by_seed]
        g_vals = [r["gan"][key] for r in results_by_seed]
        d_vals = [g - b for g, b in zip(g_vals, b_vals)]

        summary["baseline"][key] = calc_ci(b_vals)
        summary["gan"][key] = calc_ci(g_vals)
        summary["comparison"][key] = {
            "delta_mean": float(np.mean(d_vals)),
            "delta_std": float(np.std(d_vals, ddof=1)) if len(d_vals) > 1 else 0.0,
            "delta_ci95": [float(np.mean(d_vals) - t_crit * (np.std(d_vals, ddof=1)/np.sqrt(n_runs))),
                           float(np.mean(d_vals) + t_crit * (np.std(d_vals, ddof=1)/np.sqrt(n_runs)))] if n_runs > 1 else [float(np.mean(d_vals)), float(np.mean(d_vals))],
            "delta_values": d_vals,
        }

    # Statistical significance testing across seeds (Paired t-test & Wilcoxon)
    b_accs = [r["baseline"]["unseen_top1_acc"] for r in results_by_seed]
    g_accs = [r["gan"]["unseen_top1_acc"] for r in results_by_seed]
    d_accs = np.array(g_accs) - np.array(b_accs)

    b_f1s = [r["baseline"]["unseen_macro_f1"] for r in results_by_seed]
    g_f1s = [r["gan"]["unseen_macro_f1"] for r in results_by_seed]
    d_f1s = np.array(g_f1s) - np.array(b_f1s)

    stat_report = {}
    if n_runs >= 2 and np.std(d_accs) > 1e-8:
        t_res_acc = stats.ttest_rel(g_accs, b_accs)
        stat_report["paired_ttest_acc"] = {
            "t_statistic": float(t_res_acc.statistic),
            "p_value": float(t_res_acc.pvalue),
            "significant_005": bool(t_res_acc.pvalue < 0.05),
        }
        t_res_f1 = stats.ttest_rel(g_f1s, b_f1s)
        stat_report["paired_ttest_f1"] = {
            "t_statistic": float(t_res_f1.statistic),
            "p_value": float(t_res_f1.pvalue),
            "significant_005": bool(t_res_f1.pvalue < 0.05),
        }
    else:
        stat_report["paired_ttest_acc"] = {"t_statistic": 0.0, "p_value": 1.0, "significant_005": False}
        stat_report["paired_ttest_f1"] = {"t_statistic": 0.0, "p_value": 1.0, "significant_005": False}

    # Wilcoxon signed-rank test
    try:
        if n_runs >= 5 and np.std(d_accs) > 1e-8:
            w_res_acc = stats.wilcoxon(g_accs, b_accs)
            stat_report["wilcoxon_acc"] = {
                "w_statistic": float(w_res_acc.statistic),
                "p_value": float(w_res_acc.pvalue),
                "significant_005": bool(w_res_acc.pvalue < 0.05),
            }
            w_res_f1 = stats.wilcoxon(g_f1s, b_f1s)
            stat_report["wilcoxon_f1"] = {
                "w_statistic": float(w_res_f1.statistic),
                "p_value": float(w_res_f1.pvalue),
                "significant_005": bool(w_res_f1.pvalue < 0.05),
            }
        else:
            stat_report["wilcoxon_acc"] = {"w_statistic": None, "p_value": None, "note": "Need n>=5 splits"}
            stat_report["wilcoxon_f1"] = {"w_statistic": None, "p_value": None, "note": "Need n>=5 splits"}
    except Exception as e:
        stat_report["wilcoxon_acc"] = {"error": str(e)}

    summary["statistical_tests"] = stat_report

    # Step 5: Specimen Leakage Inflation Gap Analysis
    leakage_summary = {}
    if leakage_audit_data is not None:
        naive_acc = leakage_audit_data["naive_test_top1_acc"]
        naive_f1 = leakage_audit_data["naive_test_macro_f1"]
        base_unseen_acc = summary["baseline"]["unseen_top1_acc"]["mean"]
        base_unseen_f1 = summary["baseline"]["unseen_macro_f1"]["mean"]

        inflation_acc = naive_acc - base_unseen_acc
        inflation_f1 = naive_f1 - base_unseen_f1

        leakage_summary = {
            "naive_random_split_test_acc": naive_acc,
            "naive_random_split_test_f1": naive_f1,
            "naive_random_split_test_ece": leakage_audit_data["naive_test_ece"],
            "out_of_specimen_baseline_test_acc": base_unseen_acc,
            "out_of_specimen_baseline_test_f1": base_unseen_f1,
            "specimen_leakage_inflation_acc_gap_pp": float(inflation_acc),
            "specimen_leakage_inflation_f1_gap_pp": float(inflation_f1),
        }
        print("\n" + "=" * 80)
        print("  [LEAKAGE AUDIT SUMMARY: PROOF OF OPTIMISM BIAS]")
        print(f"  Naive Random Image-Level Split Acc : {naive_acc:.2f}% (Voucher Leakage Present)")
        print(f"  Honest Out-of-Specimen Test Acc     : {base_unseen_acc:.2f}% (55 Unseen Vouchers)")
        print(f"  Specimen Leakage Inflation Gap (Δ)  : +{inflation_acc:.2f} pp")
        print("=" * 80)

    full_output = {
        "benchmark_meta": {
            "seeds": active_seeds,
            "num_seeds": len(active_seeds),
            "epochs": args.epochs,
            "backbone": args.backbone,
            "num_species": len(TARGET_SPECIES),
            "total_physical_vouchers": 110,
            "test_vouchers_per_species": args.test_vouchers_per_species,
            "val_vouchers_per_species": args.val_vouchers_per_species,
            "num_test_vouchers": test_v_count,
            "num_val_vouchers": val_v_count,
            "num_train_vouchers": train_v_count,
            "adv_weight": args.adv_weight,
            "club_weight": args.club_weight,
            "label_smoothing": args.label_smoothing,
            "eval_sets": [
                f"Train Set (In-Specimen, {train_v_count} vouchers)",
                f"Val Set (Out-of-Specimen, {val_v_count} vouchers)",
                f"Seen Test Set (In-Specimen, {train_v_count} vouchers)",
                f"Unseen Test Set (Out-of-Specimen, {test_v_count} vouchers)",
            ],
            "image_root": str(args.image_root),
        },
        "leakage_audit": leakage_summary,
        "per_seed_results": results_by_seed,
        "aggregate_summary": summary,
    }

    results_json_path = out_dir / "multi_seed_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[+] Lưu kết quả multi-seed JSON thành công tại: {results_json_path}")

    # Step 6: Generate publication markdown report
    generate_multiseed_report(results_json_path, CURRENT_DIR / "EXPERIMENT_RESULTS_REPORT.md")


if __name__ == "__main__":
    main()
