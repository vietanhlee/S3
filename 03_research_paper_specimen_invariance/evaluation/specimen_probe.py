"""
specimen_invariance_framework/evaluation/specimen_probe.py
=========================================================
Specimen Recoverability Probe & Normalized Specimen Recoverability Index (SRI).

Protocol:
1. Freeze visual embeddings from the trained model.
2. For each species c with |G_c| > 1:
   - Partition images of EACH specimen exactly 50/50 into probe-train and probe-test.
   - Train a linear probe to classify specimen identity within that species.
   - Evaluate probe test accuracy Acc_probe(c).
   - Normalize against random guessing baseline (1 / |G_c|):
         SRI_c = (Acc_probe(c) - 1 / |G_c|) / (1 - 1 / |G_c|)
3. Compute mean SRI across all valid multi-specimen taxa.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def extract_all_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extracts features for all images in loader."""
    model.eval()
    all_embs = []
    all_species = []
    all_specimens = []
    all_image_ids = []
    
    for batch in loader:
        images = batch["image"].to(device)
        species_idx = batch["species_idx"].cpu().numpy()
        local_spec_idx = batch["local_specimen_idx"].cpu().numpy()
        image_ids = batch["image_id"]
        
        embs = model.extract_features(images).cpu().numpy()
        all_embs.append(embs)
        all_species.append(species_idx)
        all_specimens.append(local_spec_idx)
        all_image_ids.extend(image_ids)
        
    return (
        np.concatenate(all_embs, axis=0),
        np.concatenate(all_species, axis=0),
        np.concatenate(all_specimens, axis=0),
        all_image_ids,
    )


def evaluate_specimen_recoverability(
    embeddings: np.ndarray,
    species_labels: np.ndarray,
    specimen_labels: np.ndarray,
    specimen_counts: Dict[int, int],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates how much specimen identity can be recovered from frozen embeddings.
    
    Returns:
        dict containing:
        - 'mean_sri': Mean Specimen Recoverability Index in [0, 1]
        - 'per_species_sri': Dict mapping species_idx -> SRI
        - 'per_species_acc': Dict mapping species_idx -> probe test accuracy
        - 'per_species_chance': Dict mapping species_idx -> random chance level
    """
    rng = np.random.RandomState(seed)
    unique_species = np.unique(species_labels)
    
    per_species_sri: Dict[int, float] = {}
    per_species_acc: Dict[int, float] = {}
    per_species_chance: Dict[int, float] = {}
    
    valid_sri_list: List[float] = []
    
    for sp in unique_species:
        num_specs = specimen_counts.get(int(sp), 1)
        if num_specs <= 1:
            # Structural singleton: cannot probe within species
            continue
            
        sp_mask = (species_labels == sp)
        sp_embs = embeddings[sp_mask]
        sp_specs = specimen_labels[sp_mask]
        
        # 50/50 split of images FOR EACH SPECIMEN individually
        train_indices: List[int] = []
        test_indices: List[int] = []
        
        for spec_id in np.unique(sp_specs):
            spec_indices = np.where(sp_specs == spec_id)[0]
            rng.shuffle(spec_indices)
            split_point = max(1, len(spec_indices) // 2)
            train_indices.extend(spec_indices[:split_point])
            test_indices.extend(spec_indices[split_point:])
            
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
            
        X_train, y_train = sp_embs[train_indices], sp_specs[train_indices]
        X_test, y_test = sp_embs[test_indices], sp_specs[test_indices]
        
        # Verify that test set has valid classes
        if len(np.unique(y_train)) < 2:
            continue
            
        # Standardize embeddings with StandardScaler and fit linear probe (solver=lbfgs, max_iter=2000)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed, solver="lbfgs")
        )
        clf.fit(X_train, y_train)
        probe_acc = float(clf.score(X_test, y_test))
        
        chance_level = 1.0 / num_specs
        # Normalized Specimen Recoverability Index
        sri = (probe_acc - chance_level) / max(1e-6, 1.0 - chance_level)
        sri_clamped = max(0.0, min(1.0, sri))
        
        per_species_acc[int(sp)] = probe_acc
        per_species_chance[int(sp)] = chance_level
        per_species_sri[int(sp)] = sri_clamped
        valid_sri_list.append(sri_clamped)
        
    mean_sri = float(np.mean(valid_sri_list)) if len(valid_sri_list) > 0 else 0.0
    
    return {
        "mean_sri": mean_sri,
        "per_species_sri": per_species_sri,
        "per_species_acc": per_species_acc,
        "per_species_chance": per_species_chance,
    }
