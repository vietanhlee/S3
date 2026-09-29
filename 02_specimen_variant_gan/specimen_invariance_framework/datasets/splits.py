"""
specimen_invariance_framework/datasets/splits.py
================================================
Generates:
1. Round-Robin Leave-One-Specimen-Out (LOSO) splits (R = 5 to 10 folds) for rigorous
   paired evaluation across all invariance baselines.
2. Matched Leaky Stratified splits of identical image volume to compute the
   Generalization Gap from Specimen Leakage (GGSL).
"""

import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def generate_round_robin_loso_splits(
    metadata_df: pd.DataFrame,
    num_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Constructs R round-robin LOSO folds.
    
    For each multi-specimen species (|G_c| > 1), physical specimens are systematically
    rotated into held-out validation and test sets so that every method is tested
    on strictly unseen physical wood blocks.
    
    Single-specimen species (|G_c| == 1) are flagged: their images are split in a
    stratified 60/20/20 proportion to maintain taxonomic presence, and they are
    automatically excluded from the specimen discriminator branch.
    
    Returns:
        List of length `num_folds`, each entry is a dict:
        {"train": df_train, "val": df_val, "test": df_test, "fold": fold_idx}
    """
    rng = random.Random(seed)
    species_list = sorted(metadata_df["class_name"].unique())
    
    # Map species -> list of physical specimens
    species_specimens: Dict[str, List[str]] = {}
    for sp in species_list:
        specs = sorted(metadata_df[metadata_df["class_name"] == sp]["specimen_id"].unique())
        # Shuffle deterministically per seed
        specs_shuffled = list(specs)
        rng.shuffle(specs_shuffled)
        species_specimens[sp] = specs_shuffled
        
    folds_data: List[Dict[str, pd.DataFrame]] = []
    
    for r in range(num_folds):
        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []
        
        for sp in species_list:
            specs = species_specimens[sp]
            n_specs = len(specs)
            sp_df = metadata_df[metadata_df["class_name"] == sp]
            
            if n_specs == 1:
                # Structural single-specimen case (e.g. Dalbergia cochinchinensis)
                # Split images proportionally 60/20/20
                indices = sp_df.index.tolist()
                rng.shuffle(indices)
                n_train = int(0.6 * len(indices))
                n_val = int(0.2 * len(indices))
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
            elif n_specs == 2:
                # 2 specimens: rotate 1 for test, 1 for train/val
                test_spec = specs[r % 2]
                train_val_spec = specs[(r + 1) % 2]
                
                test_indices.extend(sp_df[sp_df["specimen_id"] == test_spec].index.tolist())
                tv_indices = sp_df[sp_df["specimen_id"] == train_val_spec].index.tolist()
                rng.shuffle(tv_indices)
                n_val = max(1, int(0.25 * len(tv_indices)))
                val_indices.extend(tv_indices[:n_val])
                train_indices.extend(tv_indices[n_val:])
            else:
                # >= 3 specimens: 1 strictly disjoint specimen for test, 1 for val, rest for train
                test_spec = specs[r % n_specs]
                val_spec = specs[(r + 1) % n_specs]
                train_specs = [s for s in specs if s not in (test_spec, val_spec)]
                
                test_indices.extend(sp_df[sp_df["specimen_id"] == test_spec].index.tolist())
                val_indices.extend(sp_df[sp_df["specimen_id"] == val_spec].index.tolist())
                train_indices.extend(sp_df[sp_df["specimen_id"].isin(train_specs)].index.tolist())
                
        folds_data.append({
            "fold": r,
            "train": metadata_df.loc[train_indices].copy(),
            "val": metadata_df.loc[val_indices].copy(),
            "test": metadata_df.loc[test_indices].copy(),
        })
        
    return folds_data


def generate_leaky_stratified_split(
    loso_fold: Dict[str, pd.DataFrame],
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generates a matched naive stratified image-level split with the exact same
    sample counts (N_train, N_val, N_test) as the corresponding LOSO fold.
    
    This provides an empirical control to calculate GGSL = Acc_leaky - Acc_LOSO.
    """
    total_df = pd.concat([loso_fold["train"], loso_fold["val"], loso_fold["test"]], ignore_index=True)
    n_total = len(total_df)
    n_train = len(loso_fold["train"])
    n_val = len(loso_fold["val"])
    n_test = len(loso_fold["test"])
    
    test_ratio = n_test / n_total
    val_ratio = n_val / (n_train + n_val)
    
    # 1. Stratified split for test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(total_df, total_df["class_name"]))
    
    df_train_val = total_df.iloc[train_val_idx].reset_index(drop=True)
    df_test = total_df.iloc[test_idx].reset_index(drop=True)
    
    # 2. Stratified split for train vs val
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss_val.split(df_train_val, df_train_val["class_name"]))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    
    return {
        "fold": loso_fold["fold"],
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }
