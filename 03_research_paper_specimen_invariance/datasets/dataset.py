"""
specimen_invariance_framework/datasets/dataset.py
=================================================
TimberDataset class supporting species labels, global specimen IDs, and
species-conditional specimen IDs with single-specimen exclusion masks.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset


def build_taxonomy_mappings(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build structured indexing dictionaries across species and specimens.
    
    Returns:
        dict containing:
        - species_to_idx: map species name -> integer index [0, C-1]
        - idx_to_species: map integer index -> species name
        - specimen_to_global_idx: map specimen_id -> global integer [0, S-1]
        - species_to_specimens: map species_idx -> list of specimen_ids
        - specimen_to_local_idx: map (species_idx, specimen_id) -> local index [0, S_c - 1]
        - single_specimen_species_mask: boolean tensor of shape (C,), True if species has |G_c| == 1
        - max_specimens_per_species: max value of |G_c| across all species
    """
    species_list = sorted(metadata_df["class_name"].unique())
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    idx_to_species = {i: sp for i, sp in enumerate(species_list)}
    
    unique_specimens = sorted(metadata_df["specimen_id"].unique())
    specimen_to_global_idx = {spec: i for i, spec in enumerate(unique_specimens)}
    
    species_to_specimens: Dict[int, List[str]] = {i: [] for i in range(len(species_list))}
    specimen_to_local_idx: Dict[Tuple[int, str], int] = {}
    
    for _, row in metadata_df[["class_name", "specimen_id"]].drop_duplicates().iterrows():
        sp_idx = species_to_idx[row["class_name"]]
        spec_id = row["specimen_id"]
        if spec_id not in species_to_specimens[sp_idx]:
            species_to_specimens[sp_idx].append(spec_id)
            
    # Sort specimens within each species for deterministic indexing
    for sp_idx in species_to_specimens:
        species_to_specimens[sp_idx] = sorted(species_to_specimens[sp_idx])
        for local_idx, spec_id in enumerate(species_to_specimens[sp_idx]):
            specimen_to_local_idx[(sp_idx, spec_id)] = local_idx
            
    num_species = len(species_list)
    single_specimen_mask = torch.zeros(num_species, dtype=torch.bool)
    specimen_counts = {}
    for sp_idx, specs in species_to_specimens.items():
        specimen_counts[sp_idx] = len(specs)
        if len(specs) <= 1:
            single_specimen_mask[sp_idx] = True
            
    max_specimens = max(len(specs) for specs in species_to_specimens.values())
    
    return {
        "species_to_idx": species_to_idx,
        "idx_to_species": idx_to_species,
        "specimen_to_global_idx": specimen_to_global_idx,
        "species_to_specimens": species_to_specimens,
        "specimen_to_local_idx": specimen_to_local_idx,
        "single_specimen_species_mask": single_specimen_mask,
        "specimen_counts": specimen_counts,
        "max_specimens_per_species": max_specimens,
        "num_species": num_species,
        "num_total_specimens": len(unique_specimens),
    }


class TimberDataset(Dataset):
    """
    PyTorch Dataset for timber macroscopic end-grain captures.
    
    Yields sample dictionary:
    - 'image': Tensor of shape (3, H, W)
    - 'species_idx': LongTensor scalar
    - 'global_specimen_idx': LongTensor scalar
    - 'local_specimen_idx': LongTensor scalar (specimen index within its species)
    - 'is_single_specimen': BoolTensor scalar (True if species has only 1 specimen)
    - 'image_id': str
    - 'image_path': str
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        mappings: Dict[str, Any],
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.mappings = mappings
        self.transform = transform
        
        # Pre-extract DataFrame columns to native Python lists for instant O(1) indexing
        species_names = self.df["class_name"].tolist()
        specimen_ids = self.df["specimen_id"].tolist()
        self.image_ids = self.df["image_id"].tolist()
        
        # Precompute integer indices to eliminate dict lookups in worker processes
        self.species_indices = [self.mappings["species_to_idx"][sp] for sp in species_names]
        self.global_spec_indices = [self.mappings["specimen_to_global_idx"][spec] for spec in specimen_ids]
        self.local_spec_indices = [
            self.mappings["specimen_to_local_idx"][(sp_idx, spec)]
            for sp_idx, spec in zip(self.species_indices, specimen_ids)
        ]
        self.is_singles = [
            self.mappings["single_specimen_species_mask"][sp_idx].item()
            for sp_idx in self.species_indices
        ]
        
        # Pre-resolve relative paths against image_root
        self.image_paths: List[Path] = []
        for p in self.df["image_path"]:
            candidate = self.image_root / p
            if not candidate.exists():
                candidate_alt = self.image_root / "images" / p
                if candidate_alt.exists():
                    candidate = candidate_alt
            self.image_paths.append(candidate)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        
        # Load image safely with immediate descriptor release
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            "image": image,
            "species_idx": torch.tensor(self.species_indices[idx], dtype=torch.long),
            "global_specimen_idx": torch.tensor(self.global_spec_indices[idx], dtype=torch.long),
            "local_specimen_idx": torch.tensor(self.local_spec_indices[idx], dtype=torch.long),
            "is_single_specimen": torch.tensor(self.is_singles[idx], dtype=torch.bool),
            "image_id": self.image_ids[idx],
            "image_path": str(img_path),
        }
