"""
specimen_invariance_framework/datasets/samplers.py
==================================================
Specimen-balanced batch samplers to ensure each mini-batch possesses
sufficient physical specimen diversity across all sampled species.
"""

import math
import random
from typing import Dict, List, Iterator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class SpecimenBalancedBatchSampler(Sampler[List[int]]):
    """
    Batch Sampler that guarantees physical specimen diversity within every mini-batch.
    
    In each batch, it selects `n_specimens` distinct physical specimens and draws
    `samples_per_specimen` images from each selected specimen, resulting in:
        batch_size = n_specimens * samples_per_specimen.
        
    If total batch_size does not cleanly divide, it adjusts samples_per_specimen.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        samples_per_specimen: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.df = df
        self.batch_size = batch_size
        self.samples_per_specimen = max(1, samples_per_specimen)
        self.n_specimens_per_batch = max(2, batch_size // self.samples_per_specimen)
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Group image indices by specimen_id
        self.specimen_to_indices: Dict[str, List[int]] = {}
        for idx, spec_id in enumerate(self.df["specimen_id"]):
            if spec_id not in self.specimen_to_indices:
                self.specimen_to_indices[spec_id] = []
            self.specimen_to_indices[spec_id].append(idx)
            
        self.unique_specimens = list(self.specimen_to_indices.keys())
        self.total_samples = len(self.df)
        self.num_batches = math.ceil(self.total_samples / (self.n_specimens_per_batch * self.samples_per_specimen))

    def __iter__(self) -> Iterator[List[int]]:
        # Clone index pools for each specimen
        specimen_pools = {
            spec: list(indices) for spec, indices in self.specimen_to_indices.items()
        }
        if self.shuffle:
            for spec in specimen_pools:
                self.rng.shuffle(specimen_pools[spec])
                
        specimen_queue = list(self.unique_specimens)
        if self.shuffle:
            self.rng.shuffle(specimen_queue)
            
        for _ in range(self.num_batches):
            batch: List[int] = []
            
            # Select n_specimens for this batch
            if len(specimen_queue) < self.n_specimens_per_batch:
                refill = list(self.unique_specimens)
                if self.shuffle:
                    self.rng.shuffle(refill)
                specimen_queue.extend(refill)
                
            selected_specimens = [specimen_queue.pop(0) for _ in range(self.n_specimens_per_batch)]
            
            for spec in selected_specimens:
                pool = specimen_pools[spec]
                if len(pool) < self.samples_per_specimen:
                    # Refill pool from master indices
                    refill = list(self.specimen_to_indices[spec])
                    if self.shuffle:
                        self.rng.shuffle(refill)
                    pool.extend(refill)
                    
                for _ in range(self.samples_per_specimen):
                    batch.append(pool.pop(0))
                    
            if self.shuffle:
                self.rng.shuffle(batch)
                
            yield batch

    def __len__(self) -> int:
        return self.num_batches
