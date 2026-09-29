"""
specimen_invariance_framework/models/heads.py
=============================================
Output heads for species classification and specimen discrimination:
1. SpeciesClassifier: Standard Linear projection or ArcFace angular margin head.
2. ConditionalDiscriminator: Specimen prediction conditioned on species (masked softmax),
   directly targeting I(Z; P | Y) -> 0 and automatically masking single-specimen species.
3. UnconditionalDiscriminator: Standard DANN baseline predicting global specimen identity.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace head implementing Additive Angular Margin Loss.
    Reference: Deng et al., CVPR 2019.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        # L2-normalize features and weights
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SpeciesClassifier(nn.Module):
    """Species identification head."""
    def __init__(
        self,
        in_features: int,
        num_species: int = 19,
        head_type: str = "linear",
        arcface_s: float = 30.0,
        arcface_m: float = 0.50,
    ):
        super().__init__()
        self.head_type = head_type.lower()
        if self.head_type == "arcface":
            self.head = ArcMarginProduct(in_features, num_species, s=arcface_s, m=arcface_m)
        else:
            self.head = nn.Linear(in_features, num_species)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.head_type == "arcface":
            return self.head(x, label)
        return self.head(x)


class ConditionalDiscriminator(nn.Module):
    """
    Species-Conditioned Specimen Discriminator.
    
    Predicts physical specimen identity ONLY within the set of specimens belonging to that species.
    For species c with |G_c| specimens, it outputs an S_c-dimensional logit vector.
    
    Species with |G_c| == 1 are structural singletons: "specimen" and "species" are mathematically
    identical, so they are automatically bypassed (loss = 0).
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        specimen_counts: Dict[int, int],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Dedicated per-species discriminator heads
        self.num_species = len(specimen_counts)
        self.per_species_heads = nn.ModuleDict()
        self.valid_species_mask: Dict[int, bool] = {}
        
        for sp_idx, count in specimen_counts.items():
            if count > 1:
                self.per_species_heads[str(sp_idx)] = nn.Linear(hidden_dim, count)
                self.valid_species_mask[sp_idx] = True
            else:
                self.valid_species_mask[sp_idx] = False

    def forward(
        self,
        z: torch.Tensor,
        species_targets: torch.Tensor,
        local_specimen_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Computes conditional specimen discrimination cross-entropy loss.
        
        Returns:
            (loss, num_evaluated_samples)
        """
        h = self.shared_mlp(z)
        total_loss = torch.tensor(0.0, device=z.device)
        total_valid = 0

        unique_species = torch.unique(species_targets).tolist()
        for sp in unique_species:
            if not self.valid_species_mask.get(sp, False):
                # Structural singleton: ignore
                continue
                
            mask = (species_targets == sp)
            sp_h = h[mask]
            sp_targets = local_specimen_targets[mask]
            
            head = self.per_species_heads[str(sp)]
            logits = head(sp_h)
            loss_sp = F.cross_entropy(logits, sp_targets, reduction="sum")
            
            total_loss = total_loss + loss_sp
            total_valid += sp_targets.size(0)

        if total_valid > 0:
            total_loss = total_loss / total_valid
            
        return total_loss, total_valid


class UnconditionalDiscriminator(nn.Module):
    """
    Standard DANN baseline: discriminates global specimen ID across all 148 physical blocks
    without species conditioning.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_total_specimens: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_total_specimens),
        )

    def forward(self, z: torch.Tensor, global_specimen_targets: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(z)
        return F.cross_entropy(logits, global_specimen_targets)
