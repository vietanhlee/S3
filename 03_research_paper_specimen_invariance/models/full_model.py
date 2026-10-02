"""
specimen_invariance_framework/models/full_model.py
==================================================
Unified model wrapper integrating visual backbone, species classification head,
gradient reversal layer, and species-conditioned specimen discriminator.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.backbones import build_backbone
    from models.grl import GradientReversalLayer
    from models.heads import SpeciesClassifier, ConditionalDiscriminator, UnconditionalDiscriminator
    from models.club import CLUBDiscrete
    from config import ModelConfig
except (ImportError, ValueError):
    try:
        from .backbones import build_backbone
        from .grl import GradientReversalLayer
        from .heads import SpeciesClassifier, ConditionalDiscriminator, UnconditionalDiscriminator
        from .club import CLUBDiscrete
        from ..config import ModelConfig
    except (ImportError, ValueError):
        from backbones import build_backbone
        from grl import GradientReversalLayer
        from heads import SpeciesClassifier, ConditionalDiscriminator, UnconditionalDiscriminator
        from club import CLUBDiscrete
        from config import ModelConfig


class SpecimenInvariantModel(nn.Module):
    """
    Unified Architecture for Specimen-Invariant Wood Identification.
    
    Components:
    1. Visual Backbone -> pooled representation z in R^D
    2. Species Classifier Head -> logits over 19 Fabaceae species
    3. GRL -> inverts gradient backward to backbone by -lambda_adv
    4. Conditional Discriminator -> predicts specimen conditioned on species
    5. Projection Head -> maps z -> unit hypersphere S^(d-1) for metric learning / CLUB
    """
    def __init__(
        self,
        config: ModelConfig,
        specimen_counts: Dict[int, int],
        num_total_specimens: int = 148,
    ):
        super().__init__()
        self.config = config
        self.backbone, self.feat_dim = build_backbone(
            backbone_name=config.backbone_name,
            pretrained=config.pretrained
        )
        
        # 1. Species classifier
        self.species_head = SpeciesClassifier(
            in_features=self.feat_dim,
            num_species=len(specimen_counts),
            head_type=config.species_head_type,
            arcface_s=config.arcface_scale,
            arcface_m=config.arcface_margin,
        )
        
        # 2. Gradient Reversal Layer
        self.grl = GradientReversalLayer(
            gamma=config.adv_gamma,
            max_lambda=config.max_lambda_adv
        )
        
        # 3. Conditional discriminator (proposed)
        self.cond_discriminator = ConditionalDiscriminator(
            in_features=self.feat_dim,
            hidden_dim=config.discriminator_hidden_dim,
            specimen_counts=specimen_counts,
            dropout=config.discriminator_dropout,
        )
        
        # 4. Unconditional discriminator (DANN baseline)
        self.uncond_discriminator = UnconditionalDiscriminator(
            in_features=self.feat_dim,
            hidden_dim=config.discriminator_hidden_dim,
            num_total_specimens=num_total_specimens,
            dropout=config.discriminator_dropout,
        )
        
        # 5. Projection head for metric learning (Triplet/SupCon)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feat_dim, config.projection_dim),
            nn.BatchNorm1d(config.projection_dim),
        )
        
        # 6. Optional CLUB MI estimator
        self.club = CLUBDiscrete(
            feature_dim=self.feat_dim,
            num_classes=num_total_specimens,
            hidden_dim=config.discriminator_hidden_dim,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts pooled backbone representations z in R^D."""
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        species_targets: Optional[torch.Tensor] = None,
        local_specimen_targets: Optional[torch.Tensor] = None,
        global_specimen_targets: Optional[torch.Tensor] = None,
        training_progress: float = 0.0,
        return_projection: bool = False,
        return_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            species_targets: Ground truth species indices (B,)
            local_specimen_targets: Local specimen index within species (B,)
            global_specimen_targets: Global specimen index (B,)
            training_progress: Float p in [0, 1] for lambda_adv annealing
            return_projection: Whether to compute and return normalized metric projection
            return_embedding: Whether to return raw embedding z
        """
        # 1. Feature extraction
        z = self.extract_features(x)
        
        # 2. Species logits
        species_logits = self.species_head(z, species_targets)
        
        out: Dict[str, Any] = {
            "species_logits": species_logits,
        }
        
        if return_embedding:
            out["embedding"] = z
            
        if return_projection:
            # 3. Metric projection (only calculated when needed, saving VRAM and compute)
            proj = self.projection_head(z)
            out["projected"] = F.normalize(proj, p=2, dim=-1)
        
        if self.training:
            # Update GRL lambda_adv
            current_lambda = self.grl.update_lambda(training_progress)
            out["lambda_adv"] = torch.tensor([current_lambda], dtype=torch.float32, device=x.device)
            
            # Pass through GRL
            z_reversed = self.grl(z)
            
            # Conditional discrimination loss
            if species_targets is not None and local_specimen_targets is not None:
                adv_cond_loss, n_valid = self.cond_discriminator(
                    z_reversed, species_targets, local_specimen_targets
                )
                out["loss_adv_cond"] = adv_cond_loss.view(1) if adv_cond_loss.dim() == 0 else adv_cond_loss
                out["n_valid_cond"] = torch.tensor([n_valid], dtype=torch.long, device=x.device)
                
            # Unconditional DANN discrimination loss
            if global_specimen_targets is not None:
                uncond_loss = self.uncond_discriminator(
                    z_reversed, global_specimen_targets
                )
                out["loss_adv_uncond"] = uncond_loss.view(1) if uncond_loss.dim() == 0 else uncond_loss
                
            # CLUB upper bound
            if global_specimen_targets is not None:
                club_mi = self.club(z, global_specimen_targets)
                club_var = self.club.loglikeli(z, global_specimen_targets)
                out["club_mi_bound"] = club_mi.view(1) if club_mi.dim() == 0 else club_mi
                out["club_var_loss"] = club_var.view(1) if club_var.dim() == 0 else club_var
                
        return out
