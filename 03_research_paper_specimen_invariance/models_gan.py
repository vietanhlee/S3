"""
models_gan.py
=============
Deep neural architectures for:
1. Standard Baseline Model (ConvNeXt-Tiny + Species Classifier without adversarial regularization).
2. Proposed Adversarial Invariant 'GAN' (Feature Extractor + Species Classifier +
   Minimax Specimen Discriminator via Gradient Reversal Layer + CLUB Mutual Information Bottleneck).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class GradientReversalFunction(torch.autograd.Function):
    """
    Standard Gradient Reversal Layer (GRL) autograd function.
    Forward pass: Identity mapping.
    Backward pass: Multiplies incoming gradients by -alpha.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """GRL Module wrapping GradientReversalFunction."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class BackboneWrapper(nn.Module):
    """Ensures backbone output is strictly 2D feature tensor [B, D]."""
    def __init__(self, base: nn.Module, feat_dim: int):
        super().__init__()
        self.base = base
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        return feat


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Constructs a visual feature extractor backbone consistently using torchvision.
    Supports ResNet-50 (default), ResNet-18, and ConvNeXt-Tiny.
    """
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return BackboneWrapper(base, feat_dim), feat_dim
    elif name.startswith("resnet"):
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet50(weights=weights)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return BackboneWrapper(base, feat_dim), feat_dim
    else:
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = tv_models.convnext_tiny(weights=weights)
        feat_dim = base.classifier[2].in_features
        base.classifier[2] = nn.Identity()
        return BackboneWrapper(base, feat_dim), feat_dim


class SpeciesClassifier(nn.Module):
    """Linear species classification head with dropout."""

    def __init__(self, in_features: int, num_classes: int = 11, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SpecimenDiscriminator(nn.Module):
    """
    Minimax Adversarial Specimen Discriminator (The 'GAN' Discriminator).
    Receives features through GRL and attempts to classify physical specimen identity.
    Adversarial reversal forces the Feature Extractor to remove specimen signatures.
    """

    def __init__(self, in_features: int, num_specimens: int = 88, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_specimens),
        )

    def set_alpha(self, alpha: float):
        self.grl.set_alpha(alpha)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reversed_features = self.grl(features)
        return self.net(reversed_features)


class VariationalCLUB(nn.Module):
    """
    Variational Contrastive Log-ratio Upper Bound (vCLUB) Estimator.
    Minimizes the mutual information upper bound I(Z; S) to compress residual specimen variance.
    Uses Intra-Taxon negative sampling to preserve biological species boundaries.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, specimen_emb_dim: int = 32):
        super().__init__()
        self.specimen_embed = nn.Embedding(88, specimen_emb_dim)
        self.p_mu = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, specimen_emb_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, specimen_emb_dim),
            nn.Tanh(),
        )

    def get_log_prob(self, z: torch.Tensor, s_emb: torch.Tensor) -> torch.Tensor:
        mu = self.p_mu(z)
        logvar = self.p_logvar(z)
        return -0.5 * (((mu - s_emb) ** 2) / (logvar.exp() + 1e-6) + logvar).sum(dim=-1)

    def compute_mi_loss(self, z: torch.Tensor, s_labels: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive mutual information upper bound with intra-taxon negative pairs.
        """
        batch_size = z.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        s_emb = self.specimen_embed(s_labels)
        positive_log_prob = self.get_log_prob(z, s_emb)

        # Intra-taxon negative shuffling (j != i, same species if possible)
        negative_indices = torch.randperm(batch_size, device=z.device)
        for i in range(batch_size):
            co_species = (y_labels == y_labels[i]).nonzero(as_tuple=True)[0]
            if len(co_species) > 1:
                cands = co_species[co_species != i]
                if len(cands) > 0:
                    negative_indices[i] = cands[torch.randint(len(cands), (1,)).item()]

        s_neg_emb = s_emb[negative_indices]
        negative_log_prob = self.get_log_prob(z, s_neg_emb)

        mi_bound = (positive_log_prob - negative_log_prob).mean()
        return F.relu(mi_bound)


class StandardBaselineModel(nn.Module):
    """
    Standard Deep Learning Baseline Model (ERM).
    Extracts features and directly predicts species without any invariance mechanism.
    """

    def __init__(self, backbone_name: str = "resnet50", num_classes: int = 11, pretrained: bool = True):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.classifier = SpeciesClassifier(self.feat_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return {
            "species_logits": logits,
            "features": features,
        }


class AdversarialInvariantGAN(nn.Module):
    """
    Proposed Adversarial Invariant 'GAN' Framework.
    Components:
    1. Feature Extractor (Generator/Encoder) E_theta
    2. Species Classifier C_w
    3. Specimen Discriminator D_phi with Gradient Reversal Layer (GRL)
    4. Variational CLUB Mutual Information Bottleneck
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 11,
        num_specimens: int = 88,
        pretrained: bool = True,
        adv_weight: float = 0.05,
        club_weight: float = 0.005,
    ):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.classifier = SpeciesClassifier(self.feat_dim, num_classes=num_classes)
        self.discriminator = SpecimenDiscriminator(self.feat_dim, num_specimens=num_specimens)
        self.club = VariationalCLUB(self.feat_dim)

        self.adv_weight = adv_weight
        self.club_weight = club_weight

    def set_alpha(self, alpha: float):
        """Sets the GRL adversarial gradient reversal magnitude."""
        self.discriminator.set_alpha(alpha)

    def forward(
        self,
        x: torch.Tensor,
        specimen_labels: Optional[torch.Tensor] = None,
        species_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        species_logits = self.classifier(features)
        specimen_logits = self.discriminator(features)

        out = {
            "species_logits": species_logits,
            "specimen_logits": specimen_logits,
            "features": features,
        }

        if specimen_labels is not None and species_labels is not None:
            out["club_loss"] = self.club.compute_mi_loss(features, specimen_labels, species_labels)

        return out
