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


def build_backbone(name: str = "resnet18", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Constructs a visual feature extractor backbone consistently using torchvision.
    Supports lightweight models: ResNet-18 (default), MobileNetV2, ResNet-50, and ConvNeXt-Tiny.
    """
    name = name.lower()
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return BackboneWrapper(base, feat_dim), feat_dim
    elif name in ["mobilenet_v2", "mobilenet"]:
        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = tv_models.mobilenet_v2(weights=weights)
        feat_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
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


class SpeciesConditionedSpecimenDiscriminator(nn.Module):
    """
    Minimax Adversarial Specimen Discriminator with Species Masked Softmax (Proposition 1 & Theorem 1).
    Receives features through GRL and attempts to classify physical specimen identity
    ONLY within the co-species voucher candidates:
    P(s_i | y=c, z) = exp(w_{c,i}^T z) * M_{c,i} / sum_j [exp(w_{c,j}^T z) * M_{c,j}]
    """

    def __init__(
        self,
        in_features: int,
        num_specimens: int = 44,
        num_classes: int = 11,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        species_specimen_mask: Optional[torch.Tensor] = None,
    ):
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
        self.num_classes = num_classes
        self.num_specimens = num_specimens

        if species_specimen_mask is not None:
            self.register_buffer("mask", species_specimen_mask.clone().float())
        else:
            self.register_buffer("mask", torch.ones(num_classes, num_specimens, dtype=torch.float32))

    def set_mask(self, mask: torch.Tensor):
        self.mask.copy_(mask.float())

    def set_alpha(self, alpha: float):
        self.grl.set_alpha(alpha)

    def forward(self, features: torch.Tensor, species_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        reversed_features = self.grl(features)
        raw_logits = self.net(reversed_features)
        if species_labels is not None:
            batch_mask = self.mask[species_labels]
            masked_logits = raw_logits + (1.0 - batch_mask) * -1e9
            return masked_logits
        return raw_logits


# Backward-compatible alias
SpecimenDiscriminator = SpeciesConditionedSpecimenDiscriminator


class VariationalCLUB(nn.Module):
    """
    Vectorized Variational Contrastive Log-ratio Upper Bound (vCLUB) Estimator.
    Minimizes the mutual information upper bound I(Z; S | Y) to compress residual specimen variance.
    100% Vectorized GPU matrix operations (Zero Python for-loops, Smooth gradients).
    """

    def __init__(self, feat_dim: int, num_specimens: int = 44, hidden_dim: int = 128, specimen_emb_dim: int = 32):
        super().__init__()
        self.specimen_embed = nn.Embedding(num_specimens, specimen_emb_dim)
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

    def compute_mi_loss(self, z: torch.Tensor, s_labels: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the contrastive mutual information upper bound with intra-taxon negative pairs.
        Vectorized without Python loops.
        """
        batch_size = z.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        mu = self.p_mu(z)              # [B, d]
        logvar = self.p_logvar(z)      # [B, d]
        var = logvar.exp() + 1e-6      # [B, d]
        s_emb = self.specimen_embed(s_labels) # [B, d]

        # 1. Positive log prob: log q(s_i | z_i)
        pos_log_prob = -0.5 * (((mu - s_emb) ** 2) / var + logvar).sum(dim=-1) # [B]

        # 2. Pairwise log prob: log q(s_j | z_i) [B, B]
        mu_exp = mu.unsqueeze(1)         # [B, 1, d]
        var_exp = var.unsqueeze(1)       # [B, 1, d]
        logvar_exp = logvar.unsqueeze(1) # [B, 1, d]
        s_emb_exp = s_emb.unsqueeze(0)   # [1, B, d]

        pairwise_log_prob = -0.5 * (((mu_exp - s_emb_exp) ** 2) / var_exp + logvar_exp).sum(dim=-1) # [B, B]

        # Intra-taxon mask: same species (y_i == y_j) but different specimen (s_i != s_j)
        same_species = (y_labels.unsqueeze(1) == y_labels.unsqueeze(0))
        diff_specimen = (s_labels.unsqueeze(1) != s_labels.unsqueeze(0))
        intra_neg_mask = same_species & diff_specimen

        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        general_neg_mask = ~diag_mask

        has_intra_neg = intra_neg_mask.any(dim=1, keepdim=True)
        final_neg_mask = torch.where(has_intra_neg, intra_neg_mask, general_neg_mask).float()

        mask_sums = final_neg_mask.sum(dim=1).clamp(min=1.0)
        neg_log_prob = (pairwise_log_prob * final_neg_mask).sum(dim=1) / mask_sums

        mi_bound = (pos_log_prob - neg_log_prob).mean()
        return mi_bound


class StandardBaselineModel(nn.Module):
    """
    Standard Deep Learning Baseline Model (ERM).
    Extracts features and directly predicts species without any invariance mechanism.
    """

    def __init__(self, backbone_name: str = "resnet18", num_classes: int = 11, pretrained: bool = True):
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
    3. Species-Conditioned Specimen Discriminator D_phi with Gradient Reversal Layer (GRL) & Masked Softmax
    4. Vectorized Variational CLUB Mutual Information Bottleneck
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 11,
        num_specimens: int = 44,
        pretrained: bool = True,
        adv_weight: float = 0.30,
        club_weight: float = 0.010,
        species_specimen_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.classifier = SpeciesClassifier(self.feat_dim, num_classes=num_classes)
        self.discriminator = SpeciesConditionedSpecimenDiscriminator(
            in_features=self.feat_dim,
            num_specimens=num_specimens,
            num_classes=num_classes,
            species_specimen_mask=species_specimen_mask,
        )
        self.club = VariationalCLUB(self.feat_dim, num_specimens=num_specimens)

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
        specimen_logits = self.discriminator(features, species_labels=species_labels)

        out = {
            "species_logits": species_logits,
            "specimen_logits": specimen_logits,
            "features": features,
        }

        if specimen_labels is not None and species_labels is not None:
            out["club_loss"] = self.club.compute_mi_loss(features, specimen_labels, species_labels)

        return out
