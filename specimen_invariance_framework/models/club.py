"""
specimen_invariance_framework/models/club.py
===========================================
Contrastive Log-ratio Upper Bound (CLUB) mutual information estimator.
Reference: Cheng et al., ICML 2020.
Used as an alternative non-adversarial baseline to directly minimize I(Z; P | Y).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLUBDiscrete(nn.Module):
    """
    vCLUB Mutual Information Upper Bound for discrete categorical variables.
    
    Given representations z and discrete specimen targets p:
    q_theta(p | z) is parameterized by a lightweight variational network.
    
    CLUB upper bound:
        I_vCLUB(Z; P) = 1/N sum_i [ log q(p_i | z_i) - 1/N sum_j log q(p_j | z_i) ]
    """
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.variational_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the CLUB upper bound on I(Z; P).
        Minimizing this bound reduces mutual information between embedding and specimen.
        """
        logits = self.variational_net(z)  # (N, C)
        log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
        
        # Positive log-likelihood: log q(p_i | z_i)
        pos = log_probs.gather(1, p.view(-1, 1)).squeeze(1)  # (N,)
        
        # Negative marginal log-likelihood: 1/N sum_j log q(p_j | z_i)
        # We compute cross pairs: log_probs is (N, C), p is (N,)
        # For each sample i, gather probabilities of all other samples' labels p:
        # p.unsqueeze(0) is (1, N) -> expand to (N, N)
        p_expanded = p.unsqueeze(0).expand(z.size(0), -1)  # (N, N)
        neg_matrix = log_probs.gather(1, p_expanded)  # (N, N)
        neg = neg_matrix.mean(dim=1)  # (N,)
        
        club_mi = (pos - neg).mean()
        return torch.clamp(club_mi, min=0.0)

    def loglikeli(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Trains the variational network q_theta to fit p given z."""
        logits = self.variational_net(z.detach())
        return F.cross_entropy(logits, p)
