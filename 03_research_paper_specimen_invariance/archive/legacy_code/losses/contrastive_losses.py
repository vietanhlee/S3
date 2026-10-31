"""
specimen_invariance_framework/losses/contrastive_losses.py
==========================================================
Representation learning losses:
- Supervised Contrastive Loss (SupCon)
- Semi-Hard Negative Triplet Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls together embeddings of the same species while repelling differing species.
    """
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized representations (B, D)
            labels: Ground truth species labels (B,)
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Dot product similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1.0)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class SemiHardTripletLoss(nn.Module):
    """
    Online Semi-Hard Triplet Loss on unit hypersphere.
    L = max(0, ||z_a - z_p||_2^2 - ||z_a - z_n||_2^2 + alpha)
    """
    def __init__(self, margin: float = 0.50):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized vectors (B, D)
            targets: Class targets (B,)
        """
        device = embeddings.device
        pdist = torch.cdist(embeddings, embeddings, p=2)  # Euclidean distance matrix (B, B)
        
        targets = targets.view(-1, 1)
        pos_mask = torch.eq(targets, targets.T)
        neg_mask = ~pos_mask
        
        # Exclude self-comparisons
        pos_mask.fill_diagonal_(False)
        
        triplet_losses = []
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            pos_indices = torch.where(pos_mask[i])[0]
            neg_indices = torch.where(neg_mask[i])[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
                
            d_ap = pdist[i, pos_indices]  # Distances to positives
            d_an = pdist[i, neg_indices]  # Distances to negatives
            
            # Semi-hard negatives: d_ap < d_an < d_ap + margin
            for d_p in d_ap:
                semi_hard = neg_indices[(d_an > d_p) & (d_an < d_p + self.margin)]
                if len(semi_hard) > 0:
                    d_n = pdist[i, semi_hard].min()
                else:
                    # Fallback to hardest negative
                    d_n = d_an.min()
                    
                loss_val = F.relu(d_p - d_n + self.margin)
                triplet_losses.append(loss_val)
                
        if len(triplet_losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return torch.stack(triplet_losses).mean()
