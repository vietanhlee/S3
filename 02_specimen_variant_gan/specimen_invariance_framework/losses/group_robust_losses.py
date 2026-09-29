"""
specimen_invariance_framework/losses/group_robust_losses.py
===========================================================
Group Robustness baselines:
1. GroupDRO: Group Distributionally Robust Optimization (group = physical specimen).
   Reference: Sagawa et al., ICLR 2020.
2. IRM: Invariant Risk Minimization (environment = physical specimen).
   Reference: Arjovsky et al., 2019.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupDROLoss(nn.Module):
    """
    GroupDRO where each group corresponds to a distinct physical specimen.
    Optimizes for the worst-case specimen loss via online exponentiated gradient updates.
    """
    def __init__(self, num_groups: int = 148, step_size: float = 0.01):
        super().__init__()
        self.num_groups = num_groups
        self.step_size = step_size
        # Uniform group weight distribution
        self.register_buffer("q", torch.ones(num_groups) / num_groups)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, group_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B,)
            group_indices: (B,) global specimen IDs in [0, num_groups-1]
        """
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
        unique_groups = torch.unique(group_indices)
        
        group_losses = []
        group_weights = []
        
        for g in unique_groups:
            mask = (group_indices == g)
            g_loss = per_sample_loss[mask].mean()
            group_losses.append(g_loss)
            group_weights.append(self.q[g])
            
            # Exponentiated gradient step on q
            with torch.no_grad():
                self.q[g] = self.q[g] * torch.exp(self.step_size * g_loss.detach())
                
        # Renormalize q buffer
        with torch.no_grad():
            self.q.div_(self.q.sum() + 1e-12)
            
        group_losses_t = torch.stack(group_losses)
        group_weights_t = torch.stack(group_weights)
        group_weights_t = group_weights_t / (group_weights_t.sum() + 1e-12)
        
        return (group_losses_t * group_weights_t).sum()


class IRMLoss(nn.Module):
    """
    Invariant Risk Minimization (IRMv1).
    Penalizes the gradient norm of a scalar dummy multiplier w=1.0 per specimen environment.
    """
    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, env_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (B, C)
            targets: (B,)
            env_indices: (B,) specimen IDs representing environments
        """
        device = logits.device
        unique_envs = torch.unique(env_indices)
        
        erm_losses = []
        penalties = []
        
        # Scalar dummy parameter
        dummy_w = torch.tensor(1.0, device=device, requires_grad=True)
        
        for env in unique_envs:
            mask = (env_indices == env)
            env_logits = logits[mask] * dummy_w
            env_targets = targets[mask]
            
            if env_targets.size(0) < 2:
                continue
                
            env_loss = F.cross_entropy(env_logits, env_targets)
            erm_losses.append(env_loss)
            
            # Compute gradient norm with respect to dummy_w
            grad = torch.autograd.grad(env_loss, [dummy_w], create_graph=True)[0]
            penalty = torch.sum(grad ** 2)
            penalties.append(penalty)
            
        if len(erm_losses) == 0:
            total_erm = F.cross_entropy(logits, targets)
            return total_erm, torch.tensor(0.0, device=device)
            
        total_erm = torch.stack(erm_losses).mean()
        total_penalty = torch.stack(penalties).mean() if len(penalties) > 0 else torch.tensor(0.0, device=device)
        
        total_loss = total_erm + self.penalty_weight * total_penalty
        return total_loss, total_penalty
