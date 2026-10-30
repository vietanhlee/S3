"""
specimen_invariance_framework/models/grl.py
===========================================
Gradient Reversal Layer (GRL) implementation with dynamic annealing schedule.
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Autograd function that acts as identity during forward propagation
    and scales/reverses the gradient during backpropagation by -lambda_adv.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse gradient
        return -ctx.lambda_adv * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer with dynamically updated lambda_adv.
    
    Schedule:
        lambda_adv(p) = (2.0 / (1.0 + exp(-gamma * p)) - 1.0) * max_lambda
    where p in [0, 1] is training progress (current_step / total_steps).
    """
    def __init__(self, gamma: float = 10.0, max_lambda: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.max_lambda = max_lambda
        self.lambda_adv = 0.0

    def update_lambda(self, progress: float) -> float:
        """
        Update lambda_adv according to training progress p in [0, 1].
        """
        p = max(0.0, min(1.0, progress))
        self.lambda_adv = (2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0) * self.max_lambda
        return self.lambda_adv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_adv)
