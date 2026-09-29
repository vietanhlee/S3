"""
01_data_article.modules.classification.losses
=============================================
Định nghĩa hàm mất mát Multiclass Focal Loss và hàm xây dựng loss tiêu chuẩn cho Technical Validation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassFocalLoss(nn.Module):
    """
    Multiclass Focal Loss giải quyết mất cân bằng mẫu tự nhiên giữa các loài gỗ (Section 4.2).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Mặc định: alpha = 0.25, gamma = 2.0
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_criterion(loss_type: str, alpha: float = 0.25, gamma: float = 2.0) -> Tuple[nn.Module, str]:
    """
    Khởi tạo hàm mất mát chuẩn hóa:
      1. 'cross_entropy' (hoặc 'ce'): Standard Cross-Entropy Baseline
      2. 'focal': Multiclass Focal Loss (alpha=0.25, gamma=2.0)
    """
    loss_key = loss_type.lower().strip()
    if loss_key in ["cross_entropy", "ce", "crossentropy", "standard"]:
        desc = "Standard Cross-Entropy Baseline"
        return nn.CrossEntropyLoss(), desc
    elif loss_key in ["focal", "focal_loss"]:
        desc = f"Multiclass Focal Loss (alpha={alpha}, gamma={gamma})"
        return MulticlassFocalLoss(gamma=gamma, alpha=alpha), desc
    else:
        raise ValueError(f"Không hỗ trợ loss '{loss_type}'. Vui lòng chọn 'focal' hoặc 'cross_entropy'.")
