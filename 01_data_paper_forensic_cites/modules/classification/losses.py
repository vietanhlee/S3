"""
01_data_article.modules.classification.losses
=============================================
Định nghĩa các hàm mất mát phân loại cho Technical Validation:
  1. Standard Cross-Entropy Baseline
  2. Multiclass Focal Loss (Lin et al., ICCV 2017)
  3. Class-Balanced Focal Loss (Cui et al., CVPR 2019) dựa trên số lượng mẫu hiệu dụng (Effective Number of Samples).
"""

from typing import Tuple, List, Optional
import numpy as np
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


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Multiclass Focal Loss (Cui et al., CVPR 2019).
    Sử dụng số lượng mẫu hiệu dụng E_n = (1 - beta^n) / (1 - beta) để tái cân bằng độ dốc gradient,
    đặc biệt ngăn ngừa sự sụp đổ Recall ở các loài thiểu số bị hạn chế mẫu vật lý (e.g. Afzelia pachyloba).

    Trọng số lớp:
        w_c = (1 / E_{N_c}) / sum_j(1 / E_{N_j}) * C
    CB-FL(p_t) = -w_c * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        samples_arr = np.array(samples_per_class, dtype=np.float64)
        num_classes = len(samples_per_class)

        # Tính số lượng mẫu hiệu dụng
        effective_num = 1.0 - np.power(beta, samples_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Chuẩn hóa tổng trọng số bằng số lớp C
        weights = weights / np.sum(weights) * num_classes

        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        weights = self.class_weights.to(logits.device)[targets]
        cb_loss = weights * focal_loss

        if self.reduction == "mean":
            return cb_loss.mean()
        elif self.reduction == "sum":
            return cb_loss.sum()
        return cb_loss


class ClassWeightedCrossEntropyLoss(nn.Module):
    """
    Inverse-Frequency Class-Weighted Cross-Entropy Loss:
    w_c = (N_total / (C * N_c)), normalized so sum(w_c) = C.
    Trọng số nghịch đảo tần suất lớp giúp tái cân bằng cập nhật gradient,
    giải quyết triệt để hiện tượng sụp đổ recall ở các loài thiểu số (như Afzelia pachyloba).
    """
    def __init__(self, samples_per_class: List[int], reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        samples_arr = np.array(samples_per_class, dtype=np.float64)
        num_classes = len(samples_per_class)
        inv_freq = 1.0 / np.maximum(samples_arr, 1.0)
        weights = inv_freq / np.sum(inv_freq) * num_classes
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights.to(logits.device), reduction=self.reduction)


def build_criterion(
    loss_type: str,
    alpha: float = 0.25,
    gamma: float = 2.0,
    samples_per_class: Optional[List[int]] = None,
    beta: float = 0.999
) -> Tuple[nn.Module, str]:
    """
    Khởi tạo hàm mất mát chuẩn hóa:
      1. 'cross_entropy' (hoặc 'ce'): Standard Cross-Entropy Baseline
      2. 'focal': Multiclass Focal Loss (alpha=0.25, gamma=2.0)
      3. 'cb_focal': Class-Balanced Focal Loss (beta=0.999, gamma=2.0, Cui et al. CVPR 2019)
      4. 'class_weighted' (hoặc 'class_weighted_ce', 'weighted_ce'): Inverse-Frequency Class-Weighted Cross-Entropy
    """
    loss_key = loss_type.lower().strip()
    if loss_key in ["cross_entropy", "ce", "crossentropy", "standard"]:
        desc = "Standard Cross-Entropy Baseline"
        return nn.CrossEntropyLoss(), desc
    elif loss_key in ["focal", "focal_loss"]:
        desc = f"Multiclass Focal Loss (alpha={alpha}, gamma={gamma})"
        return MulticlassFocalLoss(gamma=gamma, alpha=alpha), desc
    elif loss_key in ["class_weighted", "class_weighted_ce", "weighted_ce", "cw_ce"]:
        if samples_per_class is None:
            raise ValueError("Cần cung cấp 'samples_per_class' để khởi tạo ClassWeightedCrossEntropyLoss.")
        desc = "Inverse-Frequency Class-Weighted Cross-Entropy"
        return ClassWeightedCrossEntropyLoss(samples_per_class=samples_per_class), desc
    elif loss_key in ["cb_focal", "class_balanced_focal", "cbfocal"]:
        if samples_per_class is None:
            raise ValueError("Cần cung cấp 'samples_per_class' để khởi tạo ClassBalancedFocalLoss.")
        desc = f"Class-Balanced Focal Loss (beta={beta}, gamma={gamma})"
        return ClassBalancedFocalLoss(samples_per_class=samples_per_class, beta=beta, gamma=gamma), desc
    else:
        raise ValueError(
            f"Không hỗ trợ loss '{loss_type}'. Vui lòng chọn 'focal', 'cross_entropy', 'class_weighted', hoặc 'cb_focal'."
        )


