"""
train_soft_triple.py — SoftTriple Loss (ICCV 2019)
==================================================
Gán nhiều center cho mỗi lớp để học phân bố đa đỉnh (multi-modal) của vân gỗ.
Tự động phát hiện và mô hình hóa các sub-cluster (ví dụ: gỗ lõi, gỗ giác, vùng chuyển tiếp).

Các cải tiến & Tối ưu hóa siêu tham số:
1. NUM_CENTERS = 3 (Giảm từ 10 xuống 3): Khắc phục triệt để "Bệnh Phân mảnh Tâm"
   (Representation Fragmentation). Đưa số center về 3 đại diện chuẩn cho 3 dạng ngoại hình/thớ cắt
   chính của gỗ macro (mặt ngang, cắt xuyên tâm, và biến thể sắc thái).
2. SOFTTRIPLE_TAU = 0.15 (Giảm nhẹ margin từ 0.20 xuống 0.15): Tránh phạt quá đà các loài gỗ
   có độ tương đồng liên loài cao thuộc cùng một Chi (*Afzelia*, *Dalbergia*).
3. Center Regularization Term (R_center): Bổ sung phạt tương đồng giữa các sub-centers của cùng 1 loài,
   đảm bảo K centers không bị sụp đổ (collapse) về 1 điểm hoặc trôi dạt tự do.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_soft_triple",
	"NUM_CENTERS": 3,              # Giảm từ 10 xuống 3 để phù hợp với quy mô mẫu và các dạng thớ cắt gỗ
	"SOFTTRIPLE_LAMBDA": 20.0,     # Hệ số scale lambda (la)
	"SOFTTRIPLE_GAMMA": 0.1,       # Tham số entropy gamma điều tiết độ tương đồng
	"SOFTTRIPLE_TAU": 0.15,        # Margin delta (tau = 0.15) cho ranh giới phân tách mềm
	"SOFTTRIPLE_REG_WEIGHT": 0.01, # Trọng số Regularization phân tách các centers cùng lớp
	"FOCAL_GAMMA": 2.0,            # Siêu tham số gamma cho Focal Loss
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"WEIGHT_DECAY": 1e-4,
}
# =====================


class FocalLoss(nn.Module):
	"""Focal Loss có gán trọng số cân bằng lớp alpha (class_weights)."""

	def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None) -> None:
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce = F.cross_entropy(logits, targets, reduction="none")
		pt = torch.exp(-ce)
		if self.alpha is not None:
			alpha_t = self.alpha[targets]
		else:
			alpha_t = 1.0
		loss = alpha_t * (1.0 - pt) ** self.gamma * ce
		return loss.mean()


class SoftTripleLoss(nn.Module):
	"""SoftTriple Loss — Tích hợp nhiều centers trên một lớp với soft assignment và Center Regularization."""

	def __init__(self, num_classes: int, embedding_dim: int = 256,
	             num_centers: int = 3, la: float = 20.0,
	             gamma: float = 0.1, margin: float = 0.15, reg_weight: float = 0.01,
	             focal_gamma: float = 2.0, class_weights: torch.Tensor = None) -> None:
		super().__init__()
		self.num_classes = num_classes
		self.num_centers = num_centers
		self.la = la
		self.gamma = gamma
		self.margin = margin
		self.reg_weight = reg_weight
		self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)

		# Trọng số đại diện các centers: (C, K, D)
		self.fc = nn.Parameter(torch.Tensor(num_classes, num_centers, embedding_dim))
		nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# embeddings: shape (B, D)
		# Chuẩn hóa centers dọc theo chiều đặc trưng D
		centers = F.normalize(self.fc, p=2, dim=2)  # (C, K, D)

		# Cosine similarity giữa từng embedding trong batch và tất cả centers của tất cả các lớp
		emb_expanded = embeddings.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, D)
		centers_expanded = centers.unsqueeze(0)               # (1, C, K, D)

		# similarity matrix: (B, C, K)
		sim = torch.sum(emb_expanded * centers_expanded, dim=-1)

		# Tính xác suất gán mềm (soft assignment) của mẫu vào từng center trong lớp
		prob = F.softmax(sim / self.gamma, dim=2)  # (B, C, K)

		# Gộp độ tương đồng của các centers cho từng lớp: S_{i,c} = sum_k prob_k * sim_k
		S = torch.sum(prob * sim, dim=2)  # (B, C)

		# Áp dụng margin delta cho lớp mục tiêu (chính xác)
		one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
		S_margin = S - one_hot * self.margin

		# Loss chính kết hợp Focal Loss
		main_loss = self.focal(self.la * S_margin, labels)

		# Center Regularization: phạt sự trùng lặp giữa các sub-centers của cùng một lớp
		reg_loss = 0.0
		if self.num_centers > 1:
			# Pairwise similarity giữa K centers trong từng class: (C, K, K)
			center_sim = torch.matmul(centers, centers.transpose(1, 2))
			eye = torch.eye(self.num_centers, device=embeddings.device).unsqueeze(0)
			off_diag_sim = center_sim * (1.0 - eye)
			# Chỉ phạt phần similarity dương giữa các sub-centers cùng lớp
			reg_loss = torch.mean(torch.clamp(off_diag_sim, min=0.0))

		return main_loss + self.reg_weight * reg_loss


class SoftTripleTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "SoftTriple Loss"

	def get_loss_config(self) -> dict:
		return {
			"num_centers": self.config["NUM_CENTERS"],
			"softtriple_lambda": self.config["SOFTTRIPLE_LAMBDA"],
			"softtriple_gamma": self.config["SOFTTRIPLE_GAMMA"],
			"softtriple_margin": self.config["SOFTTRIPLE_TAU"],
			"softtriple_reg_weight": self.config.get("SOFTTRIPLE_REG_WEIGHT", 0.01),
			"focal_gamma": self.config["FOCAL_GAMMA"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		class_weights = getattr(self, "class_weights", None)
		return SoftTripleLoss(
			num_classes=num_classes,
			embedding_dim=self.config["EMBEDDING_DIM"],
			num_centers=self.config["NUM_CENTERS"],
			la=self.config["SOFTTRIPLE_LAMBDA"],
			gamma=self.config["SOFTTRIPLE_GAMMA"],
			margin=self.config["SOFTTRIPLE_TAU"],
			reg_weight=self.config.get("SOFTTRIPLE_REG_WEIGHT", 0.01),
			focal_gamma=self.config["FOCAL_GAMMA"],
			class_weights=class_weights,
		)

	def build_train_sampler(self, labels: list) -> Sampler | None:
		# SoftTriple là classification-based loss, không dùng PK Sampler mà dùng RandomSampler thông thường
		return None


if __name__ == "__main__":
	trainer = SoftTripleTrainer(CONFIG)
	trainer.run()
