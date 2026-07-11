"""
train_soft_triple.py — SoftTriple Loss (ICCV 2019)
==================================================
Gán nhiều center cho mỗi lớp để học phân bố đa đỉnh (multi-modal) của vân gỗ.
Tự động phát hiện và mô hình hóa các sub-cluster (ví dụ: gỗ lõi, gỗ giác, vùng chuyển tiếp).
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
	"NUM_CENTERS": 10,             # Số center cho mỗi lớp (K)
	"SOFTTRIPLE_LAMBDA": 20.0,     # Hệ số scale lambda (la)
	"SOFTTRIPLE_GAMMA": 0.1,       # Tham số entropy gamma điều tiết độ tương đồng
	"SOFTTRIPLE_TAU": 0.2,         # Margin delta (tau / margin)
	"EPOCHS": 50,
	"PATIENCE": 10,
	"LR": 1e-4,
	"WEIGHT_DECAY": 1e-4,
}
# =====================


class SoftTripleLoss(nn.Module):
	"""SoftTriple Loss — Tích hợp nhiều centers trên một lớp với soft assignment."""

	def __init__(self, num_classes: int, embedding_dim: int = 256,
	             num_centers: int = 10, la: float = 20.0,
	             gamma: float = 0.1, margin: float = 0.2) -> None:
		super().__init__()
		self.num_classes = num_classes
		self.num_centers = num_centers
		self.la = la
		self.gamma = gamma
		self.margin = margin

		# Trọng số đại diện các centers: (C, K, D)
		self.fc = nn.Parameter(torch.Tensor(num_classes, num_centers, embedding_dim))
		nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# embeddings: shape (B, D)
		# Chuẩn hóa centers dọc theo chiều đặc trưng D
		centers = F.normalize(self.fc, p=2, dim=2)  # (C, K, D)

		# Cosine similarity giữa từng embedding trong batch và tất cả centers của tất cả các lớp
		# embedding_expanded: (B, 1, 1, D)
		# centers_expanded: (1, C, K, D)
		emb_expanded = embeddings.unsqueeze(1).unsqueeze(2)
		centers_expanded = centers.unsqueeze(0)

		# similarity matrix: (B, C, K)
		sim = torch.sum(emb_expanded * centers_expanded, dim=-1)

		# Tính xác suất gán mềm (soft assignment) của mẫu vào từng center trong lớp
		# softmax theo chiều K centers
		prob = F.softmax(sim / self.gamma, dim=2)  # (B, C, K)

		# Gộp độ tương đồng của các centers cho từng lớp: S_{i,c} = sum_k prob_k * sim_k
		S = torch.sum(prob * sim, dim=2)  # (B, C)

		# Áp dụng margin delta cho lớp mục tiêu (chính xác)
		one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
		S_margin = S - one_hot * self.margin

		# Tính loss dựa trên Cross Entropy có tỷ lệ scale lambda (la)
		loss = F.cross_entropy(self.la * S_margin, labels)
		return loss


class SoftTripleTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "SoftTriple Loss"

	def get_loss_config(self) -> dict:
		return {
			"num_centers": self.config["NUM_CENTERS"],
			"softtriple_lambda": self.config["SOFTTRIPLE_LAMBDA"],
			"softtriple_gamma": self.config["SOFTTRIPLE_GAMMA"],
			"softtriple_margin": self.config["SOFTTRIPLE_TAU"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return SoftTripleLoss(
			num_classes=num_classes,
			embedding_dim=self.config["EMBEDDING_DIM"],
			num_centers=self.config["NUM_CENTERS"],
			la=self.config["SOFTTRIPLE_LAMBDA"],
			gamma=self.config["SOFTTRIPLE_GAMMA"],
			margin=self.config["SOFTTRIPLE_TAU"],
		)

	def build_train_sampler(self, labels: list) -> Sampler | None:
		# SoftTriple là classification-based loss, không dùng PK Sampler mà dùng RandomSampler thông thường
		return None


if __name__ == "__main__":
	trainer = SoftTripleTrainer(CONFIG)
	trainer.run()
