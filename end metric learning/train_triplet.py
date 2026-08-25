"""
train_triplet.py — Vanilla Triplet Loss (Baseline)
===================================================
All-valid-triplet mining trong batch.
Được cải tiến margin khoảng cách bình phương d² từ 0.3 lên 0.5
(tương đương với margin Cosine similarity sp - sn >= 0.25).
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_triplet",
	"TRIPLET_MARGIN": 0.5,  # Tăng margin d² lên 0.5 (tương ứng với Cosine similarity difference >= 0.25)
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class VanillaTripletLoss(nn.Module):
	"""Vanilla Triplet Loss — tính loss trên tất cả bộ ba hợp lệ trong batch."""

	def __init__(self, margin: float = 0.5) -> None:
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		# Tính ma trận khoảng cách bình phương d² = 2 - 2*cos(θ)
		dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)
		batch_size = embeddings.size(0)

		# Tạo mask cho positive và negative
		labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
		labels_neq = ~labels_eq

		# Loại bỏ đường chéo chính khỏi positive
		diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
		pos_mask = labels_eq & ~diag
		neg_mask = labels_neq

		# All-valid triplet mining: (B, B, B)
		ap_dists = dist_mat.unsqueeze(2)  # (B, B, 1)
		an_dists = dist_mat.unsqueeze(1)  # (B, 1, B)

		# Triplet loss: max(0, d(a,p)² - d(a,n)² + margin)
		triplet_loss = ap_dists - an_dists + self.margin  # (B, B, B)

		# Mask: chỉ lấy các triplet hợp lệ
		valid_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)  # (B, B, B)

		# Áp dụng hinge và lọc
		triplet_loss = torch.clamp(triplet_loss, min=0.0)
		triplet_loss = triplet_loss * valid_mask.float()

		# Chỉ tính trung bình trên các triplet có loss > 0 (Active triplets)
		num_positive_triplets = (triplet_loss > 1e-16).float().sum()
		if num_positive_triplets > 0:
			return triplet_loss.sum() / num_positive_triplets
		return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class VanillaTripletTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Vanilla Triplet Loss"

	def get_loss_config(self) -> dict:
		return {"triplet_margin": self.config["TRIPLET_MARGIN"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return VanillaTripletLoss(margin=self.config["TRIPLET_MARGIN"])


if __name__ == "__main__":
	trainer = VanillaTripletTrainer(CONFIG)
	trainer.run()
