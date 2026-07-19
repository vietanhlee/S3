"""
train_soft_margin_triplet.py — Soft-Margin Triplet Loss
=======================================================
Thay hàm hinge cứng bằng hàm log-entropy mịn, loại bỏ hoàn toàn
siêu tham số margin α, gradient liên tục không bao giờ bị triệt tiêu.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_soft_margin_triplet",
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class SoftMarginTripletLoss(nn.Module):
	"""Soft-Margin Triplet Loss — gradient liên tục, không cần margin.

	L = log(1 + exp(d(a,p)^2 - d(a,n)^2))
	"""

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)
		batch_size = embeddings.size(0)

		labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
		diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
		pos_mask = labels_eq & ~diag
		neg_mask = ~labels_eq

		losses = []
		for i in range(batch_size):
			pos_indices = torch.where(pos_mask[i])[0]
			neg_indices = torch.where(neg_mask[i])[0]

			if len(pos_indices) == 0 or len(neg_indices) == 0:
				continue

			# Hardest positive và hardest negative
			d_ap = dist_mat[i, pos_indices].max()
			d_an = dist_mat[i, neg_indices].min()

			# Soft margin: log(1 + exp(d_ap - d_an))
			loss = torch.log1p(torch.exp(d_ap - d_an))
			losses.append(loss)

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean()


class SoftMarginTripletTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Soft-Margin Triplet Loss"

	def get_loss_config(self) -> dict:
		return {"note": "No margin parameter needed"}

	def build_loss(self, num_classes: int) -> nn.Module:
		return SoftMarginTripletLoss()


if __name__ == "__main__":
	trainer = SoftMarginTripletTrainer(CONFIG)
	trainer.run()
