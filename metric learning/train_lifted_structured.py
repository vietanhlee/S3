"""
train_lifted_structured.py — Lifted Structured Loss (CVPR 2016)
===============================================================
Tối ưu hóa cấu trúc toàn batch bằng log-sum-exp.
Nền tảng lý thuyết của MS Loss và Circle Loss.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_lifted_structured",
	"LIFTED_MARGIN": 1.0,
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class LiftedStructuredLoss(nn.Module):
	"""Lifted Structured Loss — tối ưu hóa batch-wide."""

	def __init__(self, margin: float = 1.0) -> None:
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_mat = torch.cdist(embeddings, embeddings, p=2)
		batch_size = embeddings.size(0)

		labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
		diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
		pos_mask = labels_eq & ~diag
		neg_mask = ~labels_eq

		losses = []
		pos_pairs = torch.where(pos_mask)

		for idx in range(len(pos_pairs[0])):
			i, j = pos_pairs[0][idx].item(), pos_pairs[1][idx].item()
			if i >= j:  # Tránh đếm trùng
				continue

			d_pos = dist_mat[i, j]

			# Negative distances cho cả anchor i và positive j
			neg_i = torch.where(neg_mask[i])[0]
			neg_j = torch.where(neg_mask[j])[0]

			if len(neg_i) == 0 or len(neg_j) == 0:
				continue

			# Log-sum-exp smooth max
			neg_term_i = torch.logsumexp(self.margin - dist_mat[i, neg_i], dim=0)
			neg_term_j = torch.logsumexp(self.margin - dist_mat[j, neg_j], dim=0)

			# Max của 2 phía
			neg_term = torch.logaddexp(neg_term_i, neg_term_j)

			loss = torch.clamp(d_pos + neg_term, min=0.0).pow(2)
			losses.append(loss)

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean() * 0.5


class LiftedStructuredTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Lifted Structured Loss"

	def get_loss_config(self) -> dict:
		return {"lifted_margin": self.config["LIFTED_MARGIN"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return LiftedStructuredLoss(margin=self.config["LIFTED_MARGIN"])


if __name__ == "__main__":
	trainer = LiftedStructuredTrainer(CONFIG)
	trainer.run()
