"""
train_contrastive.py — Vanilla Contrastive Loss (Baseline, CVPR 2005)
=====================================================================
So sánh từng cặp mẫu với margin cố định.
Baseline pair-based cơ bản nhất để đo mức cải thiện của Circle Loss, MS Loss.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_contrastive",
	"CONTRASTIVE_MARGIN": 0.5,
	"EPOCHS": 30,
	"PATIENCE": 10,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class VanillaContrastiveLoss(nn.Module):
	"""Vanilla Contrastive Loss — margin cố định, kéo positive, đẩy negative."""

	def __init__(self, margin: float = 0.5) -> None:
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_mat = torch.cdist(embeddings, embeddings, p=2)
		batch_size = embeddings.size(0)

		labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
		diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
		pos_mask = labels_eq & ~diag
		neg_mask = ~labels_eq

		# Positive loss: d(xi, xj)^2
		pos_loss = dist_mat.pow(2) * pos_mask.float()

		# Negative loss: max(0, margin - d(xi, xj))^2
		neg_loss = torch.clamp(self.margin - dist_mat, min=0.0).pow(2) * neg_mask.float()

		# Trung bình
		n_pos = pos_mask.float().sum()
		n_neg = neg_mask.float().sum()

		loss = 0.0
		if n_pos > 0:
			loss = loss + pos_loss.sum() / n_pos
		if n_neg > 0:
			loss = loss + neg_loss.sum() / n_neg

		if isinstance(loss, float):
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return loss * 0.5


class VanillaContrastiveTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Vanilla Contrastive Loss"

	def get_loss_config(self) -> dict:
		return {"contrastive_margin": self.config["CONTRASTIVE_MARGIN"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return VanillaContrastiveLoss(margin=self.config["CONTRASTIVE_MARGIN"])


if __name__ == "__main__":
	trainer = VanillaContrastiveTrainer(CONFIG)
	trainer.run()
