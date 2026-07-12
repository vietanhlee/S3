"""
train_hard_contrastive.py — Online Hard Pair Contrastive Loss
=============================================================
Chỉ tính loss cho cặp positive xa nhất và negative gần nhất trong batch.
Cầu nối giữa Vanilla Contrastive và các loss pair-based nâng cao.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_hard_contrastive",
	"CONTRASTIVE_MARGIN": 0.5,
	"EPOCHS": 60,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class OnlineHardContrastiveLoss(nn.Module):
	"""Online Hard Pair Contrastive Loss — chọn hardest positive và hardest negative."""

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

		losses = []
		for i in range(batch_size):
			pos_indices = torch.where(pos_mask[i])[0]
			neg_indices = torch.where(neg_mask[i])[0]

			if len(pos_indices) == 0 or len(neg_indices) == 0:
				continue

			# Hardest positive: xa nhất
			hardest_pos_dist = dist_mat[i, pos_indices].max()
			# Hardest negative: gần nhất
			hardest_neg_dist = dist_mat[i, neg_indices].min()

			# Positive loss
			pos_loss = hardest_pos_dist.pow(2)
			# Negative loss
			neg_loss = torch.clamp(self.margin - hardest_neg_dist, min=0.0).pow(2)

			losses.append(0.5 * (pos_loss + neg_loss))

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean()


class HardContrastiveTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Online Hard Pair Contrastive Loss"

	def get_loss_config(self) -> dict:
		return {"contrastive_margin": self.config["CONTRASTIVE_MARGIN"]}

	def build_loss(self, num_classes: int) -> nn.Module:
		return OnlineHardContrastiveLoss(margin=self.config["CONTRASTIVE_MARGIN"])


if __name__ == "__main__":
	trainer = HardContrastiveTrainer(CONFIG)
	trainer.run()
