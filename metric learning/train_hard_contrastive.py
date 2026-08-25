"""
train_hard_contrastive.py — Online Top-K Hard Pair Contrastive Loss
=====================================================================
Khai thác Top-K cặp positive xa nhất và negative gần nhất với Dual Margin.
Giải quyết triệt để tình trạng nhạy cảm với nhiễu (outliers) và tối ưu hóa
cho các loài gỗ khó có intra-class variance lớn.
"""

import torch
import torch.nn as nn
from train_base import BaseMetricTrainer

# ===== CẤU HÌNH =====
CONFIG = {
	"OUTPUT_DIR": "outputs_hard_contrastive",
	"CONTRASTIVE_MARGIN": 1.1,  # Negative margin khoảng cách Euclidean (phạt khi cosine sim > 0.395)
	"POS_MARGIN": 0.2,          # Positive margin (chỉ kéo khi d > 0.2)
	"TOP_K_HARD": 3,            # Khai thác Top-3 cặp positive xa nhất và negative gần nhất per anchor
	"EPOCHS": 50,
	"PATIENCE": 25,
	"LR": 1e-4,
	"P_CLASSES": 19,
	"K_SAMPLES": 20,
}
# =====================


class OnlineHardContrastiveLoss(nn.Module):
	"""Online Top-K Hard Pair Contrastive Loss với Dual Margin."""

	def __init__(self, neg_margin: float = 1.1, pos_margin: float = 0.2, top_k: int = 3) -> None:
		super().__init__()
		self.neg_margin = neg_margin
		self.pos_margin = pos_margin
		self.top_k = top_k

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

			# Hardest positive: Top-K xa nhất
			pos_dists = dist_mat[i, pos_indices]
			if len(pos_dists) > self.top_k:
				hard_pos_dists, _ = torch.topk(pos_dists, k=self.top_k, largest=True)
			else:
				hard_pos_dists = pos_dists

			# Hardest negative: Top-K gần nhất
			neg_dists = dist_mat[i, neg_indices]
			if len(neg_dists) > self.top_k:
				hard_neg_dists, _ = torch.topk(neg_dists, k=self.top_k, largest=False)
			else:
				hard_neg_dists = neg_dists

			# Positive loss: chỉ kéo khi d > pos_margin
			pos_loss = torch.clamp(hard_pos_dists - self.pos_margin, min=0.0).pow(2).mean()
			# Negative loss: phạt khi d < neg_margin
			neg_loss = torch.clamp(self.neg_margin - hard_neg_dists, min=0.0).pow(2).mean()

			losses.append(0.5 * (pos_loss + neg_loss))

		if len(losses) == 0:
			return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
		return torch.stack(losses).mean()


class HardContrastiveTrainer(BaseMetricTrainer):

	def get_method_name(self) -> str:
		return "Online Hard Pair Contrastive Loss"

	def get_loss_config(self) -> dict:
		return {
			"contrastive_margin": self.config["CONTRASTIVE_MARGIN"],
			"pos_margin": self.config["POS_MARGIN"],
			"top_k_hard": self.config["TOP_K_HARD"],
		}

	def build_loss(self, num_classes: int) -> nn.Module:
		return OnlineHardContrastiveLoss(
			neg_margin=self.config["CONTRASTIVE_MARGIN"],
			pos_margin=self.config["POS_MARGIN"],
			top_k=self.config["TOP_K_HARD"],
		)


if __name__ == "__main__":
	trainer = HardContrastiveTrainer(CONFIG)
	trainer.run()
